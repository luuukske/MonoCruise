"""
AEB Thread — Automatic Emergency Braking with arc-based collision detection.

TTB-based detection — see ``core/aeb/AGENTS.md`` §9 for full logic description.

Registry name: ``aeb_thread``
"""

from __future__ import annotations

import enum
import logging
import math
import struct
import mmap
import threading
import time
from dataclasses import dataclass, field

from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry
from core.settings import Settings

from .traffic import (
    Position, Quaternion, Size, Trailer, Vehicle,
    ArcPath, build_arc, arc_arc_collision,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF: float = 1e9

_FULL_BRAKE_DECEL: float = 7.8
_MIN_SPEED_MS: float = 5.0 / 3.6
_MAX_RANGE: float = 100.0
_MAX_RANGE_SQ: float = _MAX_RANGE ** 2

_MIN_ARC_HORIZON: float = 3.0
_MAX_ARC_HORIZON: float = 4.0
_CORRIDOR_MARGIN: float = 0.5
_COLLISION_SAMPLES: int = 48

_WARN_TTB_THRESHOLD: float = 1.3
_BRAKE_TTB_THRESHOLD: float = 0.2
_BRAKE_RELEASE_THRESHOLD: float = 0.3
_TIME_TO_BRAKE_BUFFER: float = 0.0

_STOP_BUFFER_FIXED: float = 1.2
_ARC_START_PCTG: float = 0.2
_RISK_CONFIRM_DURATION: float = 0.1

_REAR_DOT_THRESHOLD: float = -0.5
_OVERTAKE_SPEED_MARGIN: float = 2.0

_ELEVATION_MARGIN_M: float = 5.0

_CROSS_SAFE_ZONE_BASE: float = 0.5
_CROSS_SAFE_ZONE_SPEED: float = 0.5

_EVASION_G_THRESHOLD: float = 0.1 * 9.81
_EVASION_FILTER_MAX_DELTA_KAPPA: float = 0.02

_VEHICLE_FORMAT = "ffffffffffffhhbb"
_TRAILER_FORMAT = "ffffffffff"
_VEHICLE_OBJECT_FORMAT = _VEHICLE_FORMAT + _TRAILER_FORMAT * 3
_TOTAL_FORMAT = "=" + _VEHICLE_OBJECT_FORMAT * 40
_BUF_SIZE = 6960
_VEH_STRIDE = 16 + 3 * 10


class AEBState(enum.IntEnum):
    STANDBY = 0
    WARN = 1
    BRAKE = 2


@dataclass
class AEBSnapshot:
    ego_x: float = 0.0
    ego_z: float = 0.0
    ego_yaw: float = 0.0
    ego_speed: float = 0.0
    ego_half_w: float = 1.25
    ego_half_l: float = 3.0
    ego_arc: ArcPath | None = None
    ego_braked_arc: ArcPath | None = None
    ego_has_trailer: bool = False

    vehicles: list = field(default_factory=list)
    vehicle_arcs: dict = field(default_factory=dict)
    colliding_ids: set = field(default_factory=set)
    suppressed_ids: set = field(default_factory=set)
    braking_suppressed_ids: set = field(default_factory=set)
    evasion_filtered_ids: set = field(default_factory=set)

    aeb_state: AEBState = AEBState.STANDBY
    time_to_collision: float = _INF
    time_to_brake: float = _INF
    hit_x: float = 0.0
    hit_z: float = 0.0

    evasion_left_arc: ArcPath | None = None
    evasion_right_arc: ArcPath | None = None


@dataclass
class AEBData(ThreadData):
    AEB_warn: bool = False
    AEB_brake: bool = False
    time_to_brake: float = _INF
    em_stop_requested: bool = False
    snapshot: AEBSnapshot = field(default_factory=AEBSnapshot)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class _TrafficReader:
    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._last_vehicles: dict[int, Vehicle] = {}

    def open(self) -> bool:
        if self._buf is not None:
            return True
        try:
            self._buf = mmap.mmap(0, _BUF_SIZE, r"Local\ETS2LATraffic")
            logger.info("ETS2LATraffic shared-memory buffer opened")
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
            self._buf = None

    def read(self) -> list[Vehicle] | None:
        if self._buf is None and not self.open():
            return None
        try:
            self._buf.seek(0)
        except Exception:
            return None
        try:
            raw = struct.unpack(_TOTAL_FORMAT, self._buf[:_BUF_SIZE])
        except Exception:
            self._buf = None
            return None

        vehicles: list[Vehicle] = []
        data = raw
        for _ in range(40):
            position = Position(data[0], data[1], data[2])
            rotation = Quaternion(data[3], data[4], data[5], data[6])
            size = Size(data[7], data[8], data[9])
            speed = data[10]
            acceleration = data[11]
            trailer_count = data[12]
            vid = data[13]
            is_tmp = bool(data[14])
            is_trailer = bool(data[15])

            trailers: list[Trailer] = []
            for j in range(3):
                off = 16 + j * 10
                tp = Position(data[off], data[off + 1], data[off + 2])
                tr = Quaternion(data[off + 3], data[off + 4], data[off + 5], data[off + 6])
                ts = Size(data[off + 7], data[off + 8], data[off + 9])
                if not tp.is_zero():
                    trailers.append(Trailer(tp, tr, ts, is_tmp))

            if not position.is_zero() and not rotation.is_zero():
                vehicles.append(Vehicle(
                    position, rotation, size, speed, acceleration,
                    trailer_count, trailers, vid, is_tmp, is_trailer,
                ))
            data = data[_VEH_STRIDE:]

        t_now = time.time()
        for v in vehicles:
            if v.id in self._last_vehicles:
                v.update_from_last(self._last_vehicles[v.id], t_now)
        self._last_vehicles = {v.id: v for v in vehicles}
        return vehicles


def _cross_zone_padding(ego_yaw_rad: float, v_yaw_rad: float, v_speed_ms: float) -> float:
    """Perpendicular-target ghost-arc padding (peaks at 90° yaw diff)."""
    cross_factor = abs(math.sin(ego_yaw_rad - v_yaw_rad))
    return cross_factor * (_CROSS_SAFE_ZONE_BASE + _CROSS_SAFE_ZONE_SPEED * v_speed_ms)


def _apply_cross_zone(arc: ArcPath, padding: float) -> list[ArcPath]:
    """Return [arc] plus two ghost arcs at ±padding along the target heading."""
    if padding < 0.1:
        return [arc]
    front = build_arc(
        arc.start_x + padding * arc.fwd_x,
        arc.start_z + padding * arc.fwd_z,
        arc.yaw_rad, arc.speed, arc.curvature, arc.half_width, arc.horizon,
        decel=arc.decel, accel=arc.accel,
    )
    rear = build_arc(
        arc.start_x - padding * arc.fwd_x,
        arc.start_z - padding * arc.fwd_z,
        arc.yaw_rad, arc.speed, arc.curvature, arc.half_width, arc.horizon,
        decel=arc.decel, accel=arc.accel,
    )
    return [arc, front, rear]


def _earliest_hit(
    ego_arc: ArcPath,
    check_arcs: list[ArcPath],
    margin: float,
    n_samples: int,
) -> tuple[float, float, float] | None:
    best: tuple[float, float, float] | None = None
    for ca in check_arcs:
        h = arc_arc_collision(ego_arc, ca, margin, n_samples)
        if h is not None and (best is None or h[0] < best[0]):
            best = h
    return best


def _world_to_ego_forward(dx: float, dz: float, ego_yaw_rad: float) -> float:
    """Ego-space forward component: rz > 0 = in front of ego."""
    return dx * math.sin(ego_yaw_rad) + dz * math.cos(ego_yaw_rad)


def _is_approaching(a: ArcPath, b: ArcPath, t: float, dt: float = 0.1) -> bool:
    ax0, az0 = a.position_at_time(t)
    bx0, bz0 = b.position_at_time(t)
    ax1, az1 = a.position_at_time(t + dt)
    bx1, bz1 = b.position_at_time(t + dt)
    d0_sq = (ax0 - bx0) ** 2 + (az0 - bz0) ** 2
    d1_sq = (ax1 - bx1) ** 2 + (az1 - bz1) ** 2
    return d1_sq < d0_sq


def _build_vehicle_collision_data(
    v: Vehicle,
    dynamic_horizon: float,
    ego_yaw_rad: float,
    ego_fwd_x: float,
    ego_fwd_z: float,
) -> tuple[list[ArcPath], float, list[list[ArcPath]]]:
    """Build all_target_arcs, cross_padding, and list of cross_arcs for a vehicle.
    Used for main-loop collision and for evasion-vs-other-vehicles checks.
    """
    v_hw = v.size.width / 2.0
    v_hw_coll = max(v_hw - 0.1, 0.3)
    abs_v_speed = abs(v.speed)
    v_curvature = (
        math.radians(v.angular_velocity) / abs_v_speed if abs_v_speed > 0.5 else 0.0
    )
    _, v_yaw_deg, _ = v.rotation.euler()
    v_yaw_rad = math.radians(v_yaw_deg)
    veh_fwd_x = -math.sin(v_yaw_rad)
    veh_fwd_z = -math.cos(v_yaw_rad)
    fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
    head_on = fwd_dot < -0.7
    target_decel = _FULL_BRAKE_DECEL if head_on else 0.0
    target_accel = (
        0.0
        if target_decel > 0.0
        else max(-6.0, min(4.0, v.acceleration))
    )
    veh_arc_coll = v.get_arc(
        dynamic_horizon,
        half_width=v_hw_coll,
        decel=target_decel,
        arc_start_pctg=_ARC_START_PCTG,
    )
    tr_hw_colls: list[float] = []
    trailer_arcs_coll: list[ArcPath] = []
    for tr in v.trailers:
        tr_hw = tr.size.width / 2.0
        tr_hw_colls.append(max(tr_hw - 0.1, 0.3))
        tr_pos = tr.correct_position() if tr.is_tmp else tr.position
        _, tr_yaw_deg, _ = tr.rotation.euler()
        tr_yaw_rad = math.radians(tr_yaw_deg)
        tr_is_rev_c = v.speed < -1e-3
        tr_effective_p_c = (
            (1.0 - _ARC_START_PCTG) if tr_is_rev_c else _ARC_START_PCTG
        )
        tr_fwd_x_c = -math.sin(tr_yaw_rad)
        tr_fwd_z_c = -math.cos(tr_yaw_rad)
        tr_body_offset_c = (tr_effective_p_c - 0.5) * tr.size.length
        trailer_arcs_coll.append(
            build_arc(
                tr_pos.x + tr_body_offset_c * tr_fwd_x_c,
                tr_pos.z + tr_body_offset_c * tr_fwd_z_c,
                tr_yaw_rad,
                v.speed,
                v_curvature,
                tr_hw_colls[-1],
                dynamic_horizon,
                decel=target_decel,
                accel=target_accel,
            )
        )
    all_target_arcs = [veh_arc_coll] + trailer_arcs_coll
    cross_padding = _cross_zone_padding(ego_yaw_rad, v_yaw_rad, abs_v_speed)
    cross_arcs_list = [
        _apply_cross_zone(bt, cross_padding) for bt in all_target_arcs
    ]
    return (all_target_arcs, cross_padding, cross_arcs_list)


def _evasion_path_hits_other_vehicles(
    evasion_arc: ArcPath,
    exclude_vid: int,
    vehicle_collision_data: dict[int, tuple[list[ArcPath], float, list[list[ArcPath]]]],
    margin: float,
    n_samples: int,
) -> bool:
    """True if the evasion arc collides with any vehicle other than exclude_vid."""
    for vid, (_, _, cross_arcs_list) in vehicle_collision_data.items():
        if vid == exclude_vid:
            continue
        for cross_arcs in cross_arcs_list:
            if _earliest_hit(
                evasion_arc, cross_arcs, margin, n_samples
            ) is not None:
                return True
    return False


class AEBThread(BaseThread):
    loop_interval = 1 / 30
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="aeb_thread")
        self.data = AEBData()
        self._traffic = _TrafficReader()
        self._prev_state: AEBState = AEBState.STANDBY
        self._state_hold_until: float = 0.0
        self._last_snapshot: AEBSnapshot | None = None
        self._risk_first_seen: dict[int, float] = {}

    def setup(self) -> None:
        self._traffic.open()
        logger.debug("AEB setup complete")

    def loop(self) -> None:
        if not self.running:
            return

        aeb_active = Settings.AEB_enabled
        (ego_x, ego_z, ego_yaw_norm, ego_speed, steer, paused, ego_has_trailer,
         ego_y, ego_pitch_deg) = self._read_ego()

        if paused and self._last_snapshot is not None:
            with self.data._lock:
                self.data.snapshot = self._last_snapshot
            return

        # NO +0.5 offset — see AGENTS.md §2
        ego_yaw_rad = ego_yaw_norm * 2.0 * math.pi

        if ego_speed > 0.5:
            yaw_rate_rad_s = math.radians(steer * ego_speed * 12.0)
            ego_curvature = yaw_rate_rad_s / ego_speed
        else:
            ego_curvature = 0.0

        ego_hw: float = 1.25
        ego_half_l: float = 3.0

        t_stop = ego_speed / _FULL_BRAKE_DECEL
        dynamic_horizon = min(max(_MIN_ARC_HORIZON, t_stop * 2.0), _MAX_ARC_HORIZON)

        stopping_buffer = _STOP_BUFFER_FIXED + ego_half_l

        _ego_fwd_x = -math.sin(ego_yaw_rad)
        _ego_fwd_z = -math.cos(ego_yaw_rad)
        _ego_body_offset = (_ARC_START_PCTG - 0.5) * (2.0 * ego_half_l)
        ego_front_x = ego_x + _ego_body_offset * _ego_fwd_x
        ego_front_z = ego_z + _ego_body_offset * _ego_fwd_z

        ego_arc = build_arc(
            ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
            ego_curvature, ego_hw, dynamic_horizon,
        )

        run_collision = aeb_active and ego_speed >= _MIN_SPEED_MS

        ego_braked_arc: ArcPath | None = None
        if run_collision:
            ego_braked_arc = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                ego_curvature, ego_hw, dynamic_horizon,
                decel=_FULL_BRAKE_DECEL,
            )

        ego_evasion_left: ArcPath | None = None
        ego_evasion_right: ArcPath | None = None
        if run_collision and ego_speed > 1.0:
            delta_kappa = min(
                _EVASION_G_THRESHOLD / (ego_speed * ego_speed),
                _EVASION_FILTER_MAX_DELTA_KAPPA,
            )
            # Snap to center: when path would cross center line, cap curvature at 0
            # Left path: when ego turns right, left path can cross center → snap to center (curvature 0)
            left_kappa = ego_curvature + delta_kappa
            if ego_curvature < 0 and left_kappa < 0:
                left_kappa = left_kappa/3.0
            # Right path: when ego turns left, right path can cross center → snap to center (curvature 0)
            right_kappa = ego_curvature - delta_kappa
            if ego_curvature > 0 and right_kappa > 0:
                right_kappa = right_kappa/3.0
            ego_evasion_left = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                left_kappa, ego_hw, dynamic_horizon,
            )
            ego_evasion_right = build_arc(
                ego_front_x, ego_front_z, ego_yaw_rad, ego_speed,
                right_kappa, ego_hw, dynamic_horizon,
            )

        ego_fwd_x = ego_arc.fwd_x
        ego_fwd_z = ego_arc.fwd_z

        vehicles = self._traffic.read() or []
        now_mono = time.monotonic()

        colliding_ids: set[int] = set()
        suppressed_ids: set[int] = set()
        braking_suppressed_ids: set[int] = set()
        evasion_filtered_ids: set[int] = set()
        best_ttb: float = _INF
        best_unbraked_ttc: float = _INF
        best_raw_dist: float = _INF
        best_hit_x: float = 0.0
        best_hit_z: float = 0.0
        vehicle_dicts: list[dict] = []
        vehicle_arcs: dict[int, list[ArcPath]] = {}
        newly_risky: set[int] = set()

        ego_pitch_rad = math.radians(ego_pitch_deg)

        # Precompute per-vehicle collision arcs when run_collision, for evasion-vs-other checks
        vehicle_collision_data: dict[
            int, tuple[list[ArcPath], float, list[list[ArcPath]]]
        ] = {}
        if run_collision:
            for v in vehicles:
                vx, vz = v.position.x, v.position.z
                dx = vx - ego_x
                dz = vz - ego_z
                dist_sq = dx * dx + dz * dz
                if dist_sq > _MAX_RANGE_SQ:
                    continue
                rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
                expected_y = ego_y + rz * math.tan(ego_pitch_rad)
                if abs(v.position.y - expected_y) > _ELEVATION_MARGIN_M:
                    continue
                all_t, cross_pad, cross_list = _build_vehicle_collision_data(
                    v, dynamic_horizon, ego_yaw_rad, ego_fwd_x, ego_fwd_z
                )
                vehicle_collision_data[v.id] = (all_t, cross_pad, cross_list)

        for v in vehicles:
            vx, vz = v.position.x, v.position.z
            dx = vx - ego_x
            dz = vz - ego_z
            dist_sq = dx * dx + dz * dz
            if dist_sq > _MAX_RANGE_SQ:
                continue

            # Elevation filter (slope-aware) — see AGENTS.md §13
            rz = _world_to_ego_forward(dx, dz, ego_yaw_rad)
            expected_y = ego_y + rz * math.tan(ego_pitch_rad)
            if abs(v.position.y - expected_y) > _ELEVATION_MARGIN_M:
                continue

            dist = math.sqrt(dist_sq)
            _, v_yaw_deg, _ = v.rotation.euler()
            v_yaw_rad = math.radians(v_yaw_deg)
            v_hw = v.size.width / 2.0
            v_hw_coll = max(v_hw - 0.1, 0.3)

            abs_v_speed = abs(v.speed)
            if abs_v_speed > 0.5:
                v_curvature = math.radians(v.angular_velocity) / abs_v_speed
            else:
                v_curvature = 0.0

            veh_arc = v.get_arc(dynamic_horizon, arc_start_pctg=_ARC_START_PCTG)
            trailer_dicts = []
            trailer_arcs: list[ArcPath] = []
            tr_hw_colls: list[float] = []
            for tr in v.trailers:
                tr_pos = tr.correct_position() if tr.is_tmp else tr.position
                _, tr_yaw_deg, _ = tr.rotation.euler()
                tr_yaw_rad = math.radians(tr_yaw_deg)
                tr_hw = tr.size.width / 2.0
                tr_hw_coll = max(tr_hw - 0.1, 0.3)
                tr_hw_colls.append(tr_hw_coll)

                tr_is_rev = v.speed < -1e-3
                tr_effective_p = (1.0 - _ARC_START_PCTG) if tr_is_rev else _ARC_START_PCTG
                tr_fwd_x_l = -math.sin(tr_yaw_rad)
                tr_fwd_z_l = -math.cos(tr_yaw_rad)
                tr_body_offset = (tr_effective_p - 0.5) * tr.size.length
                tr_arc = build_arc(
                    tr_pos.x + tr_body_offset * tr_fwd_x_l,
                    tr_pos.z + tr_body_offset * tr_fwd_z_l,
                    tr_yaw_rad,
                    v.speed, v_curvature, tr_hw, dynamic_horizon,
                )
                trailer_arcs.append(tr_arc)

                trailer_dicts.append({
                    "x": tr_pos.x, "z": tr_pos.z,
                    "yaw": tr_yaw_rad,
                    "half_w": tr_hw,
                    "length": tr.size.length,
                    "is_tmp": tr.is_tmp,
                })

            veh_dict = {
                "vid": v.id,
                "x": vx, "z": vz,
                "yaw": v_yaw_rad,
                "half_w": v_hw,
                "length": v.size.length,
                "is_tmp": v.is_tmp,
                "speed_kmh": abs(v.speed) * 3.6,
                "rear_suppressed": False,
                "trailers": trailer_dicts,
            }

            vehicle_arcs[v.id] = [veh_arc] + trailer_arcs

            if not run_collision:
                vehicle_dicts.append(veh_dict)
                continue

            # Rear-approach / overtaker suppression
            to_veh_len = max(dist, 1e-6)
            dot_fwd = (dx * ego_fwd_x + dz * ego_fwd_z) / to_veh_len
            if dot_fwd < _REAR_DOT_THRESHOLD:
                veh_fwd_x = -math.sin(v_yaw_rad)
                veh_fwd_z = -math.cos(v_yaw_rad)
                approach_dot = veh_fwd_x * ego_fwd_x + veh_fwd_z * ego_fwd_z
                if approach_dot > 0.5 and v.speed > ego_speed + _OVERTAKE_SPEED_MARGIN:
                    veh_dict["rear_suppressed"] = True
                    suppressed_ids.add(v.id)
                    vehicle_dicts.append(veh_dict)
                    continue

            veh_fwd_x = -math.sin(v_yaw_rad)
            veh_fwd_z = -math.cos(v_yaw_rad)
            fwd_dot = ego_fwd_x * veh_fwd_x + ego_fwd_z * veh_fwd_z
            co_directional = abs(fwd_dot) > 0.7
            head_on = fwd_dot < -0.7

            precomputed = vehicle_collision_data.get(v.id)
            if precomputed is not None:
                all_target_arcs, cross_padding, _ = precomputed
            else:
                target_decel = _FULL_BRAKE_DECEL if head_on else 0.0
                target_accel = (
                    0.0 if target_decel > 0.0
                    else max(-6.0, min(4.0, v.acceleration))
                )
                veh_arc_coll = v.get_arc(dynamic_horizon, half_width=v_hw_coll,
                                         decel=target_decel, arc_start_pctg=_ARC_START_PCTG)
                trailer_arcs_coll: list[ArcPath] = []
                for idx, tr in enumerate(v.trailers):
                    tr_pos = tr.correct_position() if tr.is_tmp else tr.position
                    _, tr_yaw_deg, _ = tr.rotation.euler()
                    tr_yaw_rad = math.radians(tr_yaw_deg)
                    tr_is_rev_c = v.speed < -1e-3
                    tr_effective_p_c = (1.0 - _ARC_START_PCTG) if tr_is_rev_c else _ARC_START_PCTG
                    tr_fwd_x_c = -math.sin(tr_yaw_rad)
                    tr_fwd_z_c = -math.cos(tr_yaw_rad)
                    tr_body_offset_c = (tr_effective_p_c - 0.5) * tr.size.length
                    trailer_arcs_coll.append(build_arc(
                        tr_pos.x + tr_body_offset_c * tr_fwd_x_c,
                        tr_pos.z + tr_body_offset_c * tr_fwd_z_c,
                        tr_yaw_rad,
                        v.speed, v_curvature, tr_hw_colls[idx], dynamic_horizon,
                        decel=target_decel, accel=target_accel,
                    ))
                all_target_arcs = [veh_arc_coll] + trailer_arcs_coll
                cross_padding = _cross_zone_padding(ego_yaw_rad, v_yaw_rad, abs_v_speed)

            for base_target_arc in all_target_arcs:
                cross_arcs = _apply_cross_zone(base_target_arc, cross_padding)

                unbraked_hit = _earliest_hit(
                    ego_arc, cross_arcs, _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                )
                # Suppress diverging co-directional moving targets only
                if (unbraked_hit is not None
                        and co_directional
                        and base_target_arc.speed > 0.5
                        and not _is_approaching(ego_arc, base_target_arc, unbraked_hit[0])):
                    unbraked_hit = None

                if unbraked_hit is None:
                    continue

                unbraked_ttc = unbraked_hit[0]

                # Evasion filter — bypassed for co-directional moving and head-on
                if (ego_evasion_left is not None
                        and ego_evasion_right is not None
                        and not head_on
                        and not (co_directional and base_target_arc.speed > 0.5)):
                    left_hit = _earliest_hit(
                        ego_evasion_left, cross_arcs,
                        _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    )
                    right_hit = _earliest_hit(
                        ego_evasion_right, cross_arcs,
                        _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                    )
                    # Only filter if at least one evasion path misses this target
                    # and that same path does not hit any other vehicle
                    left_clear = (
                        left_hit is None
                        and not _evasion_path_hits_other_vehicles(
                            ego_evasion_left, v.id, vehicle_collision_data,
                            _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                        )
                    )
                    right_clear = (
                        right_hit is None
                        and not _evasion_path_hits_other_vehicles(
                            ego_evasion_right, v.id, vehicle_collision_data,
                            _CORRIDOR_MARGIN, _COLLISION_SAMPLES,
                        )
                    )
                    if left_clear or right_clear:
                        evasion_filtered_ids.add(v.id)
                        continue

                colliding_ids.add(v.id)

                # Risk confirmation
                newly_risky.add(v.id)
                if v.id not in self._risk_first_seen:
                    self._risk_first_seen[v.id] = now_mono
                if now_mono - self._risk_first_seen[v.id] < _RISK_CONFIRM_DURATION:
                    continue

                if unbraked_ttc < best_unbraked_ttc:
                    best_unbraked_ttc = unbraked_ttc
                    best_raw_dist = dist
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

                braked_hit = _earliest_hit(
                    ego_braked_arc, cross_arcs,
                    _CORRIDOR_MARGIN + stopping_buffer,
                    _COLLISION_SAMPLES,
                )

                if braked_hit is None:
                    ttb = max(unbraked_ttc - t_stop * _TIME_TO_BRAKE_BUFFER, 0.0)
                    braking_suppressed_ids.add(v.id)
                else:
                    ttb = 0.0

                if ttb < best_ttb:
                    best_ttb = ttb
                    best_hit_x = unbraked_hit[1]
                    best_hit_z = unbraked_hit[2]

            vehicle_dicts.append(veh_dict)

        self._risk_first_seen = {
            k: v for k, v in self._risk_first_seen.items() if k in newly_risky
        }

        new_state = AEBState.STANDBY
        time_to_brake = _INF
        display_ttc = best_unbraked_ttc

        if run_collision and best_ttb < _INF:
            time_to_brake = best_ttb
            if time_to_brake < _WARN_TTB_THRESHOLD:
                new_state = AEBState.WARN
            if time_to_brake < _BRAKE_TTB_THRESHOLD:
                new_state = AEBState.BRAKE

        # BRAKE latch — prevent rapid cycling near threshold
        if self._prev_state == AEBState.BRAKE and time_to_brake < _BRAKE_RELEASE_THRESHOLD:
            new_state = AEBState.BRAKE

        # Hold escalated state 0.3 s
        if self._prev_state.value > new_state.value and now_mono < self._state_hold_until:
            new_state = self._prev_state
        if new_state != self._prev_state:
            self._state_hold_until = now_mono + 0.3
        self._prev_state = new_state

        snap = AEBSnapshot(
            ego_x=ego_x, ego_z=ego_z, ego_yaw=ego_yaw_rad,
            ego_speed=ego_speed, ego_half_w=ego_hw, ego_half_l=ego_half_l,
            ego_arc=ego_arc, ego_braked_arc=ego_braked_arc,
            ego_has_trailer=ego_has_trailer,
            vehicles=vehicle_dicts, vehicle_arcs=vehicle_arcs,
            colliding_ids=colliding_ids, suppressed_ids=suppressed_ids,
            braking_suppressed_ids=braking_suppressed_ids,
            evasion_filtered_ids=evasion_filtered_ids,
            aeb_state=new_state, time_to_collision=display_ttc,
            time_to_brake=time_to_brake,
            hit_x=best_hit_x, hit_z=best_hit_z,
            evasion_left_arc=ego_evasion_left,
            evasion_right_arc=ego_evasion_right,
        )

        with self.data._lock:
            self.data.AEB_warn = (new_state >= AEBState.WARN)
            self.data.AEB_brake = (new_state == AEBState.BRAKE)
            self.data.time_to_brake = time_to_brake
            self.data.em_stop_requested = (new_state == AEBState.BRAKE)
            self.data.snapshot = snap
        self._last_snapshot = snap

    def teardown(self) -> None:
        self._traffic.close()
        with self.data._lock:
            self.data.AEB_warn = False
            self.data.AEB_brake = False
            self.data.time_to_brake = _INF
            self.data.em_stop_requested = False
            self.data.snapshot = AEBSnapshot()
        logger.debug("AEB teardown complete")

    def _read_ego(self) -> tuple[float, float, float, float, float, bool, bool, float, float]:
        try:
            tel = registry.get_thread("telemetry_thread")
            if tel is None or not tel.is_alive():
                return 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0
            with tel.data._lock:
                return (
                    tel.data.coordinateX,
                    tel.data.coordinateZ,
                    tel.data.rotationX,
                    tel.data.speed,
                    float(getattr(tel.data, "userSteer", 0.0)),
                    bool(getattr(tel.data, "paused", False)),
                    bool(getattr(tel.data, "ego_has_trailer", False)),
                    float(getattr(tel.data, "coordinateY", 0.0)),
                    float(getattr(tel.data, "rotationY", 0.0)),
                )
        except (KeyError, AttributeError):
            return 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0