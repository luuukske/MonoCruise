"""One MonoCruise client: radar views, ACC, cruise PID, AEB and the orchestrator's glue.

Only the tracker is replaced (straight-road geometry publishes the leads) and the
thread registry is a stand-in pointed at the client being stepped. See
tools/acc_platoon/README.md.
"""
from __future__ import annotations

import threading

from core.acc.scoring import SCORE_MAX
from core.aeb.calibration import DEFAULT as AEB_CAL
from core.aeb.clip_eval import _make_headless
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController
from core.cruise_control_thread.thread import CruiseControlThread
from core.longitudinal import acc as long_acc
from core.longitudinal import cc as long_cc
from core.longitudinal.base import LongCtx, LongOutput
from core.radar.traffic import Position, Quaternion, Size, Vehicle

from .netcode import NetProfile, TmpStream, TruePath
from .plant import EGO_FRONT_OFFSET_M, RIG_LEN_M, TRACTOR_LEN_M, TruckPlant, TruckSpec

# ACCTracker's longitudinal cut-off and how many in-path leads it publishes.
TRACKER_RANGE_M: float = 150.0
TRACKER_LEADS: int = 3
# Traffic-buffer reach: a truck further out is dropped and re-sighted cold.
BUFFER_RANGE_M: float = 200.0
TRAILER_ID_BASE: int = 1000
# Orchestrator constants: `_CC_DISARM_SPEED_MS`, `_CC_DISARM_PENDING_TIMEOUT_S`.
DISARM_SPEED_MS: float = 0.3
DISARM_PENDING_S: float = 5.0
# The resuming driver waits for the truck ahead to have pulled this far away.
RESUME_LEAD_MOVE_M: float = 3.0

_FWD = Quaternion(1.0, 0.0, 0.0, 0.0)
_RIG_SIZE = Size(2.5, 3.8, RIG_LEN_M)


class _Thread:
    def __init__(self, data: object) -> None:
        self.data = data

    def is_alive(self) -> bool:
        return True


class _AccData:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.leads: list[_Lead] = []
        self.indicated_lead = None
        self.blinker_b_eff = 0.0
        self.blinker_committed = False
        self.blinker_lane_offset_m = 0.0


class _SendingData:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.hold_active = False
        self.mapper_est_max_accel_ms2 = 0.0


class _VehicleRef:
    __slots__ = ("id", "is_parked")

    def __init__(self, vid: int) -> None:
        self.id = vid
        self.is_parked = False


class _Lead:
    """The `LeadInfo` fields the controller reads."""

    __slots__ = ("vehicle", "score", "dist_m", "effective_speed_ms", "effective_accel_ms2")

    def __init__(self, vid: int, dist_m: float, speed: float, accel: float) -> None:
        self.vehicle = _VehicleRef(vid)
        self.score = SCORE_MAX
        self.dist_m = dist_m
        self.effective_speed_ms = speed
        self.effective_accel_ms2 = accel


class Registry:
    """Stands in for the thread registry, pointed at whichever client is being stepped."""

    def __init__(self) -> None:
        self.client: Client | None = None

    def get_thread(self, name: str) -> _Thread:
        threads = self.client.threads if self.client is not None else {}
        if name not in threads:
            raise KeyError(name)
        return threads[name]


class _TracedACC(AdaptiveCruiseController):
    """The shipped controller, recording ticks a safety overlay owned."""

    def __init__(self) -> None:
        super().__init__()
        self.overlay = False

    def _safety_overlays(self, primary_raw, v_ego):
        out = super()._safety_overlays(primary_raw, v_ego)
        if out is not None:
            self.overlay = True
        return out


class RadarView:
    """One remote rig through the shipped radar filter chain on one client."""

    def __init__(self, vid: int) -> None:
        self.vid = vid
        self.vehicle: Vehicle | None = None

    def read(self, front_s: float, t_now: float, ego_s: float, ego_speed: float) -> Vehicle:
        center = front_s - 0.5 * RIG_LEN_M
        v = Vehicle(Position(0.0, 0.0, -center), _FWD, _RIG_SIZE, 0.0, 0.0, 1, [],
                    self.vid, True, False)
        if self.vehicle is None:
            v.time = t_now
        else:
            v.update_from_last(self.vehicle, t_now, 0.0, 0.0, -ego_s, ego_speed)
        self.vehicle = v
        return v


def rig_front(v: Vehicle) -> float:
    """Front bumper of a rig as the radar holds it (a hold keeps the last good pose)."""
    return -v.position.z + 0.5 * RIG_LEN_M


def _ctx(t: float, dt: float, v: float, aeb_brake: bool) -> LongCtx:
    return LongCtx(now=t, dt=dt, speed_ms=v, gear_dashboard=1, park_brake=False,
                   game_throttle=0.0, game_clutch=0.0, game_brake=0.0, aeb_brake=aeb_brake,
                   connected=True, paused=False, em_stop=False, device_lost=False)


class Client:
    """One truck. Index 0 is the scripted lead; the others run MonoCruise."""

    def __init__(self, idx: int, spec: TruckSpec, net: NetProfile, s0: float, v0: float,
                 set_kmh: float, t0: float, dt: float, clock, with_aeb: bool) -> None:
        self.idx = idx
        self.spec = spec
        self.net = net
        self.set_kmh = set_kmh
        self.dt = dt
        self.truck = TruckPlant(spec, s0, v0, dt)
        self.path = TruePath(t0, s0, v0)
        self.cmd = 0.0
        self.acc_data = _AccData()
        self.sending_data = _SendingData()
        self.threads = {"acc_thread": _Thread(self.acc_data),
                        "sending_thread": _Thread(self.sending_data)}
        self.acc = long_acc.AdaptiveCruiseController()
        self.traced = _TracedACC()
        self.acc._inner = self.traced
        self.cc = long_cc.CruiseController()
        self.cc.enable()
        self.cc.set_target_kmh(set_kmh)
        self.streams: dict[int, TmpStream] = {}
        self.views: dict[int, RadarView] = {}
        self.next_frame = 0
        self.next_leads: list[_Lead] = []
        self.frame_vehicles: list[Vehicle] = []
        self.frame_t = t0
        self.aeb = _headless_aeb(clock, spec.brake_ms2) if with_aeb and idx else None
        self.aeb_brake = False
        self.aeb_target_ms2 = 0.0
        self._disarm_until = 0.0
        self.disarmed_at: float | None = None
        self._lead_at_disarm: float | None = None
        self._pulled_away_at: float | None = None
        self.resumes = 0
        self.contacts = 0

    def radar_frame(self, t: float, split_trailers: bool) -> None:
        """Read every drawn rig ahead, then publish what ACCTracker would on a straight road."""
        ego_front = self.truck.s
        ego_pos = ego_front - EGO_FRONT_OFFSET_M
        candidates: list[_Lead] = []
        seen: list[Vehicle] = []
        for j, stream in self.streams.items():
            drawn = stream.position(t)
            if not 0.0 < drawn - ego_front <= BUFFER_RANGE_M:
                self.views.pop(j, None)
                continue
            view = self.views.get(j)
            if view is None:
                view = self.views[j] = RadarView(j)
            veh = view.read(drawn, t, ego_pos, self.truck.v)
            seen.append(veh)
            front = rig_front(veh)
            rears = [(TRAILER_ID_BASE + j, front - RIG_LEN_M)]
            if split_trailers:
                rears.append((j, front - TRACTOR_LEN_M))
            for vid, rear in rears:
                dist = rear - ego_pos
                if 0.0 < dist <= TRACKER_RANGE_M:
                    candidates.append(_Lead(vid, dist, veh.acc_speed, veh.acc_accel))
        candidates.sort(key=lambda lead: lead.dist_m)
        self.next_leads = candidates[:TRACKER_LEADS]
        self.frame_vehicles = seen
        self.frame_t = t

    def aeb_tick(self) -> None:
        if self.aeb is None:
            return
        ego_pos = self.truck.s - EGO_FRONT_OFFSET_M
        # Past its 3 s arc horizon AEB has nothing to say; leaving far rigs out saves its probe.
        near = [v for v in self.frame_vehicles if rig_front(v) - ego_pos <= TRACKER_RANGE_M]
        snap = (near, 0.0, 0.0, -ego_pos, 0.0, self.truck.v, 0.0, 0.0, True,
                None, True, False, self.frame_t, frozenset(), self.frame_t)
        self.aeb._read_radar_snapshot = lambda: snap
        self.aeb.loop()
        self.aeb_brake = bool(self.aeb.data.AEB_brake)
        self.aeb_target_ms2 = float(self.aeb.data.AEB_target_decel_ms2) if self.aeb_brake else 0.0

    def control_tick(self, t: float, dt: float) -> LongOutput:
        """Cruise PID and ACC cap, min-arbitrated as the orchestrator does, then AEB on top."""
        self.sending_data.hold_active = self.truck.hold_out.active
        self.sending_data.mapper_est_max_accel_ms2 = self.spec.gas_limit_ms2(self.truck.v)
        self.traced.overlay = False
        ctx = _ctx(t, dt, self.truck.v, self.aeb_brake)
        self._disengage(ctx)
        cc_out = self.cc.step(ctx)
        acc_out = self.acc.step(ctx) if cc_out.active else LongOutput(None, False)
        if not cc_out.active:
            self.acc.reset()
        wanted, commanding, _ = CruiseControlThread._arbitrate_named(("cc", cc_out), ("acc", acc_out))
        cmd = wanted if commanding else 0.0
        if self.aeb_brake:
            cmd = min(cmd, -self.aeb_target_ms2)
        self.cmd = cmd
        return acc_out

    def _disengage(self, ctx: LongCtx) -> None:
        """AEB-then-stop disarm, as `_handle_cc_disengage_conditions` does it."""
        if not self.cc.enabled:
            self._disarm_until = 0.0
            return
        if ctx.aeb_brake:
            self._disarm_until = ctx.now + DISARM_PENDING_S
        if ctx.now < self._disarm_until and ctx.speed_ms < DISARM_SPEED_MS:
            self.cc.disable()
            self._disarm_until = 0.0
            self.disarmed_at = ctx.now
            self._lead_at_disarm = None
            self._pulled_away_at = None

    def driver(self, t: float, lead_front: float, resume_after_s: float | None) -> None:
        """After an AEB stop disarmed ACC, the driver taps resume once the truck ahead pulls away."""
        if self.disarmed_at is None or self.cc.enabled or resume_after_s is None:
            return
        if self._lead_at_disarm is None:
            self._lead_at_disarm = lead_front
        if lead_front - self._lead_at_disarm < RESUME_LEAD_MOVE_M:
            return
        if self._pulled_away_at is None:
            self._pulled_away_at = t
        if t - self._pulled_away_at >= resume_after_s:
            self.cc.enable()
            self.cc.set_target_kmh(self.set_kmh)
            self.disarmed_at = None
            self.resumes += 1


def _headless_aeb(clock, capacity_ms2: float):
    t = _make_headless(AEB_CAL)
    t._read_max_brake_ms2 = lambda: capacity_ms2
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    t._read_vehicle_key = lambda: None
    t._now = clock.monotonic
    return t
