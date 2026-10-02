"""Blinker candidacy helpers for ACCTracker. Rules: core/acc/README.md §5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.radar.traffic import Trailer, Vehicle


# Telemetry gives the lamp, not the stalk. A dark gap longer than this is
# the driver cancelling; anything shorter is the lamp between flashes.
BLINKER_LAMP_GAP_S: float = 1.0
# One intent: peak, cos-decay, sustain, release fade. README §5.
BLINKER_PEAK_S: float = 0.35
BLINKER_DECAY_S: float = 1.75
BLINKER_SUSTAIN: float = 0.55
BLINKER_RELEASE_S: float = 0.8
# One tap has to carry a whole lane change, so the intent outlives a dark lamp
# and coasts to zero at this age instead of snapping there. README §5.
BLINKER_MIN_HOLD_S: float = 4.0
BLINKER_OFFSET_M: float = 4.5
BLINKER_LANE_WIDTH_M: float = 4.5
BLINKER_RAMP_LO_KMH: float = 45.0
BLINKER_RAMP_HI_KMH: float = 60.0
BLINKER_GATE_HYST_KMH: float = 3.0
BLINKER_AHEAD_MARGIN_M: float = 1.5
BLINKER_OVERTAKE_VREL_MS: float = 1.0
BLINKER_BEHIND_WINDOW_S: float = 4.0
BLINKER_CANDIDATE_RANGE_M: float = 100.0
BLINKER_LANE_HALF_W_M: float = 2.0
BLINKER_HEADWAY_MARGIN_S: float = 2.0
# Parallel traffic: nose can clear R2 while the body still overlaps ego.
EGO_FRONT_OFFSET_M: float = 2.5
BLINKER_CLOSING_VREL_MS: float = 0.5
# A lane change takes about this long, so a rear bumper ego draws level with
# sooner than this belongs to a vehicle ego passes, not one it merges behind.
BLINKER_PASS_CLEAR_S: float = 3.0
BLINKER_STICKY_S: float = 0.4
BLINKER_LATERAL_COMMIT_LANES: float = 0.7
# Commitment: how far ego has moved toward the indicated lane, measured
# against the vehicle it was following so road curvature cannot fake it.
BLINKER_COMMIT_LAT_M: float = 0.4
# Lateral offset past which a body counts as being on the side ego is leaving.
VACATE_DEADBAND_M: float = 0.1
CODIR_DEG: float = 30.0


def trailer_corners(tr: Trailer) -> tuple[tuple[float, float], ...]:
    """World corners of one trailer body; TMP pivot shifted to center."""
    if tr.is_tmp:
        pos = tr.correct_position()
    else:
        pos = tr.position
    _, yaw_deg, _ = tr.rotation.euler()
    yaw_rad = math.radians(yaw_deg)
    fwd_x = -math.sin(yaw_rad)
    fwd_z = -math.cos(yaw_rad)
    right_x = -fwd_z
    right_z = fwd_x
    half_l = tr.size.length * 0.5
    hw = tr.size.width * 0.5
    px, pz = pos.x, pos.z
    return (
        (px + half_l * fwd_x + hw * right_x, pz + half_l * fwd_z + hw * right_z),
        (px + half_l * fwd_x - hw * right_x, pz + half_l * fwd_z - hw * right_z),
        (px - half_l * fwd_x - hw * right_x, pz - half_l * fwd_z - hw * right_z),
        (px - half_l * fwd_x + hw * right_x, pz - half_l * fwd_z + hw * right_z),
    )


def train_corners(
    v: Vehicle,
    extra_vehicles: list[Vehicle] | None = None,
) -> list[tuple[float, float]]:
    """Cab + nested trailers + linked TMP trailer vehicles (R4 / parallel)."""
    out = list(v.get_corners())
    for tr in v.trailers:
        if tr.is_zero():
            continue
        out.extend(trailer_corners(tr))
    for other in extra_vehicles or ():
        out.extend(other.get_corners())
    return out


def desired_time_headway_s() -> float:
    """Controller gap config for R4; no parallel constant."""
    try:
        from core.cruise_control_thread.acc_controller import (
            T_HEADWAY_BY_LEVEL_S,
            T_HEADWAY_S,
        )
        from core.settings import Settings

        level = int(Settings.acc_gap_level)
        if 1 <= level <= 4:
            return T_HEADWAY_BY_LEVEL_S[level]
        return T_HEADWAY_S
    except (ImportError, TypeError, ValueError, AttributeError):
        return 1.5


@dataclass
class BlinkerBias:
    """Lamp debounce, intent envelope, R0 gate, R11 collapse, candidate checks."""

    _lamp_left_mono: float = float("-inf")
    _lamp_right_mono: float = float("-inf")
    # Current intent: -1 left, +1 right, 0 idle. One flick, one intent.
    intent_sign: float = 0.0
    _intent_mono: float = float("-inf")
    # When the lamp went dark on a still-live intent; the envelope coasts down
    # from here instead of holding. Infinite while the lamp is still asking.
    _dark_mono: float = float("inf")
    # Release fade after the intent ends, so b_eff never steps to zero.
    _release_sign: float = 0.0
    _release_amp: float = 0.0
    _release_mono: float = float("-inf")
    gate_on: bool = False
    collapsed: bool = False
    rising_this_frame: bool = False
    committed: bool = False
    lane_offset_m: float = 0.0
    # Who ego was following when the intent began; commitment is ego moving
    # off that vehicle's line, so a curve cannot read as a lane change.
    commit_ref_vid: int | None = None
    commit_ref_lat: float = 0.0
    lc_origin_x: float = 0.0
    lc_origin_z: float = 0.0
    lc_right_x: float = 1.0
    lc_right_z: float = 0.0
    sticky_id: int | None = None
    sticky_mono: float = float("-inf")
    last_scalar: float = 0.0
    last_b_eff: float = 0.0

    def update_stalk(
        self,
        now_mono: float,
        ego_speed_ms: float,
        blinker_left: bool,
        blinker_right: bool,
        ego_x: float = 0.0,
        ego_z: float = 0.0,
        ego_fwd_x: float = 0.0,
        ego_fwd_z: float = -1.0,
    ) -> None:
        if blinker_left:
            self._lamp_left_mono = now_mono
        if blinker_right:
            self._lamp_right_mono = now_mono
        left_held = (now_mono - self._lamp_left_mono) <= BLINKER_LAMP_GAP_S
        right_held = (now_mono - self._lamp_right_mono) <= BLINKER_LAMP_GAP_S

        ego_kmh = ego_speed_ms * 3.6
        lo = BLINKER_RAMP_LO_KMH
        hyst = BLINKER_GATE_HYST_KMH
        if self.gate_on:
            if ego_kmh <= lo - hyst:
                self.gate_on = False
        elif ego_kmh >= lo + hyst:
            self.gate_on = True

        # Hazards (both sides) indicate nothing about a lane change.
        side = 0.0
        if left_held and not right_held:
            side = -1.0
        elif right_held and not left_held:
            side = 1.0
        want = side if self.gate_on else 0.0

        self.rising_this_frame = False
        if want == self.intent_sign:
            if want != 0.0:
                # Lamp back on before the coast ended: the driver is still asking.
                self._dark_mono = float("inf")
            return
        if want != 0.0:
            self._begin_intent(now_mono, want, ego_x, ego_z, ego_fwd_x, ego_fwd_z)
            return
        # Minimum envelope: a dark lamp does not end the intent until the tap
        # has bought a realistic lane change. The R0 gate is not held this way.
        if (
            self.gate_on
            and not self.collapsed
            and (now_mono - self._intent_mono) < BLINKER_MIN_HOLD_S
        ):
            if not math.isfinite(self._dark_mono):
                self._dark_mono = now_mono
            return
        self._end_intent(now_mono)

    def _begin_intent(
        self,
        now_mono: float,
        sign: float,
        ego_x: float,
        ego_z: float,
        ego_fwd_x: float,
        ego_fwd_z: float,
    ) -> None:
        self.rising_this_frame = True
        self.intent_sign = sign
        self._intent_mono = now_mono
        self._dark_mono = float("inf")
        self._release_sign = 0.0
        self.collapsed = False
        self.committed = False
        self.lane_offset_m = 0.0
        self.lc_origin_x = ego_x
        self.lc_origin_z = ego_z
        self.lc_right_x = -ego_fwd_z
        self.lc_right_z = ego_fwd_x
        self.sticky_id = None

    def _end_intent(self, now_mono: float) -> None:
        self._release_sign = self.intent_sign
        self._release_amp = self._envelope_amp(now_mono)
        self._release_mono = now_mono
        self.intent_sign = 0.0
        self.committed = False
        self.lane_offset_m = 0.0

    @staticmethod
    def _lamp_amp(age: float) -> float:
        """Peak, cos-decay, then sustain, for as long as the lamp is asking."""
        if age <= BLINKER_PEAK_S:
            return 1.0
        t = age - BLINKER_PEAK_S
        if t >= BLINKER_DECAY_S:
            return BLINKER_SUSTAIN
        c = math.cos((t / BLINKER_DECAY_S) * (math.pi * 0.5))
        return BLINKER_SUSTAIN + (1.0 - BLINKER_SUSTAIN) * c

    def _envelope_amp(self, now_mono: float) -> float:
        """Amplitude of a live intent: on the lamp, or coasting down off it."""
        if self.intent_sign == 0.0:
            return 0.0
        if not math.isfinite(self._dark_mono):
            return self._lamp_amp(now_mono - self._intent_mono)
        # Lamp already dark: coast what is left to zero over the rest of the
        # minimum envelope, rather than holding a sustain and stepping off it.
        dark_age = self._dark_mono - self._intent_mono
        span = max(BLINKER_MIN_HOLD_S - dark_age, BLINKER_RELEASE_S)
        t = (now_mono - self._dark_mono) / span
        if t >= 1.0:
            return 0.0
        return self._lamp_amp(dark_age) * math.cos(t * (math.pi * 0.5))

    def scalar(self, now_mono: float) -> float:
        """Signed intent envelope: sustained while indicating, faded after."""
        if self.intent_sign != 0.0:
            return self.intent_sign * self._envelope_amp(now_mono)
        if self._release_sign == 0.0:
            return 0.0
        t = now_mono - self._release_mono
        if t < 0.0 or t >= BLINKER_RELEASE_S:
            self._release_sign = 0.0
            return 0.0
        fade = math.cos((t / BLINKER_RELEASE_S) * (math.pi * 0.5))
        return self._release_sign * self._release_amp * fade

    def _speed_gate_scale(self, ego_kmh: float) -> float:
        if not self.gate_on:
            return 0.0
        span = max(BLINKER_RAMP_HI_KMH - BLINKER_RAMP_LO_KMH, 1e-3)
        return max(0.0, min(1.0, (ego_kmh - BLINKER_RAMP_LO_KMH) / span))

    def compute_b_eff(self, now_mono: float, ego_kmh: float) -> float:
        raw = self.scalar(now_mono)
        self.last_scalar = raw
        if abs(raw) < 1e-6:
            self.collapsed = False
            self.last_b_eff = 0.0
            return 0.0
        if self.collapsed:
            self.last_b_eff = 0.0
            return 0.0
        b_eff = raw * self._speed_gate_scale(ego_kmh)
        self.last_b_eff = b_eff
        return b_eff

    def lateral_commit_m(self, ego_x: float, ego_z: float) -> float:
        """Ego displacement in the frame frozen at the intent edge.

        Fallback only: a curve accumulates displacement here with no lane
        change at all, so prefer the lead-relative measure when one exists."""
        return (
            (ego_x - self.lc_origin_x) * self.lc_right_x
            + (ego_z - self.lc_origin_z) * self.lc_right_z
        )

    def set_commit_reference(self, vid: int | None, road_lat: float) -> None:
        """Freeze the vehicle commitment is measured against, once per intent."""
        self.commit_ref_vid = vid
        self.commit_ref_lat = road_lat

    def update_commit(self, lane_offset_m: float) -> bool:
        """Latch commitment once ego has measurably moved toward the lane.

        ``lane_offset_m`` is signed toward the indicated side. The latch holds
        for the life of the intent: a lane change does not un-happen, and
        re-testing it every frame is what let noise drop the release."""
        if self.intent_sign == 0.0:
            self.committed = False
            self.lane_offset_m = 0.0
            return False
        self.lane_offset_m = lane_offset_m
        if lane_offset_m >= BLINKER_COMMIT_LAT_M:
            self.committed = True
        return self.committed

    def maybe_collapse(self, b_eff: float, *, sticky_in_corridor: bool) -> float:
        """R11: collapse on handover or full lane commit. Returns gated b_eff."""
        if abs(b_eff) < 1e-6 or self.collapsed:
            return 0.0 if self.collapsed else b_eff
        lat_commit = self.lane_offset_m >= (
            BLINKER_LATERAL_COMMIT_LANES * BLINKER_LANE_WIDTH_M
        )
        if sticky_in_corridor or lat_commit:
            self.collapsed = True
            self.last_b_eff = 0.0
            return 0.0
        return b_eff

    @staticmethod
    def is_candidate(
        *,
        last_behind_mono: float,
        now_mono: float,
        b_eff: float,
        dist_m: float,
        body_rear_m: float,
        road_lat: float,
        yaw_diff_deg: float,
        ego_speed_ms: float,
        v_speed_ms: float,
        desired_th_s: float,
    ) -> tuple[bool, float]:
        """R2-R4, R12, R13. Returns (is_candidate, time_headway_s)."""
        # R4 headway is to the train rear (cab + trailers), not the nose.
        th = body_rear_m / max(ego_speed_ms, 0.5)
        if abs(b_eff) < 1e-6:
            return False, th
        if dist_m > BLINKER_CANDIDATE_RANGE_M:
            return False, th
        if abs(yaw_diff_deg) > CODIR_DEG:
            return False, th
        if dist_m - EGO_FRONT_OFFSET_M < BLINKER_AHEAD_MARGIN_M:
            return False, th
        v_rel = v_speed_ms - ego_speed_ms
        if (
            v_rel > BLINKER_OVERTAKE_VREL_MS
            and (now_mono - last_behind_mono) <= BLINKER_BEHIND_WINDOW_S
        ):
            return False, th
        # R4 lower bound: a rear bumper already beside ego belongs to a vehicle
        # ego is passing, never to the one ego follows after the merge.
        if body_rear_m <= EGO_FRONT_OFFSET_M:
            return False, th
        v_close = ego_speed_ms - v_speed_ms
        if v_close > BLINKER_CLOSING_VREL_MS:
            # Ego reaches that rear before a lane change could finish: a pass.
            if (body_rear_m - EGO_FRONT_OFFSET_M) / v_close < BLINKER_PASS_CLEAR_S:
                return False, th
        # R4 upper bound: train-rear headway, closing traffic included.
        if th >= desired_th_s + BLINKER_HEADWAY_MARGIN_S:
            return False, th
        lane_center = math.copysign(BLINKER_LANE_WIDTH_M, b_eff)
        if abs(road_lat - lane_center) > BLINKER_LANE_HALF_W_M:
            return False, th
        return True, th


def measure_lane_offset(
    b_eff: float,
    ref_lat_now: float | None,
    ref_lat_at_intent: float,
    fallback_m: float,
) -> float:
    """Signed metres ego has moved toward the indicated lane.

    Measured as the reference vehicle sliding across ego's road frame: it
    holds the road, so a curve carries ego and it together and reads zero.
    ``fallback_m`` is the frozen-frame ego displacement, used only when no
    reference is visible, where there is no lead to release anyway."""
    if abs(b_eff) < 1e-6:
        return 0.0
    sgn = math.copysign(1.0, b_eff)
    if ref_lat_now is None:
        return fallback_m * sgn
    return -(ref_lat_now - ref_lat_at_intent) * sgn


def is_vacating(body_lat_min: float, body_lat_max: float, b_eff: float) -> bool:
    """True when a body sits on the side ego is leaving (R11 hysteresis drop)."""
    if abs(b_eff) < 1e-6:
        return False
    body_mid = (body_lat_min + body_lat_max) * 0.5
    return body_mid * math.copysign(1.0, b_eff) < -VACATE_DEADBAND_M


def ego_lat_vel_ms(
    ego_history: list[tuple[float, float, float]],
    ego_fwd_x: float,
    ego_fwd_z: float,
) -> float:
    """Signed lateral velocity (m/s), positive to ego's right. Debug only.

    A vehicle's velocity lies along its own heading, so in the current ego
    frame this reads ``|v|·sin(dyaw/2)`` per frame: milli-metres per second
    during a real lane change, well under position noise. It is published for
    the debug window and must not gate control."""
    if len(ego_history) < 2:
        return 0.0
    t0, x0, z0 = ego_history[-2]
    t1, x1, z1 = ego_history[-1]
    dt = t1 - t0
    if dt <= 1e-6:
        return 0.0
    right_x = -ego_fwd_z
    right_z = ego_fwd_x
    return ((x1 - x0) * right_x + (z1 - z0) * right_z) / dt


def pick_sticky_candidate(
    scored: list[tuple[int, float]],
    *,
    sticky_id: int | None,
    sticky_mono: float,
    now_mono: float,
) -> tuple[int | None, float]:
    """R9/R7: lowest headway among candidates, identity-sticky.

    ``scored`` is ``(vid, time_headway_s)``. Returns ``(vid, new_sticky_mono)``.
    """
    if not scored:
        return None, sticky_mono
    best_vid, _ = min(scored, key=lambda item: item[1])
    if sticky_id is not None:
        sticky_present = any(vid == sticky_id for vid, _ in scored)
        if sticky_present:
            age = now_mono - sticky_mono
            if sticky_id == best_vid or age < BLINKER_STICKY_S:
                return sticky_id, sticky_mono
        return best_vid, now_mono
    return best_vid, now_mono
