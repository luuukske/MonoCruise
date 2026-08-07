"""Blinker merge arbitration (R5-R8). Candidacy lives in core/acc/blinker.py."""

from __future__ import annotations

import math
from dataclasses import dataclass


BLINKER_FREER_MARGIN_MS2: float = 0.2
BLINKER_HYST_S: float = 0.4
BLINKER_TTC_FLOOR_S: float = 5.5
BLINKER_STAGE1_HEADWAY_SCALE: float = 0.55
# Stage 2 release ramp: below the min the old lead keeps full authority,
# above the full mark it has none. Gap bounds scale with speed. README §5.
BLINKER_RELEASE_TTC_MIN_S: float = 3.0
BLINKER_RELEASE_TTC_FULL_S: float = 6.0
BLINKER_RELEASE_GAP_MIN_M: float = 8.0
BLINKER_RELEASE_GAP_FULL_M: float = 18.0
BLINKER_RELEASE_GAP_MIN_S: float = 0.35
BLINKER_RELEASE_GAP_FULL_S: float = 0.80
# R11 collapse ends the geometry shift, not the release. Keep letting the
# vacated vehicle go until it leaves the chain. README §5.
BLINKER_RELEASE_HOLD_S: float = 2.0
BLINKER_RELEASE_HOLD_MIN_M: float = 2.0


def _ramp(x: float, zero: float, full: float) -> float:
    """C1 cosine ramp: 0.0 at x <= zero, 1.0 at x >= full."""
    if x <= zero:
        return 0.0
    if x >= full:
        return 1.0
    t = (x - zero) / max(full - zero, 1e-6)
    return 0.5 * (1.0 - math.cos(math.pi * t))


def release_fraction(lead_ttc_s: float, lead_gap_m: float, v_ego: float) -> float:
    """How much of the lane being left may be dropped, on distance alone."""
    gap_min = max(BLINKER_RELEASE_GAP_MIN_M, v_ego * BLINKER_RELEASE_GAP_MIN_S)
    gap_full = max(BLINKER_RELEASE_GAP_FULL_M, v_ego * BLINKER_RELEASE_GAP_FULL_S)
    return min(
        _ramp(lead_ttc_s, BLINKER_RELEASE_TTC_MIN_S, BLINKER_RELEASE_TTC_FULL_S),
        _ramp(lead_gap_m, gap_min, gap_full),
    )


@dataclass(slots=True)
class BlinkerState:
    """What the tracker publishes about an in-progress lane change (README §5)."""
    b_eff: float = 0.0
    committed: bool = False
    lane_offset_m: float = 0.0


@dataclass
class BlinkerArbiter:
    """Controller-side stage/mode state for indicated-lane arbitration."""

    mode: str = "lane"
    mode_mono: float = float("-inf")
    committed: bool = False
    soft_ok_mono: float = float("-inf")
    soft_allowed: bool = True
    last_release: float = 0.0
    # The vehicle stage 2 let go of, held past b_eff so R11 cannot hand the
    # lane back mid-change.
    released_vid: int | None = None
    release_mono: float = float("-inf")

    def reset(self) -> None:
        self.mode = "lane"
        self.mode_mono = float("-inf")
        self.committed = False
        self.soft_ok_mono = float("-inf")
        self.soft_allowed = True
        self.last_release = 0.0
        self.released_vid = None
        self.release_mono = float("-inf")

    def soft_ok(
        self,
        *,
        ttc: float,
        a_req: float,
        b_comfort: float,
        now: float,
    ) -> bool:
        """R8 TTC floor + comfort gate, with R7 hysteresis on re-entry."""
        raw_ok = ttc > BLINKER_TTC_FLOOR_S and a_req >= -b_comfort
        if raw_ok:
            if not self.soft_allowed:
                if (now - self.soft_ok_mono) >= BLINKER_HYST_S:
                    self.soft_allowed = True
            else:
                self.soft_ok_mono = now
        else:
            self.soft_allowed = False
            self.soft_ok_mono = now
        return self.soft_allowed

    def _set_mode(self, mode: str, now: float) -> None:
        """Stamp only on a real change: mode_mono is the age of the mode.

        Refreshing it every frame made ``now - mode_mono`` one frame time,
        so the dwell test always passed and the hysteresis never expired."""
        if mode != self.mode:
            self.mode = mode
            self.mode_mono = now

    def _hold_release(
        self,
        a_lane: float,
        a_free: float,
        lane_vid: int | None,
        now: float,
        lead_ttc_s: float,
        lead_gap_m: float,
        v_ego: float,
    ) -> float:
        """Keep releasing the vacated vehicle after the intent has collapsed.

        R11 fires on merge completion, but the vehicle ego just left is still
        published for as long as its score takes to decay. Ending the release
        there put it back in full command for that window, which is the brake
        blip felt just as ego clears the old lane."""
        if self.released_vid is None:
            self.last_release = 0.0
            return a_lane
        expired = (now - self.release_mono) >= BLINKER_RELEASE_HOLD_S
        if lane_vid != self.released_vid or expired:
            self.released_vid = None
            self.last_release = 0.0
            return a_lane
        release = release_fraction(lead_ttc_s, lead_gap_m, v_ego)
        self.last_release = release
        return a_lane + release * (a_free - a_lane)

    def arbitrate(
        self,
        a_lane: float,
        a_ind: float,
        *,
        b_eff: float,
        committed: bool,
        soft_ok: bool,
        ind_vid: int | None,
        now: float,
        lead_ttc_s: float,
        lead_gap_m: float,
        v_ego: float,
        lane_vid: int | None = None,
        a_free: float = 0.0,
        lane_offset_m: float = 0.0,
    ) -> float:
        """Returns the commanded accel for the blinker stage in force."""
        if abs(b_eff) < 1e-6:
            self._set_mode("lane", now)
            self.committed = False
            return self._hold_release(
                a_lane, a_free, lane_vid, now, lead_ttc_s, lead_gap_m, v_ego,
            )

        self.committed = bool(committed)
        freer = a_ind > a_lane + BLINKER_FREER_MARGIN_MS2
        tighter = a_ind < a_lane - BLINKER_FREER_MARGIN_MS2

        desired = "lane"
        if tighter:
            desired = "merge"
        elif freer and soft_ok:
            desired = "pass" if self.committed else "soften"

        prev = self.mode
        if desired != prev:
            held = (now - self.mode_mono) < BLINKER_HYST_S
            if (
                desired in ("merge", "pass")
                and prev in ("merge", "pass")
                and ind_vid is not None
                and held
            ):
                desired = prev
            elif held and prev != "lane":
                if desired == "lane" or (
                    prev == "merge" and desired in ("soften", "pass")
                ) or (
                    prev in ("soften", "pass") and desired == "merge"
                ):
                    desired = prev
            self._set_mode(desired, now)

        mode = self.mode
        if mode == "pass":
            release = release_fraction(lead_ttc_s, lead_gap_m, v_ego)
            self.last_release = release
            # Latch only once ego is genuinely most of the way over, so an
            # aborted change never carries the release past its intent.
            if self.committed and lane_offset_m >= BLINKER_RELEASE_HOLD_MIN_M:
                self.released_vid = lane_vid
                self.release_mono = now
            if release >= 1.0:
                return a_ind
            return a_ind + (1.0 - release) * min(0.0, a_lane - a_ind)
        # merge / soften / lane all reassert the current lane: drop the hold.
        self.released_vid = None
        self.last_release = 0.0
        if mode == "merge":
            return min(a_lane, a_ind)
        return a_lane
