"""ACC standstill hold, released by the gap law's wanted accel. See core/acc/ACC_ARCHITECTURE.md §10."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import idm_cah

if TYPE_CHECKING:
    from .acc_controller import ACConfig, _LeadSnapshot

# Wanted accel that releases the hold. Matches the hold FSM's launch release in
# core/sending_thread/hold_controller.py, so a release is always a real launch bid.
LAUNCH_ACCEL_MS2: float = 0.25
# Below this ego counts as at rest: no bid under the launch release may sit there.
REST_SPEED_MS: float = 0.1
# A hold engaged with the law already near the release must see it rise this far
# first, or the stop left at the hold FSM's own threshold re-launches on noise.
RELEASE_MARGIN_MS2: float = 0.15


class StandstillHold:
    """Latch that pins the cap at zero behind a close lead while ego is stopped.

    Engages once the law stops asking for accel, releases once it asks to launch."""

    def __init__(self) -> None:
        self.held = False
        self.release_ms2 = LAUNCH_ACCEL_MS2

    def reset(self) -> None:
        self.held = False

    def step(
        self,
        cfg: ACConfig,
        gap_raw_m: float,
        primary: _LeadSnapshot,
        v_ego: float,
        t_headway: float,
    ) -> bool:
        """True while the hold owns the cap. `primary` is the smoothed immediate lead."""
        if (v_ego >= cfg.standstill_speed_ms
                or gap_raw_m > cfg.s0_m + cfg.standstill_gap_slack_m):
            self.held = False
            return False
        # The raw gap binds only when shorter: a lead closing in holds at once.
        wanted = idm_cah.lead_law(cfg, min(gap_raw_m, primary.dist_m), v_ego, primary.v_lead_ms,
                                  primary.a_lead_ms2, t_headway, primary.a_lead_ff_ms2)
        launch = cfg.standstill_launch_accel_ms2
        if not self.held:
            # Rolling, [0, launch) is hysteresis so a crawl the law still wants
            # is never pinned. At rest a sub-launch bid would wind up the mapper.
            if wanted > 0.0 and (v_ego >= REST_SPEED_MS or wanted >= launch):
                return False
            self.held = True
            self.release_ms2 = wanted + RELEASE_MARGIN_MS2
        self.release_ms2 = max(launch, min(self.release_ms2, wanted + RELEASE_MARGIN_MS2))
        self.held = wanted < self.release_ms2
        return self.held
