"""ACC brake landing: stop braking once the lead's smoothed speed stops falling.

See core/acc/ACC_ARCHITECTURE.md §13.4."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from . import idm_cah

if TYPE_CHECKING:
    from .acc_controller import ACConfig, _LeadSnapshot

# Speed the truck still sheds once the command lets go: brake lag plus the release
# chase. Braking harder than the overspeed covers lands ego below the lead.
LANDING_MOMENTUM_S: float = 0.5
# The overspeed left after that is braked off over this horizon. 0 disables. §13.4.
LANDING_HORIZON_S: float = 1.0
# A gap short of the wanted one is reopened by landing below the lead: the deficit
# spread over this many seconds.
LANDING_REOPEN_S: float = 6.0
# Trend of the lead's smoothed speed over this window; landing fades in between the
# two trends (m/s^2), so a lead still slowing keeps the plain law.
LANDING_TREND_WINDOW_S: float = 0.5
LANDING_TREND_OFF_MS2: float = -1.0
LANDING_TREND_FULL_MS2: float = -0.3
# A slower lead is a stop, not a landing.
LANDING_LEAD_MIN_MS: float = 4.0
# Ego must already be braking this hard for there to be anything to land.
LANDING_EGO_DECEL_MS2: float = -0.5
# A gap closing this much faster than the seen speeds allow means the lead is slower
# than it looks; landing stays off for the hold after.
LANDING_GAP_TOL_MS: float = 1.5
LANDING_BLOCK_S: float = 0.5
# Ego accel is a tracking differentiator on speed, like the limiter's. AGENTS.md.
EGO_ACCEL_TAU_S: float = 0.15


def _ls_slope(points: list[tuple[float, float]]) -> float | None:
    n = len(points)
    if n < 3:
        return None
    t_mean = sum(p[0] for p in points) / n
    y_mean = sum(p[1] for p in points) / n
    den = sum((p[0] - t_mean) ** 2 for p in points)
    if den <= 1e-9:
        return None
    return sum((p[0] - t_mean) * (p[1] - y_mean) for p in points) / den


class BrakeLanding:
    """Raises a braking lead law to the brake that lands ego on the lead's speed.

    Only ever raises it, and only while the lead's smoothed speed is flat."""

    def __init__(self) -> None:
        self._hist: deque[tuple[float, float, float]] = deque()
        self._vid: int | None = None
        self._v_smooth: float | None = None
        self.ego_accel_ms2 = 0.0
        self._blocked_until = -math.inf
        self.gap_excess_ms: float | None = None

    def reset(self) -> None:
        self._hist.clear()
        self._vid = None
        self._v_smooth = None
        self.ego_accel_ms2 = 0.0
        self._blocked_until = -math.inf
        self.gap_excess_ms = None

    def track_ego(self, v_ego: float, dt: float) -> None:
        """Step the ego accel estimate. Every tick, lead or not."""
        if self._v_smooth is None:
            self._v_smooth = v_ego
            return
        self.ego_accel_ms2 = (v_ego - self._v_smooth) / EGO_ACCEL_TAU_S
        self._v_smooth += (1.0 - math.exp(-dt / EGO_ACCEL_TAU_S)) * (v_ego - self._v_smooth)

    def step(
        self,
        cfg: ACConfig,
        a_law: float,
        raw: _LeadSnapshot,
        smooth: _LeadSnapshot,
        v_ego: float,
        t_headway: float,
        now: float,
    ) -> float:
        """The immediate lead's law, landed. `raw` feeds the trend, `smooth` the target."""
        if raw.vid != self._vid:
            self._hist.clear()
            self._vid = raw.vid
            self.gap_excess_ms = None
        window = cfg.landing_trend_window_s
        if not self._hist or now > self._hist[-1][0]:
            self._hist.append((now, raw.v_lead_ms, raw.dist_m))
        while self._hist and self._hist[0][0] < now - window:
            self._hist.popleft()
        if len(self._hist) < 3 or self._hist[-1][0] - self._hist[0][0] < 0.5 * window:
            return a_law
        trend = _ls_slope([(t, v) for t, v, _ in self._hist])
        gap_rate = _ls_slope([(t, d) for t, _, d in self._hist])
        if trend is None or gap_rate is None:
            return a_law
        self.gap_excess_ms = (raw.v_lead_ms - v_ego) - gap_rate
        # The gap is timely where the lead speed lags: trust it over a flat trend.
        if gap_rate < raw.v_lead_ms - v_ego - cfg.landing_gap_tol_ms:
            self._blocked_until = now + cfg.landing_block_s
        if (cfg.landing_horizon_s <= 0.0 or a_law >= 0.0 or now < self._blocked_until
                or self.ego_accel_ms2 > cfg.landing_ego_decel_ms2
                or smooth.v_lead_ms < cfg.landing_lead_min_ms):
            return a_law
        weight = 1.0 - idm_cah.fade(trend, cfg.landing_trend_off_ms2, cfg.landing_trend_full_ms2)
        s_want = cfg.s0_m + smooth.v_lead_ms * t_headway
        v_target = smooth.v_lead_ms - max(0.0, s_want - smooth.dist_m) / cfg.landing_reopen_s
        overspeed = v_ego - v_target + self.ego_accel_ms2 * cfg.landing_momentum_s
        floor = min(0.0, -overspeed / cfg.landing_horizon_s)
        return a_law + weight * max(0.0, floor - a_law)
