"""Pulling away: see a lead start to move before acc_speed does, and follow its pace.

See core/acc/ACC_ARCHITECTURE.md §10.2."""
from __future__ import annotations

import math
import random

import pytest

from core.cruise_control_thread import pull_away
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.pull_away import LeadMotion, latch_floor, pace_lift

TICK_S = 1.0 / 30.0
FRAME_S = 1.0 / 20.0
STOP_GAP_M = 4.8


def _feed(motion: LeadMotion, lead_pos, seconds: float, v_ego=None,
          crashed: bool = False) -> list[tuple[float, float | None]]:
    """Ticks at 30 Hz; the radar position only updates at 20 Hz, as live."""
    out, odo, frame_t, dist = [], 0.0, -1.0, None
    t = 0.0
    while t < seconds:
        if t - frame_t >= FRAME_S - 1e-9 or dist is None:
            frame_t = t
            dist = lead_pos(t) - odo
        v = v_ego(t) if v_ego is not None else 0.0
        out.append((t, motion.step(t, TICK_S, 1, dist, v, crashed)))
        odo += v * TICK_S
        t += TICK_S
    return out


def _first_seen(out) -> float | None:
    return next((t for t, v in out if v is not None), None)


def test_a_stationary_lead_never_reads_as_moving():
    out = _feed(LeadMotion(), lambda t: STOP_GAP_M, 30.0)
    assert _first_seen(out) is None


@pytest.mark.parametrize("accel", [0.2, 0.3, 1.0])
def test_a_lead_pulling_away_is_seen_within_a_second(accel):
    go = 3.0
    def pos(t):
        return STOP_GAP_M + 0.5 * accel * max(t - go, 0.0) ** 2
    out = _feed(LeadMotion(), pos, go + 3.0)
    seen = _first_seen(out)
    assert seen is not None and seen - go <= 1.25
    # Tracks the real speed once seen, half a fit window behind.
    t_late, v_late = out[-1]
    lag = pull_away.MOTION_WINDOW_S / 2.0
    assert v_late == pytest.approx(accel * (t_late - go - lag), abs=0.1)


def test_ego_rolling_up_to_a_stopped_lead_is_not_lead_motion():
    out = _feed(LeadMotion(), lambda t: 12.0, 6.0, v_ego=lambda t: 1.0)
    assert _first_seen(out) is None


@pytest.mark.parametrize("hz", [0.8, 1.0, 1.5, 2.0])
def test_a_crash_rocked_vehicle_is_never_followed(hz):
    """A +-1 m/s bounce, as the acc_speed latch was built for, once it is in the history."""
    amp = 1.0 / (2.0 * math.pi * hz)
    for phase in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        def pos(t, phase=phase):
            return STOP_GAP_M + amp * math.sin(2.0 * math.pi * hz * t + phase)
        out = _feed(LeadMotion(), pos, 12.0)
        assert all(v is None for t, v in out if t >= pull_away.MOTION_LOOKBACK_S), (hz, phase)


def test_a_slow_rock_is_followed_for_one_swing_at_most():
    # At 0.5 Hz the first swing out of rest looks like a start; the swing back ends it for good.
    hz, amp = 0.5, 1.0 / (2.0 * math.pi * 0.5)
    def pos(t):
        return STOP_GAP_M + amp * (1.0 - math.cos(2.0 * math.pi * hz * max(t - 3.0, 0.0)))
    out = _feed(LeadMotion(), pos, 15.0)
    assert all(v is None for t, v in out if t >= 3.0 + 1.0 / hz)


def test_a_crashed_lead_is_never_followed():
    def pos(t):
        return STOP_GAP_M + 0.15 * max(t - 1.0, 0.0) ** 2
    assert _first_seen(_feed(LeadMotion(), pos, 5.0, crashed=True)) is None


def test_a_teleport_is_not_motion():
    def pos(t):
        return STOP_GAP_M + (8.0 if t >= 2.0 else 0.0)
    assert _first_seen(_feed(LeadMotion(), pos, 10.0)) is None


def test_tmp_playback_noise_on_a_standing_lead_is_not_motion():
    rng = random.Random(7)
    jitter = {}
    def pos(t):
        return STOP_GAP_M + jitter.setdefault(round(t / FRAME_S), rng.gauss(0.0, 0.01))
    assert _first_seen(_feed(LeadMotion(), pos, 30.0)) is None


def test_the_latch_floor_only_fills_in_what_acc_speed_cannot_see():
    assert latch_floor(0.0, 0.4) == 0.4
    assert latch_floor(0.0, None) == 0.0
    assert latch_floor(1.2, 0.4) == 1.2
    assert latch_floor(1.2, 2.0) == 1.2


def _cfg():
    return AdaptiveCruiseController().config


def test_the_pace_lift_never_lowers_the_law():
    cfg, rng = _cfg(), random.Random(3)
    for _ in range(5000):
        law = rng.uniform(-6.0, 1.5)
        args = (rng.uniform(0.5, 30.0), rng.uniform(0.0, 8.0), rng.uniform(0.0, 8.0),
                rng.uniform(-3.0, 1.5), rng.choice((0.7, 1.1, 1.5, 2.2)))
        assert pace_lift(cfg, law, *args) >= law


def test_a_stopped_lead_gets_no_lift():
    assert pace_lift(_cfg(), -0.2, 5.6, 0.0, 0.0, 0.0, 1.1) == -0.2


def test_at_rest_a_lead_seen_moving_gets_a_launch_bid():
    cfg = _cfg()
    lifted = pace_lift(cfg, -0.1, STOP_GAP_M + 0.15, 0.0, 0.15, 0.05, 1.1)
    assert lifted > cfg.standstill_launch_accel_ms2


def test_the_lift_never_pulls_inside_three_quarters_of_the_wanted_gap():
    cfg = _cfg()
    for v in (0.0, 0.5, 1.5):
        s_want = cfg.s0_m + v * 1.1
        assert pace_lift(cfg, -1.0, 0.74 * s_want, v, v + 1.0, 0.5, 1.1) == -1.0


def test_the_lift_is_gone_above_the_launch_band():
    cfg = _cfg()
    v = pull_away.PACE_ZERO_MS
    assert pace_lift(cfg, -0.5, 8.0, v, v + 2.0, 0.5, 1.1) == -0.5


def test_the_lift_never_steps():
    """Continuous in the floor, so it cannot rectify jitter into a step in the command."""
    cfg = _cfg()
    prev = None
    for k in range(401):
        v_lead = 0.8 + k * 0.001
        out = pace_lift(cfg, -0.4, 5.6, 1.2, v_lead, 0.0, 1.1)
        if prev is not None:
            assert abs(out - prev) < 0.02
        prev = out


def _tick(ctrl, now, dist_m, v_ego=0.0):
    ctrl._prev_mono = now
    snap = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=0.0, a_lead_ms2=0.0, score=6.0)
    smooth = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=0.0, a_lead_ms2=0.0, score=6.0,
                           conf=1.0, a_lead_ff_ms2=0.0)
    return ctrl._compute_command([snap], [smooth], v_ego, TICK_S)[0]


def _release_time(share: float, accel: float) -> float | None:
    """When the hold lets go behind a lead whose speed and accel both still read zero."""
    ctrl = AdaptiveCruiseController()
    ctrl.config.pull_away_share = share
    go, t = 2.0, 0.0
    while t < 8.0:
        dist = STOP_GAP_M + 0.5 * accel * max(t - go, 0.0) ** 2
        if _tick(ctrl, 1000.0 + t, dist) >= ctrl.config.standstill_launch_accel_ms2:
            return t - go
        t += TICK_S
    return None


def test_a_departing_lead_releases_the_hold_sooner():
    before, after = _release_time(0.0, 0.3), _release_time(1.0, 0.3)
    assert before is not None and after is not None
    assert after <= before - 0.5


def test_a_standing_lead_still_never_releases():
    ctrl = AdaptiveCruiseController()
    for k in range(600):
        assert _tick(ctrl, 1000.0 + k * TICK_S, STOP_GAP_M) == 0.0
