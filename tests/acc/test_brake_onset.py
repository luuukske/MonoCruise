"""A gap law at the clamp is reached fast, but never in one tick. See ACC_ARCHITECTURE.md §13.2.

The at-clamp branch used to set the emergency flag, which skips the jerk limit and
the output filter. The law got there behind a 2.5 m/s^3 ramp, so the pedal crept
and then jumped the rest of the way to full brake in a single tick: the stomp."""
from __future__ import annotations

from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.idm_cah import jerk_step

DT = 1.0 / 30.0


def _chain(dist_m: float, v_lead: float, a_lead: float):
    raw = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead, a_lead_ms2=a_lead, score=6.0)
    smooth = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead, a_lead_ms2=a_lead,
                           score=6.0, conf=1.0, a_lead_ff_ms2=a_lead)
    return [raw], [smooth]


def _dive(ctrl: AdaptiveCruiseController, ticks: int = 30):
    """Settled behind a steady lead, then the lead reads a hard brake 35 m ahead (TTC 3.7 s)."""
    out = []
    for i in range(30 + ticks):
        if i < 30:
            raw, smooth = _chain(45.0, 24.5, 0.0)
        else:
            raw, smooth = _chain(35.0, 15.0, -5.0)
        a_raw, em = ctrl._compute_command(raw, smooth, 24.5, DT)
        cap = ctrl._output_filter(ctrl._jerk_limit(a_raw, DT, em, 24.5), DT, em)
        if i >= 29:
            out.append((cap, em))
    return out


def test_the_law_at_the_clamp_is_not_an_emergency():
    ctrl = AdaptiveCruiseController()
    raw, smooth = _chain(35.0, 15.0, -5.0)
    ctrl._compute_command(raw, smooth, 24.5, DT)
    a_raw, em = ctrl._compute_command(raw, smooth, 24.5, DT)
    assert a_raw <= ctrl.config.max_decel_ms2 + 1e-6
    assert not em


def test_full_brake_arrives_quickly_but_never_in_one_tick():
    ctrl = AdaptiveCruiseController()
    caps = [c for c, _ in _dive(ctrl)]
    drops = [a - b for a, b in zip(caps, caps[1:])]
    assert max(drops) < 1.5, "a single tick stepped the brake"
    cfg = ctrl.config
    # Chased to within its dead band of the clamp, then the plain rate covers the rest.
    t_near = next(i for i, c in enumerate(caps) if c <= cfg.max_decel_ms2 + cfg.j_onset_dead_ms2) * DT
    assert t_near < 0.5, "the onset chase must still get near the clamp promptly"


def test_the_old_snap_is_one_switch_away():
    """The A/B switch reproduces the stomp, so probes can compare against it."""
    ctrl = AdaptiveCruiseController()
    ctrl.config.at_clamp_slam = True
    caps = [c for c, _ in _dive(ctrl)]
    assert max(a - b for a, b in zip(caps, caps[1:])) > 4.0


def test_the_ttc_overlay_still_slams():
    """An imminent collision keeps its single-tick full brake."""
    ctrl = AdaptiveCruiseController()
    raw, smooth = _chain(12.0, 5.0, 0.0)
    a_raw, em = ctrl._compute_command(raw, smooth, 20.0, DT)
    assert em and a_raw == ctrl.config.max_decel_ms2


def test_onset_chase_off_is_the_plain_limiter_bit_for_bit():
    j = AdaptiveCruiseController().config.j_max_ms3
    for prev, target in ((0.0, -6.55), (-2.0, -6.55), (-1.0, -1.5), (1.0, -3.0)):
        plain = prev + max(target - prev, -j * DT)
        assert jerk_step(prev, target, DT, j, 0.0) == plain
        assert jerk_step(prev, target, DT, j, 0.0, 1.0, 0.0, 1.0) == plain


def test_onset_chase_is_the_plain_limiter_inside_its_dead_band():
    cfg = AdaptiveCruiseController().config
    prev = -2.0
    target = prev - cfg.j_onset_dead_ms2 * 0.9
    assert jerk_step(prev, target, DT, cfg.j_max_ms3, 0.0, 1.0,
                     cfg.j_onset_tau_s, cfg.j_onset_dead_ms2) == prev - cfg.j_max_ms3 * DT
