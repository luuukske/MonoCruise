"""Multi-vehicle anticipation adds bounded prediction to the lead law and nothing else.

Two ways it used to do more, both replayed on the clip corpus: its sum could set
the emergency flag, which skips the jerk limit, and the virtual lead was
differenced against ``a_base``, re-imposing the full law on a low-confidence
lead. It also brakes no harder than ``ant_brake_floor_ms2`` on its own. See
core/acc/ACC_ARCHITECTURE.md §9."""
from __future__ import annotations

from core.cruise_control_thread import anticipation
from core.cruise_control_thread.acc_controller import (
    TTC_MIN_VCLOSE_MS, AdaptiveCruiseController, _LeadSnapshot,
)

DT = 1.0 / 30.0


def _conf(score: float) -> float:
    cfg = AdaptiveCruiseController().config
    span = cfg.ant_score_full - cfg.ant_score_min
    return max(0.0, min(1.0, (score - cfg.ant_score_min) / span))


def _chain(*leads: tuple[int, float, float, float, float]):
    """(vid, dist_m, v_lead, a_lead, score) per lead, nearest first."""
    raw = [_LeadSnapshot(vid, d, v, a, s) for vid, d, v, a, s in leads]
    smooth = [_LeadSnapshot(vid, d, v, a, s, conf=_conf(s), a_lead_ff_ms2=a)
              for vid, d, v, a, s in leads]
    return raw, smooth


def _run(controller, raw, smooth, v_ego: float, ticks: int = 90):
    out = []
    for _ in range(ticks):
        out.append(controller._compute_command(raw, smooth, v_ego, DT))
    return out


def test_anticipation_never_raises_the_emergency_flag():
    """Even with its floor opened to the clamp, anticipation stays on the jerk limit."""
    controller = AdaptiveCruiseController()
    cfg = controller.config
    cfg.ant_brake_floor_ms2 = cfg.max_decel_ms2
    raw, smooth = _chain((1, 40.0, 25.0, 0.0, 6.0), (2, 52.0, 0.0, 0.0, 6.0))
    ticks = _run(controller, raw, smooth, v_ego=25.0, ticks=45)
    # The lead then brakes too: the filtered delta still carries the old gap to its law.
    raw, smooth = _chain((1, 36.0, 22.0, -2.0, 6.0), (2, 48.0, 0.0, 0.0, 6.0))
    ticks += _run(controller, raw, smooth, v_ego=25.0, ticks=15)
    assert min(a for a, _ in ticks) <= cfg.max_decel_ms2 + 1e-6, (
        "anticipation lost its authority: the stopped car should still reach the clamp"
    )
    assert not any(emergency for _, emergency in ticks)


def test_anticipation_alone_brakes_no_harder_than_its_floor():
    """A stopped car past a steady lead eases ego down to the floor, never past it."""
    controller = AdaptiveCruiseController()
    floor = controller.config.ant_brake_floor_ms2
    raw, smooth = _chain((1, 40.0, 25.0, 0.0, 6.0), (2, 52.0, 0.0, 0.0, 6.0))
    ticks = _run(controller, raw, smooth, v_ego=25.0)
    lowest = min(a for a, _ in ticks)
    assert lowest >= floor - 1e-9
    assert lowest < floor + 0.05, "anticipation should still brake down to the floor"


def test_floor_leaves_the_immediate_lead_alone():
    """Past the floor the lead law commands on its own, unsoftened and unsharpened."""
    controller = AdaptiveCruiseController()
    raw, smooth = _chain((1, 30.0, 12.0, -1.0, 6.0), (2, 45.0, 0.0, 0.0, 6.0))
    alone = AdaptiveCruiseController()
    ticks = _run(controller, raw, smooth, v_ego=25.0, ticks=30)
    solo = _run(alone, raw[:1], smooth[:1], v_ego=25.0, ticks=30)
    assert solo[-1][0] < controller.config.ant_brake_floor_ms2
    assert ticks[-1][0] == solo[-1][0]


def test_immediate_lead_at_the_clamp_is_still_an_emergency():
    controller = AdaptiveCruiseController()
    raw, smooth = _chain((1, 30.0, 0.0, 0.0, 6.0), (2, 45.0, 0.0, 0.0, 6.0))
    accel, emergency = controller._compute_command(raw, smooth, 25.0, DT)
    assert emergency
    assert accel == controller.config.max_decel_ms2


def test_virtual_lead_does_not_undo_the_confidence_blend():
    """A barely-scored slower car beside the real lead must stay softened.

    The confidence blend hands most of the command to the lead behind it; the
    virtual lead used to add the uncertain car's full law straight back."""
    controller = AdaptiveCruiseController()
    raw, smooth = _chain((1, 30.0, 15.0, 0.0, 0.6), (2, 45.0, 25.0, 0.0, 6.0))
    _run(controller, raw, smooth, v_ego=25.0)
    assert controller._ant_delta_ms2 > -0.5


def test_virtual_lead_is_differenced_against_the_lead_law_not_a_base():
    """With no upstream weight the delta is zero whatever arbitration did to a_base."""
    cfg = AdaptiveCruiseController().config
    raw, smooth = _chain((1, 35.0, 20.0, 0.0, 6.0), (2, 60.0, 5.0, 0.0, 0.51))
    for a_base in (-3.0, -1.0, 0.5):
        delta = anticipation.anticipation_delta(
            cfg, raw, smooth, 25.0, a_base, 1.1, TTC_MIN_VCLOSE_MS)
        assert abs(delta) < 0.05, a_base


def test_braking_wave_ahead_of_the_lead_is_still_anticipated():
    cfg = AdaptiveCruiseController().config
    raw, smooth = _chain((1, 37.0, 25.0, 0.0, 6.0), (2, 62.0, 20.0, -3.0, 6.0))
    controller = AdaptiveCruiseController()
    a_base = controller._lead_law(37.0, 25.0, 25.0, 0.0, 1.1, 0.0)
    delta = anticipation.anticipation_delta(
        cfg, raw, smooth, 25.0, a_base, 1.1, TTC_MIN_VCLOSE_MS)
    assert delta < -0.3


def test_the_controller_chain_fits_what_the_tracker_publishes():
    """The chain is cut from leads[]; asking for more than the tracker publishes changes nothing."""
    from core.acc import tracker
    from core.cruise_control_thread import acc_controller
    from tools.acc_platoon import stack

    assert acc_controller.MA_MAX_LEADS <= tracker.PUBLISHED_LEADS
    assert stack.TRACKER_LEADS == tracker.PUBLISHED_LEADS
