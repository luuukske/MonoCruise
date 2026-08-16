"""Gap-error shaping and the stopped-lead CAH branch.

A stopped, non-accelerating lead put both sides of the CAH branch test at zero,
so it took the 0/0 branch and reported "no braking needed". The ACC blend then
relaxed IIDM's demand onto that, holding the command near -2 m/s2 all the way
in until the TTC overlay slammed the pedal. See `core/acc/ACC_ARCHITECTURE.md`
§8.2 and §8.4."""
from __future__ import annotations

import pytest

from core.cruise_control_thread.acc_controller import (
    T_HEADWAY_BY_LEVEL_S, AdaptiveCruiseController, _LeadSnapshot,
)
from core.cruise_control_thread.idm_cah import acc_blend, cah, comfort_gain, iidm
from core.settings import Settings

DT = 1.0 / 30.0


def _gain(ctrl, dist_m, v_ego, v_lead, t_headway) -> float:
    return comfort_gain(ctrl.config, dist_m, v_ego, v_lead, t_headway)


def _unshaped(ctrl, s, v_ego, v_lead, a_lead, t_headway) -> float:
    """The IIDM+CAH blend with no gap shaping, i.e. the pre-change command."""
    cfg = ctrl.config
    return acc_blend(
        iidm(s=s, v=v_ego, v_lead=v_lead, a_max=cfg.a_max_ms2,
             b_comfort=cfg.b_comfort_ms2, s0=cfg.s0_m, t_headway=t_headway,
             v0=cfg.v0_ms, delta=cfg.delta),
        cah(s=s, v=v_ego, v_lead=v_lead, a_lead=a_lead, a_max=cfg.a_max_ms2),
        cfg.b_comfort_ms2, cfg.cool_factor_c,
    )


@pytest.fixture()
def level_two():
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = 2
    yield 2
    Settings.instance().acc_gap_level = previous


def _controller() -> AdaptiveCruiseController:
    return AdaptiveCruiseController()


def _snap(dist_m: float, v_lead: float = 0.0, a_lead: float = 0.0):
    lead = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead,
                         a_lead_ms2=a_lead, score=6.0)
    smooth = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead,
                           a_lead_ms2=a_lead, score=6.0, conf=1.0)
    return [lead], [smooth]


def _approach_stopped(level: int, v0_ms: float = 25.0, d0_m: float = 250.0):
    """Closed-loop approach to a stopped vehicle. Returns (speed, command) rows."""
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = level
    try:
        ctrl = _controller()
        v, d, t, rows = v0_ms, d0_m, 0.0, []
        while t < 45.0 and d > 0.3:
            ctrl._prev_mono = t
            raw, _ = _snap(max(d - 2.5, 0.01))
            smooth = ctrl._smooth_chain(raw, DT, t)
            a_raw, emergency = ctrl._compute_command(raw, smooth, v, DT)
            a = ctrl._output_filter(ctrl._jerk_limit(a_raw, DT, emergency),
                                    DT, emergency)
            v = max(0.0, v + max(a, -8.0) * DT)
            d -= v * DT
            rows.append((v, a, emergency))
            t += DT
            if v < 0.6 and a <= 0.0:
                break
        return rows
    finally:
        Settings.instance().acc_gap_level = previous


def test_stopped_lead_gets_the_kinematic_deceleration():
    """CAH for a stopped lead is the glide-to-stop rate, not zero."""
    for s in (200.0, 120.0, 60.0):
        expected = -(25.0 ** 2) / (2.0 * s)
        assert cah(s=s, v=25.0, v_lead=0.0, a_lead=0.0, a_max=1.5) == \
            pytest.approx(expected, rel=1e-9)


def test_cah_stays_continuous_across_the_branch_boundary():
    """The branch test must not step, or the fix trades one artefact for another."""
    v, v_lead, s = 25.0, 18.0, 60.0
    boundary = -v_lead * (v - v_lead) / (2.0 * s)
    below = cah(s=s, v=v, v_lead=v_lead, a_lead=boundary - 1e-6, a_max=1.5)
    above = cah(s=s, v=v, v_lead=v_lead, a_lead=boundary + 1e-6, a_max=1.5)
    assert below == pytest.approx(above, abs=1e-4)


def test_approach_to_a_stopped_vehicle_stays_under_comfort_braking(level_two):
    """The old law flatlined near -2 then tripped the TTC overlay at -6.55."""
    rows = _approach_stopped(2)
    moving = [r for r in rows if r[0] > 2.0]
    assert min(r[1] for r in moving) > -3.5
    assert not any(r[2] for r in moving), "no overlay trip on a plain stop"


def test_approach_to_a_stopped_vehicle_is_jerk_free(level_two):
    """The overlay bypasses the jerk limiter, so tripping it steps the pedal."""
    moving = [r for r in _approach_stopped(2) if r[0] > 2.0]
    steps = [abs(moving[i][1] - moving[i - 1][1]) / DT
             for i in range(1, len(moving))]
    assert max(steps) < AdaptiveCruiseController().config.j_max_ms3 + 0.1


def test_stopped_approach_tracks_the_kinematic_need_instead_of_a_plateau():
    """The old law sat near -2 from 120 m all the way in, whatever the gap."""
    ctrl = _controller()
    commands = []
    for s in (120.0, 90.0, 60.0, 45.0):
        needed = -(25.0 ** 2) / (2.0 * s)
        a = ctrl._lead_law(s, 25.0, 0.0, 0.0, 1.1)
        assert a <= needed + 1e-6, f"under the glide-to-stop rate at {s} m"
        commands.append(a)
    assert commands == sorted(commands, reverse=True)


def test_gain_is_unity_at_level_two():
    """Level 2 is the calibration pivot and must come out untouched."""
    ctrl = _controller()
    t_two = T_HEADWAY_BY_LEVEL_S[2]
    for v in (10.0, 20.0, 25.0):
        s_ref = ctrl.config.s0_m + v * t_two
        for s in (s_ref, 2.0 * s_ref, 6.0 * s_ref):
            assert _gain(ctrl, s, v, v, t_two) == pytest.approx(1.0)


def test_gain_comes_from_the_wanted_gap_not_the_current_one():
    """A far-off lead must not make the loop go slack. Only the setting does."""
    ctrl = _controller()
    v = 22.2
    s_ref = ctrl.config.s0_m + v * T_HEADWAY_BY_LEVEL_S[2]
    for level in (1, 2, 3, 4):
        t_headway = T_HEADWAY_BY_LEVEL_S[level]
        flat = [_gain(ctrl, s, v, v, t_headway)
                for s in (s_ref, 60.0, 90.0, 140.0, 200.0)]
        assert flat == pytest.approx([flat[0]] * len(flat))

    wide = [_gain(ctrl, 90.0, v, v, T_HEADWAY_BY_LEVEL_S[level])
            for level in (1, 2, 3, 4)]
    assert wide == sorted(wide, reverse=True), "a closer setting is the eager one"
    assert wide[0] > 1.0 > wide[2]


def test_being_inside_the_reference_gap_adds_firmness():
    ctrl = _controller()
    v, t_two = 22.2, T_HEADWAY_BY_LEVEL_S[2]
    s_ref = ctrl.config.s0_m + v * t_two
    gains = [_gain(ctrl, s, v, v, t_two) for s in (12.0, 20.0, s_ref)]
    assert gains == sorted(gains, reverse=True)
    assert gains[0] > gains[-1] == pytest.approx(1.0)
    assert _gain(ctrl, 1.0, v, v, t_two) <= ctrl.config.gap_gain_max


def test_every_gap_level_keeps_its_equilibrium():
    """Shaping changes the gain, never where the command crosses zero."""
    ctrl = _controller()
    v = 22.2
    for level in (1, 2, 3, 4):
        t_headway = T_HEADWAY_BY_LEVEL_S[level]
        s_star = ctrl.config.s0_m + v * t_headway
        assert ctrl._lead_law(s_star, v, v, 0.0, t_headway) == pytest.approx(
            0.0, abs=1e-6,
        )


def test_a_lead_pulling_away_at_the_wanted_gap_loses_most_of_its_demand():
    ctrl = _controller()
    v = 22.2
    for level in (2, 4):
        t_headway = T_HEADWAY_BY_LEVEL_S[level]
        s_want = ctrl.config.s0_m + v * t_headway
        assert _gain(ctrl, s_want, v, v + 2.5, t_headway) == pytest.approx(
            _gain(ctrl, s_want, v, v, t_headway)
            * ctrl.config.opening_gain_min, rel=1e-6,
        )


def test_a_lead_pulling_away_up_close_keeps_the_close_range_gain():
    """Inside the wanted gap the deficit is not worth leaving to the lead."""
    ctrl = _controller()
    v, t_two = 22.2, T_HEADWAY_BY_LEVEL_S[2]
    s_want = ctrl.config.s0_m + v * t_two
    tight = s_want * ctrl.config.opening_relief_fade_frac * 0.9
    assert _gain(ctrl, tight, v, v + 2.5, t_two) == pytest.approx(
        _gain(ctrl, tight, v, v, t_two), rel=1e-6,
    )
    assert _gain(ctrl, tight, v, v + 2.5, t_two) > 1.0


def test_opening_relief_is_continuous_at_zero_relative_speed():
    ctrl = _controller()
    v, s, t_two = 22.2, 40.0, T_HEADWAY_BY_LEVEL_S[2]
    assert _gain(ctrl, s, v, v + 1e-6, t_two) == pytest.approx(
        _gain(ctrl, s, v, v, t_two), rel=1e-6,
    )


def test_shaping_never_relaxes_past_the_kinematic_requirement():
    """Relief may soften the comfort term; the glide-to-stop rate holds under it."""
    ctrl = _controller()
    t_far = T_HEADWAY_BY_LEVEL_S[4]
    clamp_bit = False
    for s in (150.0, 120.0, 90.0, 60.0, 45.0):
        unshaped = _unshaped(ctrl, s, 25.0, 0.0, 0.0, t_far)
        needed = cah(s=s, v=25.0, v_lead=0.0, a_lead=0.0,
                     a_max=ctrl.config.a_max_ms2)
        gain = _gain(ctrl, s, 25.0, 0.0, t_far)
        assert gain < 1.0, "relief is live here"
        assert ctrl._lead_law(s, 25.0, 0.0, 0.0, t_far) <= needed + 1e-6
        clamp_bit |= unshaped * gain > needed
    assert clamp_bit, "no tested gap actually exercised the clamp"


def test_shaping_never_softens_a_braking_lead():
    """A decelerating lead is a hazard, not a gap error. Comfort relief stops there."""
    ctrl = _controller()
    v, s = 22.2, 20.0
    for a_lead in (-2.0, -4.0, -6.0):
        assert ctrl._lead_law(s, v, v, a_lead, 1.1) <= \
            _unshaped(ctrl, s, v, v, a_lead, 1.1) + 1e-6


def test_acceleration_is_scaled_upward_only():
    """A close setting pulls in harder; a far one keeps full pull, never less."""
    ctrl = _controller()
    v = 22.2
    for level in (1, 2, 3, 4):
        t_headway = T_HEADWAY_BY_LEVEL_S[level]
        for s in (60.0, 120.0, 200.0):
            a = ctrl._lead_law(s, v, v, 0.0, t_headway)
            reference = _unshaped(ctrl, s, v, v, 0.0, t_headway)
            assert a >= reference - 1e-9, "a far setting must not be throttled"
            if level == 1:
                assert a > reference

