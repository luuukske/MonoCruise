"""Gap-error shaping and the stopped-lead CAH branch.

A stopped, non-accelerating lead put both sides of the CAH branch test at zero,
so it took the 0/0 branch and reported "no braking needed". The ACC blend then
relaxed IIDM's demand onto that, holding the command near -2 m/s2 all the way
in until the TTC overlay slammed the pedal. See `core/acc/ACC_ARCHITECTURE.md`
§8.2 and §8.4."""
from __future__ import annotations

import pathlib

import pytest

from core.cruise_control_thread.acc_controller import (
    T_HEADWAY_BY_LEVEL_S, AdaptiveCruiseController, _LeadSnapshot,
)
from core.cruise_control_thread.idm_cah import (
    LEAD_BRAKE_FF_MAX_MS2, LEAD_BRAKE_FF_SHARE, _soft_min, _soft_negative, acc_blend, alead_tau_s, cah,
    comfort_gain, iidm, lead_accel_nudge, lead_brake_ff,
)
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


def _brake_from_matched(lead_decel: float, level: int = 2, v0_ms: float = 22.2):
    """Lead brakes from the level's own equilibrium. Returns (peak decel, min gap).

    Starting anywhere else measures the gap-closing transient on top."""
    ctrl = _controller()
    v_ego = v_lead = v0_ms
    gap0_m = ctrl.config.s0_m + v0_ms * T_HEADWAY_BY_LEVEL_S[level]
    gap, t, peak, min_gap = gap0_m, 0.0, 0.0, gap0_m
    while t < 12.0 and gap > 0.5 and v_ego > 0.05:
        ctrl._prev_mono = t
        raw, _ = _snap(gap, v_lead=v_lead, a_lead=-lead_decel)
        smooth = ctrl._smooth_chain(raw, DT, t)
        a_raw, emergency = ctrl._compute_command(raw, smooth, v_ego, DT)
        a = ctrl._output_filter(ctrl._jerk_limit(a_raw, DT, emergency), DT, emergency)
        a = min(0.0, a)
        peak = min(peak, a)
        v_ego = max(0.0, v_ego + a * DT)
        v_lead = max(0.0, v_lead - lead_decel * DT)
        gap += (v_lead - v_ego) * DT
        min_gap = min(min_gap, gap)
        t += DT
    return peak, min_gap


def test_feedforward_is_silent_for_a_lead_that_is_not_braking():
    """It must move the a_lead axis and nothing else, or it shifts equilibrium."""
    ctrl = _controller()
    for a_lead in (0.0, 0.5, 1.5):
        for s, v_lead in ((30.0, 22.2), (60.0, 18.0), (120.0, 0.0)):
            assert lead_brake_ff(ctrl.config, s, 22.2, v_lead, a_lead) == 0.0


def test_feedforward_is_bounded():
    """A spurious a_lead spike must cost one comfort brake, never a slam."""
    ctrl = _controller()
    for a_lead in (-4.0, -8.0, -20.0):
        for s in (10.0, 40.0, 150.0):
            ff = lead_brake_ff(ctrl.config, s, 22.2, 20.0, a_lead)
            assert -LEAD_BRAKE_FF_MAX_MS2 - 1e-9 <= ff <= 0.0


def test_command_responds_to_lead_braking_at_the_wanted_gap():
    """§8.5: the blend discarded a_cah here, freezing the command above ~2 m/s2."""
    ctrl = _controller()
    v, t_three = 22.2, T_HEADWAY_BY_LEVEL_S[3]
    s_want = ctrl.config.s0_m + v * t_three
    commands = [ctrl._lead_law(s_want, v, v - 2.0, -a, t_three)
                for a in (0.0, 2.0, 4.0, 6.0)]
    assert commands == sorted(commands, reverse=True)
    assert commands[0] - commands[-1] > 0.75, "lead braking still barely reaches it"


def test_a_braking_lead_that_is_pulling_away_still_cuts_the_pull():
    """The cheapest anticipation window: closing speed has not built yet."""
    ctrl = _controller()
    v, t_three = 22.2, T_HEADWAY_BY_LEVEL_S[3]
    coasting = ctrl._lead_law(40.0, v, v + 4.0, 0.0, t_three)
    braking = ctrl._lead_law(40.0, v, v + 4.0, -6.0, t_three)
    assert coasting > 0.0, "a lead pulling away should still allow pull"
    assert braking < coasting * 0.5


def test_lead_law_is_continuous_across_the_cah_branch_test():
    """The feedforward sits outside the blend so it cannot step at the branch."""
    ctrl = _controller()
    v, v_lead, s = 22.2, 18.0, 45.0
    boundary = -v_lead * (v - v_lead) / (2.0 * s)
    below = ctrl._lead_law(s, v, v_lead, boundary - 1e-6, 1.5)
    above = ctrl._lead_law(s, v, v_lead, boundary + 1e-6, 1.5)
    assert below == pytest.approx(above, abs=1e-4)


@pytest.mark.parametrize("lead_decel", [2.0, 4.0, 6.0, 8.0])
def test_following_a_braking_lead_is_string_stable(level_two, lead_decel):
    """Ego must not amplify the lead's deceleration back up the platoon (§5)."""
    peak, _ = _brake_from_matched(lead_decel)
    assert -peak <= lead_decel + 1e-6, "ego out-braked the disturbance"


@pytest.mark.parametrize("lead_decel", [5.0, 6.0, 7.0])
def test_a_hard_braking_lead_no_longer_needs_the_ttc_overlay(level_two, lead_decel):
    """The overlay slams unfiltered, which is the shape §4 exists to avoid."""
    peak, _ = _brake_from_matched(lead_decel)
    ctrl = _controller()
    assert peak > ctrl.config.max_decel_ms2 + 0.05


def _gain_jumps(ctrl, decels, s=37.5, v=22.22, closing=0.5, t_headway=1.5):
    """Adjacent-sample changes in d(cap)/d(a_lead) across a fine sweep."""
    caps = [ctrl._lead_law(s, v, v - closing, -d, t_headway) for d in decels]
    step = decels[1] - decels[0]
    gain = [(caps[i + 1] - caps[i]) / step for i in range(len(caps) - 1)]
    return [abs(gain[i + 1] - gain[i]) for i in range(len(gain) - 1)]


def test_soft_min_never_relaxes_the_value_it_clamps():
    """It may only deviate downward, or the kinematic floor stops holding."""
    for a in (-6.0, -1.0, -0.05, 0.0, 0.4):
        for b in (-6.0, -1.0, -0.05, 0.0, 0.4):
            for eps in (0.0, 0.02, 0.12, 0.5):
                assert _soft_min(a, b, eps) <= min(a, b) + 1e-12


def test_soft_negative_is_c1_monotone_and_never_positive():
    eps = 0.5
    xs = [i / 500.0 for i in range(-1000, 400)]
    vals = [_soft_negative(x, eps) for x in xs]
    assert max(vals) <= 0.0
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    slopes = [(vals[i + 1] - vals[i]) / (xs[1] - xs[0]) for i in range(len(vals) - 1)]
    assert max(abs(b - a) for a, b in zip(slopes, slopes[1:])) < 0.02
    assert _soft_negative(-2.0, eps) == pytest.approx(-2.0)


def test_alead_tau_ramps_instead_of_switching():
    """A lead on the deadband edge must not re-pick a 4x bandwidth every tick."""
    cfg = _controller().config
    xs = [i / 400.0 for i in range(-800, 800)]
    taus = [alead_tau_s(cfg, x) for x in xs]
    assert alead_tau_s(cfg, 0.0) == pytest.approx(cfg.tau_alead_relax_s)
    assert alead_tau_s(cfg, -4.0) == pytest.approx(cfg.tau_alead_brake_s)
    assert all(b >= a - 1e-12 for a, b in zip(taus, taus[1:])), "monotone in braking"
    assert max(abs(b - a) for a, b in zip(taus, taus[1:])) < 0.01, "tau still steps"


def test_no_step_in_gain_through_the_small_braking_zone():
    """Hard clamps here made the command jump as the lead dipped in and out."""
    ctrl = _controller()
    decels = [(-0.6 + i * 0.0025) for i in range(881)]
    assert max(_gain_jumps(ctrl, decels)) < 0.20


def test_softening_off_reproduces_the_old_hard_transitions():
    """The probe's `before` column depends on 0 restoring the old behaviour."""
    ctrl = _controller()
    decels = [(-0.6 + i * 0.0025) for i in range(881)]
    smooth = max(_gain_jumps(ctrl, decels))
    ctrl.config.lead_law_floor_soft_ms2 = 0.0
    ctrl.config.lead_brake_ff_soft_ms2 = 0.0
    assert max(_gain_jumps(ctrl, decels)) > smooth * 1.4


def test_accel_nudge_is_silent_unless_the_lead_is_accelerating():
    """It must not touch the braking side, which carries the safety argument."""
    cfg = _controller().config
    for a_lead in (0.0, -0.5, -2.0, -8.0):
        assert lead_accel_nudge(cfg, 22.2, 20.0, a_lead) == 0.0


def test_accel_nudge_is_bounded_and_capped_at_ego_authority():
    cfg = _controller().config
    for a_lead in (0.5, 1.5, 4.0, 20.0):
        n = lead_accel_nudge(cfg, 22.2, 22.2, a_lead)
        assert 0.0 <= n <= cfg.lead_accel_nudge_max_ms2 + 1e-12
    assert lead_accel_nudge(cfg, 22.2, 22.2, 4.0) == pytest.approx(
        lead_accel_nudge(cfg, 22.2, 22.2, 20.0)), "must saturate at a_max"


def test_accel_nudge_gates_off_while_closing():
    """A phantom a_lead must never add pull toward a lead ego is catching."""
    cfg = _controller().config
    v = 22.2
    closing = [lead_accel_nudge(cfg, v, v - dv, 1.0)
               for dv in (0.0, 0.5, 1.0, 1.5, 2.0, 4.0)]
    assert closing == sorted(closing, reverse=True)
    assert closing[0] > 0.2
    assert closing[-2] == 0.0 and closing[-1] == 0.0


def test_a_gently_accelerating_lead_is_distinguishable_from_a_coasting_one():
    """§8.8 said flat was defensible; §8.9 overrode that for feel."""
    ctrl = _controller()
    v, t_three = 22.2, T_HEADWAY_BY_LEVEL_S[3]
    s_want = ctrl.config.s0_m + v * t_three
    coasting = ctrl._lead_law(s_want, v, v, 0.0, t_three)
    assert coasting == pytest.approx(0.0, abs=1e-6), "equilibrium still exact"
    for a_lead, floor in ((0.25, 0.03), (0.5, 0.08), (1.0, 0.20)):
        assert ctrl._lead_law(s_want, v, v, a_lead, t_three) > floor


def _probe_baseline() -> dict:
    """Read BASELINE from the probe source; importing it needs tools/ on sys.path."""
    import ast
    src = (pathlib.Path(__file__).resolve().parents[2]
           / "tools" / "acc_transition_probe.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id == "BASELINE":
            return ast.literal_eval(node.value)
    raise AssertionError("tools/acc_transition_probe.py defines no BASELINE")


def test_feedforward_reads_its_own_a_lead_estimate_not_cah_s():
    """CAH keeps the fast filter, the feedforwards get a slower one. See §8.10."""
    ctrl = _controller()
    cfg = ctrl.config
    s, v, v_lead, T = 37.5, 22.2, 20.2, T_HEADWAY_BY_LEVEL_S[3]

    braking = ctrl._lead_law(s, v, v_lead, -6.0, T)
    ff_still_coasting = ctrl._lead_law(s, v, v_lead, -6.0, T, 0.0)
    no_brake_at_all = ctrl._lead_law(s, v, v_lead, 0.0, T)

    assert ff_still_coasting > braking, "a lagging ff estimate must soften the term"
    assert ff_still_coasting < no_brake_at_all, "CAH must still see the real a_lead"
    # A coasting ff estimate must be exactly equivalent to the term being off.
    cfg.lead_brake_ff_share = 0.0
    assert ctrl._lead_law(s, v, v_lead, -6.0, T) == pytest.approx(
        ff_still_coasting, abs=1e-9)
    cfg.lead_brake_ff_share = LEAD_BRAKE_FF_SHARE
    assert ctrl._lead_law(s, v, v_lead, -6.0, T, None) == braking
    assert cfg.tau_alead_ff_s > cfg.tau_alead_brake_s, "ff filter must be the slower one"


def test_every_feature_knob_disables_its_feature_at_zero():
    """`tools/acc_transition_probe.BASELINE` is only a baseline if this holds.

    It once omitted the two shares, so the probe compared the change to itself."""
    ctrl = _controller()
    baseline = _probe_baseline()
    for key, value in baseline.items():
        assert hasattr(ctrl.config, key), f"BASELINE names a dead field {key!r}"
        setattr(ctrl.config, key, value)
    cfg = ctrl.config
    for s, v, v_lead, a_lead in ((37.5, 22.2, 22.2, -4.0), (20.0, 13.9, 10.0, -1.0),
                                 (60.0, 25.0, 27.0, 1.0), (12.0, 8.0, 0.0, 0.2)):
        assert lead_brake_ff(cfg, s, v, v_lead, a_lead) == 0.0
        assert lead_accel_nudge(cfg, v, v_lead, a_lead) == 0.0
    for a, b in ((-1.0, -2.0), (0.5, 0.5), (-0.05, 0.0)):
        assert _soft_min(a, b, cfg.lead_law_floor_soft_ms2) == min(a, b)
    for d in (-1.0, -0.2, 0.0, 0.3):
        assert _soft_negative(d, cfg.lead_brake_ff_soft_ms2) == min(0.0, d)
    hard = cfg.a_lead_deadband_ms2
    assert alead_tau_s(cfg, -hard - 1e-6) == pytest.approx(cfg.tau_alead_brake_s)
    assert alead_tau_s(cfg, -hard + 1e-6) == pytest.approx(cfg.tau_alead_relax_s)


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

