"""Shift debt: through a lead's gear-shift stall the gap law sees its pull held and the speed
it lost credited back, until ego's own stall repays it. Behind a stepping TruckersMP truck ACC
used to lift off at every upshift and then stall again at its own. See
core/acc/ACC_ARCHITECTURE.md §9.11."""
from __future__ import annotations

from core.cruise_control_thread import shift_debt
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.shift_debt import OwnShiftHold, ShiftDebt

DT = 1.0 / 60.0
PULL = 0.8


def _cfg():
    return AdaptiveCruiseController().config


def _drive(lead_accel, ego_accel=None, v0=8.0, dist=25.0, s_want=18.0, ahead=None, cfg=None,
           secs=None):
    """Step a ShiftDebt along a scripted lead; returns per-tick (t, raw, out, credit)."""
    cfg = cfg or _cfg()
    sd = ShiftDebt()
    ego_accel = ego_accel or (lambda t: PULL)
    v, rows = v0, []
    n = int((secs or 12.0) / DT)
    for i in range(n):
        t = i * DT
        a = lead_accel(t)
        v += a * DT
        raw = _LeadSnapshot(vid=1, dist_m=dist, v_lead_ms=v, a_lead_ms2=a, score=6.0)
        smooth = _LeadSnapshot(vid=1, dist_m=dist, v_lead_ms=v, a_lead_ms2=a, score=6.0,
                               conf=1.0, a_lead_ff_ms2=a)
        out = sd.step(cfg, t, raw, smooth, ego_accel(t), s_want, ahead(t, v) if ahead else [])
        rows.append((t, raw, out, out.v_lead_ms - smooth.v_lead_ms))
    return rows


def _stall(t, start=3.0, length=1.0):
    """A steady pull with one upshift: drive gone in 0.3 s, back after `length`."""
    if t < start:
        return PULL
    if t < start + 0.3:
        return PULL * (1.0 - (t - start) / 0.3)
    if t < start + length:
        return 0.0
    return PULL


def _during(rows, lo, hi):
    return [r for r in rows if lo <= r[0] <= hi]


def test_a_shift_stall_holds_the_pull_and_credits_the_speed():
    rows = _drive(_stall)
    stall = _during(rows, 3.4, 4.0)
    assert all(r[2].a_lead_ms2 >= 0.7 * PULL for r in stall)
    assert max(r[3] for r in stall) > 0.2
    after = _during(rows, 4.5, 6.0)
    assert all(r[2].a_lead_ms2 == r[1].a_lead_ms2 for r in after)


def test_egos_own_stall_repays_the_credit():
    rows = _drive(_stall, ego_accel=lambda t: PULL if not 4.3 <= t < 5.3 else 0.0)
    peak = max(r[3] for r in _during(rows, 3.0, 4.3))
    assert peak > 0.2
    assert max(r[3] for r in _during(rows, 5.4, 6.0)) < 0.25 * peak


def test_a_credit_nobody_repays_runs_out():
    rows = _drive(_stall)
    assert max(r[3] for r in _during(rows, 3.0, 5.0)) > 0.2
    assert max(r[3] for r in _during(rows, 3.0 + shift_debt.CREDIT_S + 4.0, 12.0)) == 0.0


def test_a_braking_lead_is_left_alone():
    rows = _drive(lambda t: PULL if t < 3.0 else -1.0)
    for t, raw, out, credit in _during(rows, 3.0, 6.0):
        assert out.a_lead_ms2 == raw.a_lead_ms2 and credit == 0.0


def test_a_lead_easing_off_its_pull_is_not_a_stall():
    """A follower ahead settling on its own lead fades its pull out; that is no upshift."""
    rows = _drive(lambda t: PULL if t < 3.0 else max(0.0, PULL * (1.0 - (t - 3.0) / 2.5)))
    for t, raw, out, credit in rows:
        assert out.a_lead_ms2 == raw.a_lead_ms2 and credit == 0.0


def test_a_weak_pull_arms_nothing():
    rows = _drive(lambda t: 0.4 if t < 3.0 else 0.0)
    assert all(r[2].a_lead_ms2 == r[1].a_lead_ms2 and r[3] == 0.0 for r in rows)


def test_a_stall_that_outlives_the_hold_lets_go():
    rows = _drive(lambda t: _stall(t, length=6.0))
    late = _during(rows, 3.0 + shift_debt.HOLD_S + 0.4, 9.0)
    assert all(r[2].a_lead_ms2 == r[1].a_lead_ms2 for r in late)
    assert max(r[3] for r in _during(rows, 3.0 + shift_debt.HOLD_S + 3.0, 9.0)) == 0.0


def test_below_walking_pace_nothing_changes():
    rows = _drive(_stall, v0=0.2, secs=3.95)
    assert all(r[1].v_lead_ms < shift_debt.SPEED_MIN_MS for r in rows)
    assert all(r[2].a_lead_ms2 == r[1].a_lead_ms2 and r[3] == 0.0 for r in rows)


def test_a_lead_closing_on_a_slower_vehicle_gets_nothing():
    def ahead(t, v):
        return [_LeadSnapshot(vid=2, dist_m=45.0, v_lead_ms=v - 1.0, a_lead_ms2=0.0, score=6.0)]

    rows = _drive(_stall, ahead=ahead)
    assert all(r[2].a_lead_ms2 == r[1].a_lead_ms2 and r[3] == 0.0 for r in rows)


def test_the_credit_never_closes_more_than_its_share_of_the_gap():
    rows = _drive(_stall, dist=6.0, s_want=6.0, v0=5.0)
    assert max(r[3] for r in rows) <= shift_debt.CREDIT_GAP_SHARE * 6.0 + 1e-9


def test_no_credit_inside_the_wanted_gap():
    rows = _drive(_stall, dist=10.0, s_want=18.0)
    assert max(r[3] for r in rows) == 0.0


def test_zero_hold_switches_it_off():
    cfg = _cfg()
    cfg.shift_hold_s = 0.0
    rows = _drive(_stall, cfg=cfg)
    assert all(r[2].a_lead_ms2 == r[1].a_lead_ms2 and r[3] == 0.0 for r in rows)


def _own(ego_accel, law, v=10.0, cfg=None, secs=8.0):
    """Step an OwnShiftHold closed on its own output; returns per-tick (t, law, out)."""
    cfg = cfg or _cfg()
    hold, prev, rows = OwnShiftHold(), law(0.0), []
    for i in range(int(secs / DT)):
        t = i * DT
        out = hold.step(cfg, DT, ego_accel(t), prev, law(t), v)
        rows.append((t, law(t), out))
        prev = out
    return rows


def _cut(t, start=3.0, length=1.8):
    """Ego pulling at PULL with its own drive gone for `length`."""
    return 0.0 if start <= t < start + length else PULL


def _rising(t):
    """The law climbing while ego's drive is cut: the lead pulls away meanwhile."""
    return PULL if t < 3.2 else min(1.4, PULL + 0.8 * (t - 3.2))


def test_the_command_does_not_climb_while_egos_drive_is_cut():
    rows = _own(_cut, _rising)
    assert all(abs(r[2] - PULL) < 1e-9 for r in rows if 3.0 <= r[0] < 4.8)


def test_after_the_cut_the_command_climbs_back_gently():
    rows = _own(_cut, _rising)
    after = [r for r in rows if r[0] >= 4.8]
    steps = [(b[2] - a[2]) / DT for a, b in zip(after, after[1:])]
    assert max(steps) <= shift_debt.OWN_RISE_MS3 + 1e-6
    assert after[-1][2] == after[-1][1]


def test_the_law_can_always_go_lower_through_a_cut():
    rows = _own(_cut, lambda t: PULL if t < 3.3 else -1.0)
    assert all(r[2] == r[1] for r in rows if r[0] >= 3.3)


def test_a_slowdown_the_command_asked_for_is_no_cut():
    """The command falls first and ego follows: nothing is held when the law climbs again."""
    def law(t):
        return PULL if t < 3.0 else (0.1 if t < 4.5 else 1.2)
    rows = _own(lambda t: PULL if t < 3.4 else 0.1, law)
    assert all(r[2] == r[1] for r in rows)


def test_a_cut_that_outlives_the_hold_lets_go():
    rows = _own(lambda t: _cut(t, length=6.0), _rising, secs=10.0)
    late = [r for r in rows if r[0] > 3.0 + shift_debt.OWN_HOLD_S + 1.5]
    assert all(r[2] == r[1] for r in late)


def test_below_walking_pace_the_own_hold_does_nothing():
    rows = _own(_cut, _rising, v=2.0)
    assert all(r[2] == r[1] for r in rows)


def test_zero_own_hold_switches_it_off():
    cfg = _cfg()
    cfg.own_shift_hold_s = 0.0
    rows = _own(_cut, _rising, cfg=cfg)
    assert all(r[2] == r[1] for r in rows)
