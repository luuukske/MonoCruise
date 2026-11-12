"""Lane keeping assist: short steer-backs where the lane departure rumble starts, run on the LA core."""

from __future__ import annotations

import math
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.la.controller import AssistTuning, Gains, LaneAssist, Wheel
from core.la.hand_push import HandPush, HandPushEstimator, PushTuning
from core.lane_departure import command as cmd
from core.lane_departure.detector import (
    LEVEL_CLOSE, LEVEL_WARN, SIDE_LEFT, SIDE_NONE, SIDE_RIGHT, Decision, LaneDetector, Tuning, _Edge,
)
from core.lane_departure.lka import LaneKeeping, LkaTuning
from core.settings import Settings
from tests.la.servo_sim import Rig
from tests.la.test_thread_assist import FORCE, GAIN, servo_in, wheel
from tests.lane_departure.test_detector import DRIFT_START, DT, HALF_WIDTH, SPEED, frame_at, make_route
from tests.lane_departure.test_thread import World

PULSE = AssistTuning(pulse=True, settle_s=1.0)
LINE = Tuning().half_lane_m - HALF_WIDTH
RESTING = dict(stick=350.0, hand_mass=3.0, hand_damp=1.5)


def keep(drift=None, radius=None, seconds=12.0, hands=0.0, hands_at=None, driver=None, pushes=None, feel=False,
         decisions=None, **frame_kw):
    """Detector, pulse trigger and the LA core in pulse mode, closed loop on the modelled wheel and truck.

    ``drift`` kicks the truck sideways at 3 s; ``driver(t, rig)`` moves the hands. Rows: ``(t, e, ed, out, pulse)``.
    ``pushes`` collects ``(t, servo torque counterclockwise, the pulse's side)`` from the modelled plugin. ``feel``
    hands the core the driver's push estimated as the runner does, from the plugin's readings, so a pulse yields to it.
    ``decisions`` collects the detector's decision per tick, what the rumble follows.
    """
    rig = Rig(radius=radius, **{k: frame_kw.pop(k) for k in ("stick", "hand_mass", "hand_damp", "yaw_lag_s")
                                if k in frame_kw})
    rig.settle_on_lane()
    rig.hands, rig.hands_at = hands, rig.steer if hands_at is None else hands_at
    assist, det, keeping = LaneAssist(PULSE), LaneDetector(), LaneKeeping()
    gains = Gains(rig.spring / (rig.gain * SPEED * SPEED), rig.gain)
    route = make_route(radius, length=2500.0 if radius is None else min(2500.0, 1.5 * math.pi * abs(radius)))
    rows, out, hand = [], None, None
    est = HandPushEstimator(PushTuning(inertia=rig.inertia * (1.0 + rig.hand_mass)))
    rate_lp, game_lp = 0.0, rig.game_force
    for i in range(int(seconds / DT)):
        t = i * DT
        if drift is not None and abs(t - 3.0) < 1e-6:
            rig.ed = drift
        if driver is not None:
            driver(t, rig)
        f = frame_at(t, route, radius, rig.e, rig.psi, steer_right=-rig.steer, **frame_kw)
        decision = det.update(f)
        if decisions is not None:
            decisions.append(decision)
        pulse = keeping.update(t, decision, det, f, out)
        wheel_now = Wheel(ok=True, steer=rig.steer, rate=rig.plugin.rate, push=rig.plugin.servo_applied,
                          buzz=rig.plugin.scale, game=rig.game_force)
        out = assist.update(f, gains, wheel_now, engage=pulse.side, hand=hand if feel else None)
        rows.append((t, rig.e, rig.ed, out, pulse))
        rig.run(out.servo, out.torque, DT)
        rate_lp += (rig.rate - rate_lp) * (1.0 - math.exp(-DT / 0.04))
        game_lp += (rig.game_force - game_lp) * (1.0 - math.exp(-DT / 0.10))
        hand = est.update(DT if i else 0.0, rate_lp, game_lp, rig.plugin.value + rig.plugin.base, 0.0, 1,
                          out.servo.only)
        if pushes is not None:
            pushes.append((t, rig.plugin.value * (1 if gains.force_per_ms2 > 0.0 else -1), out.servo.only))
    return rows


def past_line(rows) -> float:
    return max(abs(e) for _, e, *_ in rows) - LINE


def events(rows) -> list[tuple[float, str]]:
    return [(t, p.event) for t, _, _, _, p in rows if p.event]


@pytest.mark.parametrize("drift", [0.3, 0.5, 0.8, 1.2])
def test_a_drift_on_a_straight_is_stopped_at_the_line_and_the_swing_home_caught_at_the_other(drift):
    rows = keep(drift, seconds=16.0)
    assert past_line(rows) < 0.3, "Euro NCAP's lane keeping bar: no more than 0.3 m past either line"
    (start, first), (end, why) = events(rows)[:2]
    assert first == "start" and why == "back" and end - start <= 3.0


@pytest.mark.parametrize("radius, worst", [(400.0, 0.3), (-400.0, 0.3), (250.0, 0.5)])
def test_running_wide_in_a_bend_is_caught_before_the_line(radius, worst):
    rows = keep(None, radius=radius, hands=20000.0, hands_at=0.9 * (-1.0 / radius / GAIN), **RESTING)
    first = next(i for i, r in enumerate(rows) if r[4].event == "start")
    assert abs(rows[first][1]) < LINE, "the approach cue starts it in a bend too, inside the line"
    assert past_line(rows) < worst


def test_a_pulse_starts_only_where_the_detector_starts_so_the_rumble_and_the_pulse_share_one_trigger():
    """Edges close enough for an approach start: the pulse still waits for the detector, which the rumble follows."""
    det = _detector()
    det.edges = {SIDE_RIGHT: _edge(margin=0.2, drift=0.6, tlc=0.4), SIDE_LEFT: _edge(margin=2.0, drift=-0.6)}
    steered = SimpleNamespace(event="", reason="standby", offset_m=0.6, drift_ms=0.6, steer=0.0)
    assert LaneKeeping().update(1.0, Decision(), det, _frame(), steered).side == SIDE_NONE
    for level in (LEVEL_CLOSE, LEVEL_WARN):
        started = Decision(started=True, side=SIDE_RIGHT, level=level)
        assert LaneKeeping().update(1.0, started, det, _frame(), steered).event == "start"


@pytest.mark.parametrize("kw", [
    dict(drift=0.5), dict(drift=1.2),
    dict(radius=400.0, hands=20000.0, hands_at=0.9 * (-1.0 / 400.0 / GAIN), **RESTING),
    dict(radius=-250.0, hands=20000.0, hands_at=0.9 * (1.0 / 250.0 / GAIN), **RESTING),
])
def test_every_pulse_starts_with_the_detector_or_catches_the_swing_after_one(kw):
    """One trigger: a pulse starts on the tick the detector starts, on its side, or within the catch after a pulse."""
    decisions = []
    rows = keep(seconds=16.0, decisions=decisions, **kw)
    starts = [(i, r[4].side) for i, r in enumerate(rows) if r[4].event == "start"]
    ended = [r[0] for r in rows if r[4].event and r[4].event not in ("start", "skipped")]
    assert starts
    for i, side in starts:
        d, t = decisions[i], rows[i][0]
        caught = any(0.0 <= t - end <= LkaTuning().catch_s for end in ended)
        assert (d.started and d.side == side) or caught
    for i, d in enumerate(decisions):
        if d.started and rows[i][4].side == SIDE_NONE:
            assert rows[i][4].event in ("skipped", "back", "override", "driver", "timeout", "out")


@pytest.mark.parametrize("radius", [250.0, -250.0])
@pytest.mark.parametrize("lag", [0.1, 0.05])
def test_a_sleeping_driver_in_a_bend_on_a_quick_truck_stays_near_the_lane(radius, lag):
    """The game's truck follows its wheel within a tick. Released early, the wheel came back toward the line, the detector
    read it as intent and latched, and the truck ran off the road with no pulse and no rumble.
    """
    rows = keep(None, radius=radius, seconds=16.0, hands=20000.0, hands_at=0.85 * (-1.0 / radius / GAIN),
                yaw_lag_s=lag, **RESTING)
    assert past_line(rows) < 0.5


def test_a_pulse_holds_on_while_the_truck_itself_still_heads_out():
    """In a bend the cab's corner reads home as soon as it turns in; nine 0.2 s pulses in a row on 2026-11-02."""
    keeping = LaneKeeping()
    keeping.update(0.0, START, _detector(), _frame(), None)
    det = _detector()
    det.edges = {SIDE_RIGHT: _edge(margin=0.3, drift=-0.1)}
    steered = SimpleNamespace(event="", reason="pulse", offset_m=0.5, drift_ms=0.3, steer=0.0)
    assert keeping.update(0.2, Decision(), det, _frame(), steered).event == ""
    steered.drift_ms = 0.0
    assert keeping.update(0.4, Decision(), det, _frame(), steered).event == "back"


def test_no_pulse_starts_while_the_core_still_holds_off_for_the_driver():
    held = SimpleNamespace(event="", reason="override", offset_m=0.0, drift_ms=0.0, steer=0.0)
    pulse = LaneKeeping().update(1.0, START, _detector(), _frame(), held)
    assert (pulse.side, pulse.event, pulse.reason) == (SIDE_NONE, "skipped", "override")


@pytest.mark.parametrize("steer, starts", [(0.025, True), (0.0, False)])
def test_the_wheel_coming_back_after_a_pulse_is_not_the_driver(steer, starts):
    """The pulse turned the wheel from 0.02 to 0.05 (counterclockwise, away from the right line). Coming back to 0.025
    is its own wheel; past 0.02 toward the line is the driver.
    """
    keeping, det = LaneKeeping(), _detector()
    steered = SimpleNamespace(event="", reason="pulse", offset_m=0.5, drift_ms=0.0, steer=0.02)
    keeping.update(0.0, START, det, _frame(), steered)
    det.edges = {SIDE_RIGHT: _edge(margin=0.3, drift=-0.5)}
    steered.steer = 0.05
    assert keeping.update(1.0, Decision(), det, _frame(), steered).event == "back"
    det.edges = {SIDE_RIGHT: _edge(margin=0.3, drift=0.6, tlc=0.5, hard="intent")}
    steered.steer, steered.drift_ms = steer, 0.3
    pulse = keeping.update(1.5, Decision(), det, _frame(), steered)
    assert (pulse.event == "start") == starts
    if starts:
        assert keeping.update(1.6, Decision(), det, _frame(), steered).event == "", "and it does not end it either"


def test_a_blinker_toward_the_line_means_no_pulse():
    rows = keep(0.5, blinker_right=True)
    assert not events(rows)
    assert past_line(rows) > 1.0


def test_steering_toward_the_line_ends_a_pulse_at_once():
    def driver(t, rig):
        if t >= 4.2:
            rig.hands, rig.hands_at = 60000.0, -0.08
    rows = keep(0.5, driver=driver)
    start = next(t for t, why in events(rows) if why == "start")
    end, why = next((t, why) for t, why in events(rows) if why != "start")
    assert why in ("driver", "override") and end - 4.2 < 0.6
    assert all(o.authority == 0.0 for t, _, _, o, _ in rows if t > end + 0.5), "faded out within half a second"
    assert start < 4.2


def _edge(margin=0.5, drift=0.0, tlc=math.inf, bend=0.0, hard="", gap=None):
    gap = margin + 0.1 if gap is None else gap
    return _Edge(gap, margin - gap, bend, drift, tlc, drift <= -0.2, drift >= 0.4, hard, True, True)


def _frame(kind="road"):
    route = SimpleNamespace(pts=((0.0, 0.0), (0.0, -100.0)), kind=kind)
    return SimpleNamespace(route=route, x=0.0, z=-10.0, front_m=3.3, speed_ms=SPEED, lanes=None)


def _detector(**edges):
    return SimpleNamespace(evade_until=-1e9, tune=Tuning(), edges=edges)


START = Decision(started=True, side=SIDE_RIGHT)


def test_a_departure_the_detector_starts_starts_a_pulse_and_names_what_lies_past_the_line():
    pulse = LaneKeeping().update(1.0, START, _detector(), _frame(), None)
    assert (pulse.side, pulse.event, pulse.beyond) == (SIDE_RIGHT, "start", "unknown")


def test_a_pulse_ends_once_back_inside_and_no_longer_drifting_out_and_never_runs_past_its_limit():
    keeping, tn = LaneKeeping(), LkaTuning()
    keeping.update(0.0, START, _detector(), _frame(), None)
    det = _detector()
    det.edges = {SIDE_RIGHT: _edge(margin=0.3, drift=0.4)}
    assert keeping.update(0.5, Decision(), det, _frame(), None).side == SIDE_RIGHT, "inside, still drifting out"
    det.edges = {SIDE_RIGHT: _edge(margin=0.1, drift=-0.5)}
    assert keeping.update(1.0, Decision(), det, _frame(), None).event == "", "heading home, not yet past re-arm"
    det.edges = {SIDE_RIGHT: _edge(margin=0.3, drift=-0.8)}
    done = keeping.update(1.05, Decision(), det, _frame(), None)
    assert (done.side, done.event) == (SIDE_NONE, "back"), "the swing home is no longer this pulse's to stop"
    keeping.update(2.0, START, _detector(), _frame(), None)
    det.edges = {SIDE_RIGHT: _edge(margin=-0.2, drift=0.3)}
    assert keeping.update(2.0 + tn.max_s, Decision(), det, _frame(), None).event == "timeout"


@pytest.mark.parametrize("edge, why", [
    (_edge(margin=-0.1, drift=0.3, hard="blinker"), "driver"),
    (_edge(margin=-0.1, drift=0.3, hard="deliberate"), "driver"),
    (_edge(margin=-1.7, drift=0.3, gap=-1.6), "out"),
    (None, "gate"),
])
def test_the_driver_a_gate_or_a_truck_gone_for_good_ends_a_pulse(edge, why):
    keeping = LaneKeeping()
    keeping.update(0.0, START, _detector(), _frame(), None)
    det = _detector()
    det.edges = {SIDE_RIGHT: edge} if edge is not None else {}
    assert keeping.update(0.2, Decision(), det, _frame(), None).event == why


def test_an_override_ends_a_pulse_and_starts_none():
    keeping = LaneKeeping()
    keeping.update(0.0, START, _detector(), _frame(), None)
    det = _detector()
    det.edges = {SIDE_RIGHT: _edge(margin=-0.1, drift=0.3)}
    yielded = SimpleNamespace(event="override", accel_fb_ms2=0.0)
    assert keeping.update(0.2, Decision(), det, _frame(), yielded).event == "override"
    assert keeping.update(0.25, START, _detector(), _frame(), yielded).side == SIDE_NONE


def test_a_swerve_holds_pulses_off_for_five_seconds():
    keeping = LaneKeeping()
    det = _detector()
    det.evade_until = 2.0
    assert keeping.update(1.0, START, det, _frame(), None).side == SIDE_NONE
    det.evade_until = -1e9
    assert keeping.update(5.9, START, det, _frame(), None).side == SIDE_NONE, "the detector's own hold is 2 s"
    assert keeping.update(6.1, START, det, _frame(), None).side == SIDE_RIGHT


def test_no_per_minute_limit_and_a_straight_prefab_lane_counts_like_a_road():
    """Lukas, 2026-11-02: no cap; a driver gripping the wheel is paying attention, and one who is not feels the pulls."""
    keeping = LaneKeeping()
    assert keeping.update(0.0, START, _detector(), _frame("prefab"), None).event == "start", "drive 12: LDW warned there"
    keeping.reset()
    for i in range(1, 20):
        assert keeping.update(3.0 * i, START, _detector(), _frame(), None).event == "start"
        keeping.reset()


def _held(push, against, rate=0.0, seconds=3.0):
    """A pulse off the right line against a driver whose estimated push stays fixed; ``rate`` turns the wheel home."""
    assist = LaneAssist(replace(PULSE, settle_s=0.0))
    route = make_route(None, length=2500.0)
    hand = HandPush(hand=-against, accel=0.0, push=push, against=against)
    target, rows = 0.0, []
    for i in range(int(seconds / DT)):
        out = assist.update(frame_at(i * DT, route, None, 0.5, 0.01), Gains(FORCE, GAIN),
                            Wheel(ok=True, steer=target, rate=rate), engage=SIDE_RIGHT, hand=hand)
        target = out.target
        rows.append((i * DT, out))
    return rows


@pytest.mark.parametrize("push, against, yields", [(3000.0, 2400.0, True), (3000.0, 1000.0, False),
                                                    (400.0, 400.0, False)])
def test_a_pulse_hands_back_gradually_as_the_driver_pushes_back(push, against, yields):
    """2026-11-02 and 03: pulses wound up to 3000 in 0.7 s against a held wheel and took up to 3 s to let go."""
    rows = _held(push, against)
    first = next((t for t, o in rows if o.event == "override"), None)
    assert (first is not None) == yields
    if yields:
        assert first <= 1.0
        handing = [o.authority for t, o in rows if t < first and 0.05 < o.authority < 0.95]
        assert len(handing) >= 3, "eased out, not dropped"


def test_a_pulse_pushes_no_harder_once_the_driver_holds_back_with_40_percent_of_it():
    """Its integral wound the force up against the held wheel faster than the driver's push showed."""
    held = _held(1500.0, 700.0)
    assert not any(o.event == "override" for _, o in held), "under half: no hand-back"
    assert max(o.servo.cap for t, o in held if t > 1.0) <= 1500
    assert max(o.servo.cap for t, o in _held(1500.0, 300.0) if t > 1.0) == 3000, "light hands change nothing"


@pytest.mark.parametrize("drift", [0.3, 0.8, 1.2])
def test_a_free_wheel_never_reads_as_a_driver_pushing_back(drift):
    rows = keep(drift, seconds=16.0, feel=True)
    assert not any(o.event == "override" for _, _, _, o, _ in rows)
    assert past_line(rows) < 0.3


def test_heavy_resting_hands_in_a_bend_count_as_a_grip():
    """By decision (Lukas, 2026-11-03): a driver gripping the wheel is paying attention, and one who is not feels the
    pulls. The model's sleeping driver whose resting hands hold 90 % of an R 400 is handed back once; past the line
    the detector stays disarmed, and the truck ran 18.7 m wide in 12 s, against 0.03 m when the pulse ignored the push.
    """
    rows = keep(None, radius=400.0, seconds=16.0, hands=20000.0, hands_at=0.9 * (-1.0 / 400.0 / GAIN), feel=True,
                **RESTING)
    assert any(o.event == "override" for _, _, _, o, _ in rows)
    assert past_line(rows) < 20.0, "a ratchet: lower it when the trade-off improves, never raise it"


def test_resistance_does_not_count_while_the_pulse_turns_the_wheel_home():
    """Resting hands' damping and friction resist only a moving wheel; a driver fighting holds it or turns it out."""
    rows = _held(3000.0, 2400.0, rate=0.05)
    assert not any(o.event == "override" for _, o in rows)
    assert max(o.servo.cap for t, o in rows if t > 1.0) == 3000


def _core(offset, drift, accel=0.0):
    return SimpleNamespace(offset_m=offset, drift_ms=drift, accel_fb_ms2=accel, event="")


def test_a_pulse_rides_through_a_fork_ahead_on_the_lane_the_core_still_sees():
    keeping = LaneKeeping()
    keeping.update(0.0, START, _detector(), _frame(), None)
    fork = Decision(reason="branch")
    frame = SimpleNamespace(**vars(_frame()), half_width_m=HALF_WIDTH, blinker_left=False, blinker_right=False,
                            hazards=False)
    assert keeping.update(0.5, fork, _detector(), frame, _core(0.95, 0.4, -1.5)).side == SIDE_RIGHT, "drive 12: cut"
    assert keeping.update(1.0, fork, _detector(), frame, _core(0.6, -0.05, 0.1)).event == "back"
    keeping.update(2.0, START, _detector(), frame, None)
    lamp = SimpleNamespace(**{**vars(frame), "blinker_right": True})
    assert keeping.update(2.5, fork, _detector(), lamp, _core(0.95, 0.4)).event == "driver"
    keeping.update(4.0, START, _detector(), frame, None)
    assert keeping.update(4.5, Decision(reason="no_map"), _detector(), frame, _core(0.95, 0.4)).event == "gate"


@pytest.mark.parametrize("evade_until, steered, reason", [
    (7.0, None, "swerve"),
    (-1e9, SimpleNamespace(event="override", accel_fb_ms2=0.0), "override"),
])
def test_a_departure_no_pulse_takes_says_why(evade_until, steered, reason):
    det = _detector()
    det.evade_until = evade_until
    pulse = LaneKeeping().update(6.0, START, det, _frame(), steered)
    assert (pulse.side, pulse.event, pulse.line, pulse.reason) == (SIDE_NONE, "skipped", SIDE_RIGHT, reason)


def test_the_pulse_core_rises_fast_on_a_pulse_only_and_has_no_ring_weave_or_hands_check():
    assist = LaneAssist(replace(PULSE, settle_s=0.0))
    route = make_route(None, length=2500.0)
    gains = Gains(FORCE, GAIN)
    seen = []
    for i in range(int(6.0 / DT)):
        t = i * DT
        engage = SIDE_RIGHT if 1.0 <= t < 5.0 else SIDE_NONE
        swing = 0.05 * math.sin(2.0 * math.pi * 2.0 * t) if t >= 2.0 else 0.0
        f = frame_at(t, route, None, 0.9, 0.02 * math.sin(t * 3.0), steer_right=-swing)
        seen.append((t, assist.update(f, gains, Wheel(ok=True, steer=swing, rate=0.6), engage=engage)))
    assert all(o.authority == 0.0 for t, o in seen if t < 1.0), "standby until a pulse"
    assert next(t for t, o in seen if o.authority >= 1.0) <= 1.0 + PULSE.pulse_rise_s + DT
    assert assist.ring_scale == 1.0 and assist.gain == 1.0, "a ringing wheel and a flipping correction cut nothing"
    assert not any(o.rumble or o.event in ("ring", "weave", "hands_off") for _, o in seen)
    assert all(o.authority == 0.0 for t, o in seen if t > 5.0 + PULSE.pulse_fall_s + DT)


@pytest.mark.parametrize("hand", [0.0, 8000.0, 30000.0])
def test_a_pulse_never_pushes_the_wheel_back_toward_its_line(hand):
    def driver(t, rig):
        if 4.0 <= t < 5.5:
            rig.hands, rig.hands_at = hand, 0.04
    pushes = []
    rows = keep(0.6, driver=driver, pushes=pushes)
    ccw = [push * only for _, push, only in pushes if only]
    assert ccw and min(ccw) >= 0.0, "drive 12: a driver steering home was pushed back toward the line"
    assert max(ccw) > 300.0, "while it still pushes home"
    assert not any(p.event == "override" for *_, p in rows), "a wheel turned further home is no driver to yield to"


def test_a_pulse_asks_the_plugin_for_the_one_way_away_from_its_line_and_la_for_both():
    for side, way in ((SIDE_RIGHT, 1), (SIDE_LEFT, -1)):
        assist = LaneAssist(replace(PULSE, settle_s=0.0))
        f = frame_at(1.0, make_route(None), None, 0.9 * way, 0.0)
        assist.update(frame_at(0.95, make_route(None), None, 0.9 * way, 0.0), Gains(FORCE, GAIN), Wheel(ok=True))
        assert assist.update(f, Gains(FORCE, GAIN), Wheel(ok=True), engage=side).servo.only == way
    la = LaneAssist(replace(AssistTuning(), settle_s=0.0, rise_s=0.05, rearm_s=0.0))
    outs = [la.update(frame_at(i * DT, make_route(None), None, 0.9, 0.0), Gains(FORCE, GAIN), Wheel(ok=True))
            for i in range(40)]
    assert outs[-1].servo.kp > 0.0 and outs[-1].servo.only == 0


def test_just_after_a_pulse_the_other_line_is_caught_through_the_mutes_its_wheel_raised():
    keeping = LaneKeeping()
    keeping.update(0.0, START, _detector(), _frame(), None)
    det = _detector()
    det.edges = {SIDE_RIGHT: _edge(margin=0.3, drift=-0.8)}
    assert keeping.update(1.0, Decision(), det, _frame(), None).event == "back"
    det.edges = {SIDE_LEFT: _edge(margin=-0.05, drift=0.7, hard="intent"), SIDE_RIGHT: _edge(margin=2.0)}
    caught = keeping.update(2.0, Decision(), det, _frame(), None)
    assert (caught.side, caught.event) == (SIDE_LEFT, "start"), "the pulse turned the wheel: no intent of the driver"
    assert keeping.update(2.5, Decision(), det, _frame(), None).event == "", "and it is not ended by that mute"
    det.edges[SIDE_LEFT] = _edge(margin=-0.05, drift=0.7, hard="blinker")
    assert keeping.update(2.6, Decision(), det, _frame(), None).event == "driver", "a blinker still ends it"
    late = LaneKeeping()
    late.update(0.0, START, _detector(), _frame(), None)
    late.update(1.0, Decision(), _detector(**{}), _frame(), None)
    det.edges = {SIDE_LEFT: _edge(margin=-0.05, drift=0.7, hard="intent")}
    assert late.update(4.0, Decision(), det, _frame(), None).side == SIDE_NONE, "only just after a pulse"


def test_a_pulse_counts_a_wheel_held_short_of_its_target_as_the_driver_not_one_held_past_it():
    def held(at, push=600.0):
        assist = LaneAssist(replace(PULSE, settle_s=0.0))
        route = make_route(None, length=2500.0)
        outs = []
        for i in range(int(2.0 / DT)):
            f = frame_at(i * DT, route, None, 0.9, 0.01)
            outs.append(assist.update(f, Gains(FORCE, GAIN), Wheel(ok=True, steer=at, push=push), engage=SIDE_RIGHT))
        return outs
    assert any(o.event == "override" for o in held(-0.05)), "held toward the line, against the pulse"
    assert not any(o.event == "override" for o in held(0.2)), "held far home, past what it asks"
    assert not any(o.event == "override" for o in held(-0.05, push=240.0)), (
        "2026-11-12: four of ten pulses gave up 0.4 s in on a wheel their 20 to 240 units had not moved yet")


def test_a_pulse_steers_toward_a_line_half_a_metre_inside_its_edge():
    assist = LaneAssist(replace(PULSE, settle_s=0.0))
    route = make_route(None, length=2500.0)
    inside = PULSE.half_lane_m - HALF_WIDTH - PULSE.pulse_inset_m
    out = None
    for i in range(int(3.0 / DT)):
        f = frame_at(i * DT, route, None, inside, 0.0)
        out = assist.update(f, Gains(FORCE, GAIN), Wheel(ok=True), engage=SIDE_RIGHT)
    assert abs(out.accel_fb_ms2) < 0.05, "on that line, nothing to correct"


def lka_wheel(world, one_sided=True):
    """A hooked plugin that reads the wheel and, from 2026-11-02, servos one way when asked."""
    wheel(world)
    if one_sided:
        bits = struct.unpack_from("=I", world.buf, 20)[0]
        struct.pack_into("=I", world.buf, 20, bits | cmd.STATUS_ONE_SIDED)
    return world


@pytest.fixture
def lka_only(monkeypatch):
    settings = Settings.instance()
    monkeypatch.setattr(settings, "ldw_enabled", False, raising=False)
    monkeypatch.setattr(settings, "la_enabled", False, raising=False)
    monkeypatch.setattr(settings, "lka_enabled", True, raising=False)
    monkeypatch.setattr(settings, "la_force_per_ms2", FORCE, raising=False)
    monkeypatch.setattr(settings, "la_kappa_per_steer", GAIN, raising=False)
    return settings


def _drive(world, thread, rate=0.6, steps=200):
    commands, servos, reasons = [], [], []
    for i in range(steps):
        t = i * DT
        struct.pack_into("=I", world.buf, 56, i + 1)
        lateral = 0.0 if t < DRIFT_START else rate * (t - DRIFT_START)
        world.step(thread, i, lateral, rate / SPEED if t >= DRIFT_START else 0.0)
        commands.append(world.command())
        servos.append(servo_in(world) + struct.unpack_from("=b", world.buf, 47))
        reasons.append(thread.data.assist_reason)
    return commands, servos, reasons


def test_with_the_rumble_off_it_steers_back_and_never_rumbles(monkeypatch, lka_only, caplog):
    caplog.set_level("INFO")
    world = lka_wheel(World(monkeypatch))
    commands, servos, reasons = _drive(world, world.thread())
    assert any(s[1] > 0.0 and s[0] > 0.0 for s in servos), "a counterclockwise target away from the right line"
    assert all(s[5] == 1 for s in servos if s[1] > 0.0), "and the plugin told to push only that way"
    assert all(c[0] == cmd.CMD_NONE for c in commands), "no LDW, no rumble"
    assert "pulse" in reasons
    assert "LKA steers back from the right line" in caplog.text


def test_each_pulse_end_says_how_hard_the_driver_pushed_against_it(monkeypatch, lka_only, caplog):
    caplog.set_level("INFO")
    world = lka_wheel(World(monkeypatch))
    thread = world.thread()
    _drive(world, thread)
    ends = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LKA pulse ends")]
    assert ends and all("the driver against it up to" in m for m in ends)
    assert math.isfinite(thread._lka.push.hand)


def test_lane_departure_warns_exactly_the_same_with_it_on(monkeypatch, lka_only):
    monkeypatch.setattr(lka_only, "ldw_enabled", True, raising=False)
    world = lka_wheel(World(monkeypatch))
    with_lka = _drive(world, world.thread())[0]
    monkeypatch.setattr(lka_only, "lka_enabled", False, raising=False)
    world = lka_wheel(World(monkeypatch))
    without = _drive(world, world.thread())[0]
    assert with_lka == without and (cmd.CMD_RUMBLE, 2, cmd.PEAK_UNITS) in without


def test_a_pulse_with_no_warning_behind_it_still_comes_with_the_soft_rumble(monkeypatch, lka_only):
    """The catch after a pulse starts on its own tests; with the rumble on it gets the soft cue for a second."""
    from core.lane_departure.lka import Pulse

    monkeypatch.setattr(lka_only, "ldw_enabled", True, raising=False)
    world = lka_wheel(World(monkeypatch))
    thread = world.thread()
    clock = {"s": 0.0}
    thread._keeping.update = lambda *a, **k: Pulse(SIDE_LEFT, "", "lane", clock["s"], SIDE_LEFT)
    seen = []
    for i in range(40):
        clock["s"] = i * DT
        world.step(thread, i)
        seen.append(world.command())
    soft = (cmd.CMD_RUMBLE, SIDE_LEFT, cmd.intensity_units(Tuning().approach_scale))
    assert seen[0] == soft and seen.count(soft) == round(Tuning().min_rumble_s / DT)
    assert seen[-1][0] == cmd.CMD_NONE
    monkeypatch.setattr(lka_only, "ldw_enabled", False, raising=False)
    thread = world.thread()
    thread._keeping.update = lambda *a, **k: Pulse(SIDE_LEFT, "", "lane", 0.0, SIDE_LEFT)
    world.step(thread, 41)
    assert world.command()[0] == cmd.CMD_NONE, "rumble off: a pulse never rumbles"


def test_an_older_plugin_that_servos_both_ways_keeps_it_off_and_says_so(monkeypatch, lka_only, caplog):
    caplog.set_level("INFO")
    world = lka_wheel(World(monkeypatch), one_sided=False)
    commands, servos, reasons = _drive(world, world.thread())
    assert all(s[1] == 0.0 for s in servos) and "pulse" not in reasons
    assert "LKA needs the FFB plugin with the one-sided servo" in caplog.text


def test_la_outranks_it(monkeypatch, lka_only, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(lka_only, "la_enabled", True, raising=False)
    world = lka_wheel(World(monkeypatch))
    reasons = _drive(world, world.thread())[2]
    assert "pulse" not in reasons and "standby" not in reasons
    assert "LKA on (one-way pulses), waiting while LA steers" in caplog.text

