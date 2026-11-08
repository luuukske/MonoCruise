"""The plugin's settle law (servo flag 4, status bit 4096): the C constants and its Python mirror agree, it is off
unless asked, and on model wheels it parks a sticky wheel nearer its target. Simulated wheels only."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from core.la.actuator import Gains, Wheel, WheelActuator, WheelTuning
from core.la.hold import HoldTuning
from core.la.wheel_learner import RUNG_MAX, WheelModel, learned_overrides
from core.lane_departure import command as cmd
from core.lane_departure.command import ServoCommand
from tests.la import servo_sim
from tests.la.servo_sim import Plugin
from tools.la_bench.design import push_ahead_of
from tools.la_bench.plant import SIM_WHEELS
from tools.la_bench.population import Design, hold_test

ROOT = Path(__file__).resolve().parents[2]


def _define(text: str, name: str) -> str:
    return re.search(rf"#define\s+{name}\s+(\S+)", text).group(1)


def test_the_mirror_keeps_the_plugins_settle_constants_and_bits():
    signal = (ROOT / "plugins" / "ffb" / "ffb_signal.h").read_text(encoding="utf-8")
    for name in ("SETTLE_BAND", "SETTLE_RISE", "SETTLE_RAMP_S", "SETTLE_DROP_S", "SETTLE_STEP_MAX", "SETTLE_RELEASE"):
        assert float(_define(signal, f"FFB_{name}")) == getattr(servo_sim, name), name
    header = (ROOT / "plugins" / "ffb" / "ffb_cmd.h").read_text(encoding="utf-8")
    assert int(_define(header, "FFB_ST_SETTLE")) == cmd.STATUS_SETTLE
    assert int(_define(header, "FFB_SF_SETTLE")) == cmd.SERVO_SETTLE


def _stand(servo: ServoCommand, seconds: float, steer: float = 0.0) -> Plugin:
    plugin = Plugin()
    for i in range(round(seconds * 120)):
        plugin.step(servo, 0.0, steer, 1 / 120, fresh=i % 2 == 0)
    return plugin


def test_a_standing_wheel_in_the_band_is_parked_with_its_push_held():
    servo = ServoCommand(target=0.0003, kp=6000.0, ki=4000.0, cap=3000, direction=1, ff_friction=250.0,
                         ff_breakaway=500.0, settle=True)
    plugin = _stand(servo, 1.0)
    assert plugin.held
    before = (plugin.value, plugin.integ)
    for _ in range(120):
        plugin.step(servo, 0.0, 0.0, 1 / 120, fresh=False)
    assert (plugin.value, plugin.integ) == before, "the output and the integral stand"
    assert not _stand(replace(servo, settle=False), 1.0).held


def test_without_the_flag_the_mirror_is_the_layout_5_law():
    servo = ServoCommand(target=0.001, kp=6000.0, ki=0.0, cap=3000, direction=1, ff_friction=250.0, ff_breakaway=500.0,
                         ff_plan=True)
    assert _stand(servo, 2.0).ff == 0.0, "within 1.5 mlock the layout 5 law pushes nothing"
    assert _stand(replace(servo, settle=True), 2.0).ff > 450.0, "the settle law climbs to the breakaway there"


def test_the_actuator_asks_for_it_only_when_tuned_to():
    gains, wheel = Gains(500.0, 0.19), Wheel(ok=True, steer=0.0)
    off = WheelActuator(WheelTuning()).commands(gains, 0.02, 0.0, 1.0, wheel, 25.0)
    on = WheelActuator(replace(WheelTuning(), settle=True)).commands(gains, 0.02, 0.0, 1.0, wheel, 25.0)
    assert not off.settle and on.settle and replace(on, settle=False) == off
    assert cmd.servo_flags(on, cmd.servo_fields(on)) & cmd.SERVO_SETTLE
    assert not cmd.servo_flags(off, cmd.servo_fields(off)) & cmd.SERVO_SETTLE


def _gentle(name: str, settle: bool) -> Design:
    p = SIM_WHEELS[name]
    return Design(bandwidth_hz=0.5, damping=1.4, inertia=150.0, damping_first=True, settle=settle,
                  **push_ahead_of(p, breakaway_share=1.2))


def test_on_the_old_wheels_model_it_parks_within_the_limit_without_hunting():
    p = SIM_WHEELS["old"]
    settled = hold_test(p, _gentle("old", True), hold=HoldTuning())
    assert settled.flips <= 1 and settled.jitter <= 0.05 and settled.jump <= 0.5 and settled.final_mlock <= 1.5


def test_where_it_hunts_the_learner_turns_it_off():
    p = SIM_WHEELS["sticky"]
    hunts = hold_test(p, _gentle("sticky", True), hold=HoldTuning())
    assert hunts.flips > 1, "the light, very sticky rim: each start throws it 8 to 15 mlock (the known limit)"
    assert learned_overrides(WheelTuning(), WheelModel(rung=RUNG_MAX)).get("settle") is False
    assert "settle" not in learned_overrides(WheelTuning(), WheelModel(rung=RUNG_MAX - 1))
