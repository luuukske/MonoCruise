"""The wheel learner on a model wheel: a manual drive, then LA holding targets, then the hold test on what it learned.

Simulated only. The manual drive puts a driver's hands on the rim (a stiff pull toward a wandering wanted angle) with
the servo off; LA's part holds a few still targets with the servo on, the target hold parking it, the learner fed the
plugin's trace samples throughout. Nothing is pushed that LA would not push.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

from core.la.actuator import WheelTuning
from core.la.hold import HoldOut, HoldTuning, TargetHold
from core.la.wheel_learner import FLAG_SERVO, WheelLearner, WheelModel, learned_overrides
from core.lane_departure.command import SERVO_OFF
from tools.la_bench.plant import SimWheel, WheelParams
from tools.la_bench.population import Hold, command, hold_test
from tools.la_bench.protocol import Design

MANUAL_S = 60.0
EPISODES = 30
EPISODE_S = 4.0
SEND_HZ = 20.0
DEFAULT = Design()


def design_for(model: WheelModel, base: Design = DEFAULT, settle: bool = False) -> Design:
    """``base`` with what the learner learned applied, as LA's actuator applies it; ``settle`` for a plugin with the
    settle law (status bit 4096)."""
    tune = WheelTuning(inertia=base.inertia, bandwidth_hz=base.bandwidth_hz, damping=base.damping,
                       damping_first=base.damping_first)
    over = {"settle": settle, **learned_overrides(tune, model, pushes_set=base.pushes_ahead)}
    return replace(base, **over)


def _clock(learner: WheelLearner) -> float:
    """Where a new drive's clock starts for ``learner``: after its last sample, as the plugin's clock never goes back
    (the learner skips samples no newer than the last)."""
    last = learner._last_t
    return 0.0 if last == -math.inf else last + 1.0


def _sample(wheel: SimWheel, learner: WheelLearner, servo_on: bool, t0: float) -> None:
    p = wheel.params
    game = -p.spring * wheel.steer * p.sign
    learner.feed(t0 + wheel.time, wheel.reading, game, game + wheel.applied, wheel.plugin.value, wheel.readings,
                 FLAG_SERVO if servo_on else 0)


def manual_drive(p: WheelParams, learner: WheelLearner, seconds: float = MANUAL_S, seed: int = 0) -> None:
    """The driver steers with the servo off: hands pull the rim toward a wanted angle made of slow sines."""
    rng = random.Random(seed)
    waves = [(rng.uniform(0.05, 0.4), rng.uniform(0.005, 0.04), rng.uniform(0.0, 2.0 * math.pi)) for _ in range(3)]
    k_hand, c_hand = rng.uniform(10000.0, 40000.0), rng.uniform(100.0, 400.0)
    wheel = SimWheel(p)
    learner.direction = 0
    t0 = _clock(learner)
    while wheel.time < seconds:
        want = sum(a * math.sin(2.0 * math.pi * f * wheel.time + ph) for f, a, ph in waves)
        wheel.external = k_hand * (want - wheel.rim) - c_hand * wheel.rim_rate
        tick = wheel.n % wheel.substeps == 0
        wheel.substep(SERVO_OFF)
        if tick:
            _sample(wheel, learner, False, t0)


def la_drive(p: WheelParams, learner: WheelLearner, base: Design = DEFAULT, episodes: int = EPISODES,
             seed: int = 0, hold: HoldTuning | None = None, settle: bool = False) -> int:
    """LA holds ``episodes`` still targets one after the other, the design rebuilt from the learner every send.
    Returns how many hunts the hold reported."""
    rng = random.Random(seed + 1)
    wheel = SimWheel(p)
    parker = TargetHold(hold or HoldTuning())
    learner.direction = p.sign
    t0 = _clock(learner)
    hunts, target, sent_at = 0, 0.0, -1.0
    goals = [rng.choice((-1.0, 1.0)) * rng.uniform(0.005, 0.03) for _ in range(episodes)]
    servo = SERVO_OFF
    while wheel.time < episodes * EPISODE_S:
        target = goals[int(wheel.time / EPISODE_S)]
        if wheel.time - sent_at >= 1.0 / SEND_HZ - 1e-9:
            m = learner.model
            out = parker.update(wheel.time - sent_at if sent_at >= 0.0 else 0.0, target, 0.0, wheel.reading,
                                m.quantum, wheel.plugin.value * p.sign) if sent_at >= 0.0 else HoldOut()
            if out.hunted:
                learner.note_hunt()
                hunts += 1
            servo = command(design_for(learner.model, base, settle), target, p.sign, out)
            sent_at = wheel.time
        tick = wheel.n % wheel.substeps == 0
        wheel.substep(servo)
        if tick:
            _sample(wheel, learner, True, t0)
    return hunts


def learn_and_hold(p: WheelParams, seed: int = 0, base: Design = DEFAULT, episodes: int = EPISODES,
                   model: WheelModel | None = None, settle: bool = False) -> tuple[WheelModel, Hold, int]:
    """The whole loop on one wheel: what was learned, the hold test on it, and the hunts on the way."""
    learner = WheelLearner(model)
    manual_drive(p, learner, seed=seed)
    hunts = la_drive(p, learner, base, episodes, seed, settle=settle)
    m = learner.model
    return m, hold_test(p, design_for(m, base, settle), hold=HoldTuning(), quantum=m.quantum), hunts
