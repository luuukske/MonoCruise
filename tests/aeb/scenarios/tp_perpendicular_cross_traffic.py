"""TP: vehicle crossing ego path perpendicularly at 50 km/h (T-bone).

Zero curvature — straight-line cross. TurningCrossTrafficFilter only suppresses
tightly-curving targets; curvature=0 passes through to collision eval.
Ego at 70 km/h straight. EXPECTED: BRAKE.
"""

import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 70.0 / 3.6
_TARGET_SPEED = 50.0 / 3.6

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    # Vehicle starts west of ego (negative X), heading east (yaw=270)
    # fwd_x = -sin(270°) = 1.0, so it moves in +X direction across ego path
    for i in range(60):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Vehicle crosses from left (-X) to right (+X) at z=20 m ahead
        x_start = -20.0
        x_pos = x_start + _TARGET_SPEED * t
        if x_pos > 20.0:
            break
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=20.0,
            yaw_deg=270.0,  # east-facing (cross traffic)
            speed=_TARGET_SPEED,
            curvature=0.0,  # straight — TurningCrossTrafficFilter won't suppress,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
