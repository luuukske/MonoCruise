"""TP: vehicle cuts across ego lane at 90 deg on a direct collision course.

Target approaches from the right at 50 km/h heading west while ego travels at
70 km/h north. x_start is chosen so both vehicles arrive at the intercept
point simultaneously (unbraked). The braked-ego arc still reaches z=25 m when
the target is within vehicle-width of x=0, so BRAKE is expected.
"""

import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 70.0 / 3.6
_TARGET_SPEED = 50.0 / 3.6
_INTERCEPT_Z = 25.0

_T_COLLISION = _INTERCEPT_Z / _EGO_SPEED
_X_START = _TARGET_SPEED * _T_COLLISION  # direct collision course

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    for i in range(60):
        t = i * _DT
        x_pos = _X_START - _TARGET_SPEED * t
        if x_pos < -20.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Target starts east of ego (positive X), heading west (yaw=90°)
        # fwd_x = -sin(90°) = -1, so it moves in -X direction across ego path
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=_INTERCEPT_Z,
            yaw_deg=90.0,
            speed=_TARGET_SPEED,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
