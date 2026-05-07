"""FP: parked vehicle near ego's predicted arc on a tight right turn.

Ego turns right at 50 km/h on R=50 m (kappa=+0.02). Parked car at (5, 25)
is just outside the arc circle (~1.5 m), so cross-product lateral_offset
would read 5 m (clearly opposite-lane), but lane_frame's arc projection
correctly reads ~1.5 m (in ego's curving corridor). The hit is real;
EgoEvasionFilter must suppress because ego could steer further left to
clear (left-evasion arc bends back toward straight).
"""

import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 50.0 / 3.6
_EGO_KAPPA = 0.02  # R=50 m right turn under harness orientation (+kappa curves toward +x)
_N_FRAMES = 40

EXPECTED = {
    "max_state": "STANDBY",
    "must_be_suppressed_by": "EgoEvasionFilter",
}


def build() -> list[Frame]:
    steer = _EGO_KAPPA * 180.0 / (12.0 * math.pi)
    frames = []
    target_z = 25.0
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED, steer=steer)
        target = make_vehicle(
            vid=1, x=5.0, z=target_z,
            yaw_deg=180.0, speed=0.0, curvature=0.0,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        target_z -= _EGO_SPEED * _DT
        if target_z < 2.0:
            break
    return frames
