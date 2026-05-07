"""TP: wrong-way driver directly in ego lane (x≈0), head-on at 80 km/h each.

OppositeLaneFilter must NOT suppress because the vehicle is in ego's lane (EGO).
EXPECTED: BRAKE.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 80.0 / 3.6
_N_FRAMES = 50

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED + _TARGET_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 60.0 - closing * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1,
            x=0.0,  # directly in ego lane — zero lateral offset
            z=distance,
            yaw_deg=0.0,  # south-facing = head-on
            speed=_TARGET_SPEED,
            curvature=0.0,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
