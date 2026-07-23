"""TP: close-range cut-in with sustained lateral drift (evasion over-suppress guard)."""
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 70.0 / 3.6
_TARGET_SPEED = 50.0 / 3.6
_INITIAL_GAP = 10.0
_LATERAL_RATE = 1.5
_START_X = 3.5
_N_FRAMES = 90

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    target_x = _START_X
    target_z = _INITIAL_GAP
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1, x=target_x, z=target_z,
            yaw_deg=180.0, speed=_TARGET_SPEED, curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        target_x = max(target_x - _LATERAL_RATE * _DT, -1.5)
        target_z += (_TARGET_SPEED - _EGO_SPEED) * _DT
        if target_z < 1.0:
            break
    return frames
