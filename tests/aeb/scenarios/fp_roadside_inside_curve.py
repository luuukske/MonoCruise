"""FP: parked car near ego arc on tight right turn; EgoEvasionFilter suppresses."""
import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 50.0 / 3.6
_EGO_KAPPA = 0.02  # R=50 m right turn under harness orientation (+kappa curves toward +x)
_N_FRAMES = 40

EXPECTED = {
    "max_state": "STANDBY",
    "must_be_suppressed_by": "OutOfLaneParallelFilter",
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
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        target_z -= _EGO_SPEED * _DT
        if target_z < 2.0:
            break
    return frames
