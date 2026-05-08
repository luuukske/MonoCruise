"""FP: vehicle U-turning at distant intersection (Fix D over-rotation suppression)."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
import math

_EGO_SPEED = 60.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "TurningCrossTrafficFilter",
}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Cross-traffic vehicle at intersection 50m ahead, turning with high curvature
        # Vehicle is 10m to the side, heading ~90° (East → yaw≈270 in ETS2 convention)
        # kappa = 0.1 → R=10m tight turn; Fix D should dampen arc_curvature
        target = make_vehicle(
            vid=1,
            x=10.0,
            z=50.0,
            yaw_deg=90.0,  # East-facing, turning toward South (toward ego)
            speed=10.0,
            curvature=-0.1,  # high curvature right turn,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
