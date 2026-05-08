"""FP: parked vehicle visible past upcoming bend (corner-entry stationary)."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
import math

_EGO_SPEED = 60.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "CornerEntryStationaryFilter",
}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED, steer=0.0)
        # Parked vehicle ahead and laterally offset (around the bend)
        # yaw_deg ~30 degrees: road curves to the left 30 deg over 40m → kappa~0.013
        # Anti-parallel to ego = 0 deg. Tilted 30 deg from anti-parallel means yaw=30.
        target = make_vehicle(
            vid=1,
            x=4.0,
            z=40.0,
            yaw_deg=30.0,  # anti-parallel to ego (0=South) tilted 30° → implies road bend
            speed=0.0,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
