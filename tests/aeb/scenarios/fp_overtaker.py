"""FP: vehicle behind ego, faster, in adjacent lane (overtaking)."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 70.0 / 3.6
_TARGET_SPEED = 100.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "BrakingWorsens",
}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Faster vehicle behind ego (negative Z relative to ego = behind when facing +Z)
        # Adjacent lane (x=3.5)
        z_behind = -(20.0 - (_TARGET_SPEED - _EGO_SPEED) * t)
        if z_behind > -2.0:
            break
        target = make_vehicle(
            vid=1,
            x=3.5,
            z=z_behind,
            yaw_deg=180.0,  # North-facing (same as ego)
            speed=_TARGET_SPEED,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
