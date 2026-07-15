"""FP: parked car on right shoulder, ego in lane (evasion filter)."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "OutOfLaneParallelFilter",
}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Parked car on right shoulder: x=2.2m (right), at z=40m ahead.
        # x=2.2 < corridor width (2.9m) so arc corridors overlap.
        # Ego can steer left to avoid (evasion left arc clears), so
        # EgoEvasionFilter suppresses it.
        distance = 40.0 - _EGO_SPEED * t * 0.3
        if distance < 5.0:
            break
        target = make_vehicle(
            vid=1,
            x=2.2,
            z=distance,
            yaw_deg=180.0,
            speed=0.0,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
