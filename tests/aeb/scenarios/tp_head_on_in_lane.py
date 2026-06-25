"""TP: head-on vehicle drifted into ego lane (zero lateral offset)."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 60.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "WARN",
    "t_warn_max": 3.0,
    "t_brake_max": None,
    "must_be_suppressed_by": None,
}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED + _TARGET_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 80.0 - closing * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Head-on: facing South (yaw_deg=0) at 0 lateral offset: in ego lane
        target = make_vehicle(
            vid=1,
            x=0.0,
            z=distance,
            yaw_deg=0.0,  # South-facing = head-on to North-facing ego
            speed=_TARGET_SPEED,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames

