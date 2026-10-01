"""TP: fast-closing in-lane lead (clip fbc397b3); CoDirectional pass-through dip check."""
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 100.0 / 3.6
_LEAD_SPEED = 20.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "WARN",
    "t_warn_max": 2.0,
    "t_brake_max": None,
    "must_be_suppressed_by": None,
}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED - _LEAD_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 70.0 - closing * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1,
            x=0.5,
            z=distance,
            yaw_deg=180.0,
            speed=_LEAD_SPEED,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
