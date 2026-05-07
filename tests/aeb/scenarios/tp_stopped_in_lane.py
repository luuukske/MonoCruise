"""TP: stopped vehicle in ego lane, ego at 70 km/h."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

# Ego faces North (yaw_norm=0.5). Forward = +Z. Targets ahead at positive Z.
_EGO_SPEED = 70.0 / 3.6
_N_FRAMES = 120

EXPECTED = {
    "max_state": "WARN",
    "t_warn_max": 4.5,
    "t_brake_max": None,
    "must_be_suppressed_by": None,
}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 60.0 - _EGO_SPEED * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Stopped vehicle directly ahead in lane; yaw=180 = North (same direction as ego)
        target = make_vehicle(
            vid=1,
            x=0.0,
            z=distance,
            yaw_deg=180.0,
            speed=0.0,
            curvature=0.0,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
