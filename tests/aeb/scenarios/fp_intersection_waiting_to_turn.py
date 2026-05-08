"""FP: stationary vehicle in opposite lane, yaw rotated 30° from anti-parallel.

Ego at 60 km/h approaching. Vehicle oriented at 30° from south-facing,
indicating left-turn intent. CornerEntryStationaryFilter + EgoEvasionFilter
suppresses.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 60.0 / 3.6

EXPECTED = {"max_state": "STANDBY"}


def build() -> list[Frame]:
    frames = []
    for i in range(60):
        t = i * _DT
        distance = 50.0 - _EGO_SPEED * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Stationary in opposite lane, yaw=30° (30° offset from anti-parallel south=0)
        target = make_vehicle(
            vid=1,
            x=3.5,  # opposite lane
            z=distance,
            yaw_deg=30.0,  # 30° from south = left-turn orientation
            speed=0.0,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
