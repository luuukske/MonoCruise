"""FP: slow oncoming TMP vehicle at 10 km/h, ego at 70 km/h, near intersection.

Rel speed = (70+10)/3.6 = 22 m/s = 80 km/h (above TMP rel-speed threshold),
but geometry places it in opposite lane (x=3.5). OppositeLaneFilter suppresses.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 70.0 / 3.6
_TARGET_SPEED = 10.0 / 3.6
_N_FRAMES = 60

EXPECTED = {"max_state": "STANDBY"}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED + _TARGET_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 50.0 - closing * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1,
            x=3.5,  # opposite lane
            z=distance,
            yaw_deg=0.0,  # south-facing (head-on)
            speed=_TARGET_SPEED,
            curvature=0.0,
            is_tmp=True,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
