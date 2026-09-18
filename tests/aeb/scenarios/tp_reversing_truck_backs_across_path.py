"""TP: TMP truck facing ego at an angle, backing across ego's lane at 70 km/h.

d80936f9's pose above the TMP split, where the rel-speed floor lets it through. Heading
reads oncoming (fwd_dot -0.67) and was dropped as such; travel reads crossing.
"""
import math

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 70.0 / 3.6
_REVERSE_MS = 2.9
_YAW_DEG = 48.0               # heading (-0.74, -0.67): faces ego, travel is +x/+z
_X0, _Z0 = -8.7, 38.4
_N_FRAMES = 75

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    yaw = math.radians(_YAW_DEG)
    travel_x, travel_z = math.sin(yaw), math.cos(yaw)
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=_EGO_SPEED * t, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1,
            x=_X0 + travel_x * _REVERSE_MS * t,
            z=_Z0 + travel_z * _REVERSE_MS * t,
            yaw_deg=_YAW_DEG, speed=-_REVERSE_MS, length=6.0, width=3.0,
            is_tmp=True,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
