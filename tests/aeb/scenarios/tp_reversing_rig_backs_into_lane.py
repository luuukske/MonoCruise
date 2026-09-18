"""TP: TMP rig heading ego's way, backing diagonally into ego's lane (clip 66874532).

Heading reads co-directional, travel reads head-on. A reversing target is never
oncoming, so the oncoming evasion and lateral-gap rules must not claim it.
"""
import math

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 45.0 / 3.6
_REVERSE_MS = 4.5
_YAW_DEG = 212.0              # heading (+0.53, +0.85): ego's way; travel is -x/-z
_X0, _Z0 = 7.0, 40.0
_N_FRAMES = 75

# Travel frame warns at 0.8 s; the oncoming class for reversing waited until 2.27 s.
EXPECTED = {"max_state": "BRAKE", "t_warn_max": 1.5}


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
            yaw_deg=_YAW_DEG, speed=-_REVERSE_MS, length=6.0,
            is_tmp=True,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
