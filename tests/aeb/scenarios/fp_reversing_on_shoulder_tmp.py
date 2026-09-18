"""FP: TMP truck backing along the shoulder, parallel to ego's lane, never entering it.

Travel reads head-on; losing the oncoming class must not turn a clear pass into a threat.
"""
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 50.0 / 3.6
_REVERSE_MS = 3.0
_X = 3.9                      # one lane separation off ego's centreline
_Z0 = 45.0
_N_FRAMES = 90

EXPECTED = {"max_state": "STANDBY"}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego_z = _EGO_SPEED * t
        target_z = _Z0 - _REVERSE_MS * t
        if target_z - ego_z < 1.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=ego_z, yaw_norm=0.5, speed=_EGO_SPEED)
        # yaw 180 faces +Z like ego; negative speed backs it toward ego.
        target = make_vehicle(
            vid=1, x=_X, z=target_z, yaw_deg=180.0, speed=-_REVERSE_MS,
            length=12.0, is_tmp=True,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
