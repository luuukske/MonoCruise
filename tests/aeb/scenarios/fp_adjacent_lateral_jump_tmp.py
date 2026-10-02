"""FP: TMP adjacent lane with one-frame lateral jump; EgoEvasionFilter suppresses."""
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TMP_SPEED = 60.0 / 3.6
_INITIAL_GAP = 12.0
_JUMP_FRAME = 15
_N_FRAMES = 40

EXPECTED = {
    "max_state": "STANDBY",
    "must_be_suppressed_by": "OutOfLaneParallelFilter",
}


def build() -> list[Frame]:
    frames = []
    target_z = _INITIAL_GAP
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        x_pos = 2.0 if i == _JUMP_FRAME else 3.5
        target = make_vehicle(
            vid=1, x=x_pos, z=target_z,
            yaw_deg=180.0, speed=_TMP_SPEED, curvature=0.0,
            is_tmp=True,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        target_z += (_TMP_SPEED - _EGO_SPEED) * _DT
        if target_z < 1.0:
            break
    return frames
