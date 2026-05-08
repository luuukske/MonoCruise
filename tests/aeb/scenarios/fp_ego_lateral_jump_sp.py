"""FP: ego pose jumps 1.5 m laterally for one frame, snaps back.

Ego at 80 km/h, AI vehicle at 60 km/h in adjacent lane (x=3.5, z=12). Ego
normally at x=0; at frame 10 ego jumps to x=1.5, bringing the relative
geometry into ego's corridor. Pipeline must suppress via EgoEvasionFilter.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 60.0 / 3.6
_INITIAL_GAP = 12.0
_JUMP_FRAME = 10
_N_FRAMES = 40

EXPECTED = {
    "max_state": "STANDBY",
    "must_be_suppressed_by": "EgoEvasionFilter",
}


def build() -> list[Frame]:
    frames = []
    target_z = _INITIAL_GAP
    for i in range(_N_FRAMES):
        t = i * _DT
        ego_x = 1.5 if i == _JUMP_FRAME else 0.0
        ego = EgoState(x=ego_x, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1, x=3.5, z=target_z,
            yaw_deg=180.0, speed=_TARGET_SPEED, curvature=0.0,
            is_tmp=False,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        target_z += (_TARGET_SPEED - _EGO_SPEED) * _DT
        if target_z < 1.0:
            break
    return frames
