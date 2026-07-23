"""TP: straight TMP perpendicular crosser (TmpCrossTraffic endpoint-lane regression)."""
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6       # 22.22 m/s
_CROSS_SPEED = 90.0 / 3.6     # 25.0 m/s
_TTC0 = 2.5                   # seconds to contact at the first frame
_DMISS = 0.0                  # dead-center collision course
_N_FRAMES = 70

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ttc = _TTC0 - t
        if ttc <= 0.15:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Contact point: ego reaches z = ego_speed*ttc; the crosser reaches x=0
        # there. yaw=90 -> fwd=(-1,0) so the crosser sweeps from +x toward ego.
        cross_x = _CROSS_SPEED * ttc
        cross_z = _EGO_SPEED * ttc + _DMISS
        target = make_vehicle(
            vid=1,
            x=cross_x,
            z=cross_z,
            yaw_deg=90.0,
            speed=_CROSS_SPEED,
            curvature=0.0,
            is_tmp=True,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
