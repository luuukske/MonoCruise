"""TP: genuine straight perpendicular TMP crosser on a dead-center collision course.

Ego 80 km/h straight; a TMP (multiplayer) vehicle at 90 km/h crosses
perpendicularly with zero curvature, its reference point arriving at ego's
reference point exactly (dead-center). A collision hit is present every frame.

Regression for the TmpCrossTrafficFilter endpoint-lane bug: a straight crosser
always ends tens of metres past ego's lane at the 3 s horizon, so the
endpoint-lane test suppressed it at every range (no warn, no brake in TMP
sessions until the crosser's measured speed dropped). The straight-path pass
(|v_curvature| < turning_diverge_kappa) plus the imminence floor keep it alive.
Corpus FN clip ffd29f9e.

Steer formula for scenarios: steer = kappa * 180 / (12 * pi); ego is straight
so steer = 0.
"""

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
