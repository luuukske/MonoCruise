"""TP: fast perpendicular crosser on a body-contact course inside the legacy
dead band.

Ego 80 km/h straight; a 90 km/h vehicle crosses perpendicularly on a true
collision course whose reference-point miss (4.0 m) lands in the band the old
ghost-arc comb left uncovered (3.4-5.0 m for this speed pair). With fixed
speed-scaled ghost spacing the three strips sat far enough apart in the
relative frame that this offset produced ZERO hits across the entire approach
(TTC 2.5 s to 0.15 s). The ArcPath capsule body covers the crosser's length,
so the impending side contact is detected every frame.

Steer formula for scenarios: steer = kappa * 180 / (12 * pi); ego is straight
so steer = 0.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6       # 22.22 m/s
_CROSS_SPEED = 90.0 / 3.6     # 25.0 m/s
_TTC0 = 2.5                   # seconds to contact at the first frame
_DMISS = 4.0                  # reference-miss inside the former dead band
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
        # there, offset by _DMISS in z (the reference miss). yaw=90 -> fwd=(-1,0)
        # so the crosser sweeps from +x toward ego's lane.
        cross_x = _CROSS_SPEED * ttc
        cross_z = _EGO_SPEED * ttc + _DMISS
        target = make_vehicle(
            vid=1,
            x=cross_x,
            z=cross_z,
            yaw_deg=90.0,
            speed=_CROSS_SPEED,
            curvature=0.0,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
