"""FP: vehicle far behind ego closing at high speed in same lane.

Ego at 80 km/h, vehicle 50 m behind in ego's lane at 144 km/h (overtaking).
Closing rate 17.8 m/s. Braking ego would worsen the impact, so AEB must
not trigger — the braking_worsens vector-magnitude check owns this case.
Companion to fp_overtaker which uses an adjacent lane; this variant puts
the closer in ego's lane.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 144.0 / 3.6
_INITIAL_GAP_BEHIND = 50.0
_N_FRAMES = 90

EXPECTED = {
    "max_state": "STANDBY",
    "must_be_suppressed_by": "BrakingWorsens",
}


def build() -> list[Frame]:
    frames = []
    target_z = -_INITIAL_GAP_BEHIND
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        target = make_vehicle(
            vid=1, x=0.0, z=target_z,
            yaw_deg=180.0, speed=_TARGET_SPEED, curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        target_z += (_TARGET_SPEED - _EGO_SPEED) * _DT
        if target_z > -2.0:
            break
    return frames
