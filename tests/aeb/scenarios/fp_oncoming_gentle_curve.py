"""FP: oncoming vehicle in opposite lane, R = 200 m gentle curve."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
import math

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 60.0 / 3.6
_N_FRAMES = 90
# R=200m → kappa=0.005; left turn for ego (kappa>0)
_EGO_KAPPA = 0.005
_TGT_KAPPA = -0.005  # same road, opposite direction → opposite kappa sign

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "OppositeLaneFilter",
}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED + _TARGET_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 100.0 - closing * t
        if distance < 3.0:
            break
        # kappa = radians(steer * speed * 12) / speed = radians(steer * 12)
        # steer = degrees(kappa) / 12 = kappa * 180 / (12 * pi)
        steer = _EGO_KAPPA * 180.0 / (12.0 * math.pi)
        ego = EgoState(
            x=0.0, y=0.0, z=0.0, yaw_norm=0.5,
            speed=_EGO_SPEED, steer=steer,
        )
        target = make_vehicle(
            vid=1,
            x=3.5,
            z=distance,
            yaw_deg=0.0,
            speed=_TARGET_SPEED,
            curvature=_TGT_KAPPA,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
