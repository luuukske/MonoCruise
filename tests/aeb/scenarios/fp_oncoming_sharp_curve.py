"""FP: oncoming vehicle in opposite lane, R = 50 m sharp curve."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
import math

_EGO_SPEED = 50.0 / 3.6
_TARGET_SPEED = 40.0 / 3.6
_N_FRAMES = 90
_EGO_KAPPA = 0.02  # R=50m left turn
_TGT_KAPPA = -0.02

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
        distance = 60.0 - closing * t
        if distance < 3.0:
            break
        # kappa = radians(steer * 12); steer = kappa * 180 / (12 * pi)
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
