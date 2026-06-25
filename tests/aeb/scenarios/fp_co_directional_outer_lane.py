"""FP: co-directional vehicle in outer lane of same-turn corner, ego overtaking."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
import math

_EGO_SPEED = 60.0 / 3.6   # 60 km/h
_TARGET_SPEED = 40.0 / 3.6  # 40 km/h: slower, ego overtakes
_N_FRAMES = 90
_EGO_KAPPA = 0.015  # R≈67m left turn
_TGT_KAPPA = 0.010  # same turn, slightly wider radius

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "CoDirectionalDivergeFilter",
}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED - _TARGET_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        # kappa = radians(steer * 12); steer = kappa * 180 / (12 * pi)
        steer = _EGO_KAPPA * 180.0 / (12.0 * math.pi)
        ego = EgoState(
            x=0.0, y=0.0, z=0.0, yaw_norm=0.5,
            speed=_EGO_SPEED, steer=steer,
        )
        # Slower co-directional vehicle at x=2.2 (right, outer lane).
        # Arc corridors overlap (2.2 < 2.9m). With extended same-turn lookahead,
        # CoDirectionalDivergeFilter detects that ego overtakes (diverges) within
        # the horizon and suppresses.
        distance = 5.0 + closing * t
        target = make_vehicle(
            vid=1,
            x=2.2,
            z=distance,
            yaw_deg=180.0,
            speed=_TARGET_SPEED,
            curvature=_TGT_KAPPA,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames

