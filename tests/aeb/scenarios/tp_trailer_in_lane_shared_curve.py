"""TP: trailer body in ego lane on shared curve (clip 434f0401); tractor stays outer lane."""
import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
from core.radar.traffic import Trailer, Position, Quaternion, Size

_EGO_SPEED = 100.0 / 3.6
_TARGET_SPEED = 50.0 / 3.6
_EGO_KAPPA = 0.009           # shared left-ish curve, above turning_diverge_kappa
_TGT_KAPPA = 0.009           # same sign -> same-turn gate
_TRAILER_LENGTH = 13.9
_N_FRAMES = 40

# Trailer in lane frame 0; pre-fix waited for cab ref (t_warn_max catches fix).
EXPECTED = {
    "max_state": "WARN",
    "t_warn_max": 0.5,
    "must_be_suppressed_by": None,
}


def build() -> list[Frame]:
    steer = _EGO_KAPPA * 180.0 / (12.0 * math.pi)
    closing = _EGO_SPEED - _TARGET_SPEED
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        # Cab in outer lane by reference; trailer body still in ego lane.
        z_cab = 26.0 - closing * t
        if z_cab < 11.0:
            break
        x_cab = -(1.4 + 0.14 * (z_cab - 14.0))   # left of ego, outer lane

        ego = EgoState(
            x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED, steer=steer,
        )
        tractor = make_vehicle(
            vid=1,
            x=x_cab,
            z=z_cab,
            yaw_deg=166.0,          # co-directional (~ -14 deg off ego heading)
            speed=_TARGET_SPEED,
            curvature=_TGT_KAPPA,
            width=2.5,
            length=6.1,
        )
        # Trailer trails behind the cab, swung so its rear body lies in ego's lane.
        tr = Trailer(
            position=Position(x_cab + 1.5, 0.0, z_cab - 7.0),
            rotation=Quaternion(1.0, 0.0, 0.0, 0.0),
            size=Size(2.6, 1.5, _TRAILER_LENGTH),
        )
        tr.rotation._euler_cache = (0.0, 172.0, 0.0)   # ~ -8 deg off ego heading
        tractor.trailers = [tr]
        tractor.trailer_count = 1
        frames.append(Frame(ego=ego, vehicles=[tractor], t=t))
    return frames
