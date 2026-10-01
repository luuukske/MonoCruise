"""FP: oncoming truck + 1 trailer in opposite lane, straight road. Ego at 80 km/h, oncoming
combination at 80 km/h. Trailer length 13 m. Tractor and trailer both remain in opposite lane.
OppositeLaneFilter suppresses."""
import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
from core.radar.traffic import Trailer, Position, Quaternion, Size

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 80.0 / 3.6
_TRAILER_LENGTH = 13.0

EXPECTED = {"max_state": "STANDBY"}


def _make_trailer_quat(yaw_deg: float) -> Quaternion:
    """Identity quaternion: _smooth_yaw handles arc direction."""
    return Quaternion(1.0, 0.0, 0.0, 0.0)


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED + _TARGET_SPEED
    for i in range(60):
        t = i * _DT
        distance = 80.0 - closing * t
        if distance < 5.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Tractor in opposite lane (x=3.5), south-facing (yaw=0)
        tractor = make_vehicle(
            vid=1,
            x=3.5,
            z=distance,
            yaw_deg=0.0,
            speed=_TARGET_SPEED,
            curvature=0.0,
            width=2.5,
            length=6.0,
            noise_seed=i,
        )
        # Trailer hooked behind tractor: south-facing, tractor length behind pivot
        trailer_z = distance + 6.0 * 0.82 + _TRAILER_LENGTH * 0.5
        tr = Trailer(
            position=Position(3.5, 0.0, trailer_z),
            rotation=_make_trailer_quat(0.0),
            size=Size(2.5, 1.5, _TRAILER_LENGTH),
        )
        tr.rotation._euler_cache = (0.0, 0.0, 0.0)
        tractor.trailers = [tr]
        tractor.trailer_count = 1
        frames.append(Frame(ego=ego, vehicles=[tractor], t=t))
    return frames

