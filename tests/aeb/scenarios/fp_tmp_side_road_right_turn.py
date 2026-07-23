"""FP: TMP side road east-bound, right turn into opposite lane. OppositeLaneFilter suppresses."""
import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 50.0 / 3.6
_TMP_SPEED = 30.0 / 3.6

EXPECTED = {"max_state": "STANDBY"}


def build() -> list[Frame]:
    frames = []
    # Phase 1: warm-up, TMP east-bound (yaw 270), west of ego on side road.
    for i in range(25):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        x_pos = -20.0 + _TMP_SPEED * t  # moves east
        z_pos = 15.0  # north of ego, in side road
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=z_pos,
            yaw_deg=270.0,  # east-facing
            speed=_TMP_SPEED,
            curvature=0.0,
            is_tmp=True,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))

    # Phase 2: right turn (yaw 270 toward 0), drift into opposite lane x≈3.5.
    for i in range(30):
        t = (25 + i) * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        frac = i / 30.0
        yaw_deg = 270.0 - frac * 90.0  # sweeps from east (270) toward north... wait
        yaw_deg = 270.0 - frac * 90.0  # 270→180: would be north. Let's reconsider.
        yaw_deg = (270.0 + frac * 90.0) % 360.0  # 270→360=0: east to south
        x_pos = -5.0 + frac * 8.5   # moves from x=-5 to x=3.5
        z_pos = 15.0 - frac * 12.0  # moves from z=15 toward ego (z decreases toward 0)
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=z_pos,
            yaw_deg=yaw_deg,
            speed=_TMP_SPEED,
            curvature=-0.04,  # right turn (negative = right in convention)
            is_tmp=True,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))

    return frames
