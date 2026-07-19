"""FP: vehicle completing left turn from west road into south-bound opposite lane.

Vehicle was heading west on a perpendicular road; now in final arc of a left
turn depositing it into the opposite lane (south-bound, x≈3.5). End of arc
transiently crosses ego's predicted forward path. OppositeLaneFilter suppresses.
"""

import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 50.0 / 3.6
_TARGET_SPEED = 20.0 / 3.6

EXPECTED = {"max_state": "STANDBY"}


def build() -> list[Frame]:
    frames = []
    for i in range(60):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Vehicle starts east of ego heading west (yaw=90), completing a left turn
        # into south-bound. Yaw progresses from ~90 (west) toward 0 (south).
        frac = min(i / 40.0, 1.0)
        yaw_deg = 90.0 * (1.0 - frac)  # 90→0 (west to south)
        x_pos = 8.0 - frac * 4.5  # comes from east toward opposite lane (x≈3.5)
        z_pos = 5.0 + _TARGET_SPEED * t * 0.3  # moves slightly forward
        # Track authored under the pre-2026-07-19 asymmetric pivot; shift the
        # pivot back along heading so the symmetric body keeps the same
        # world-space sweep (body clears ego's path, per scenario intent).
        yaw_rad = math.radians(yaw_deg)
        x_pos += 0.32 * 10.0 * math.sin(yaw_rad)
        z_pos += 0.32 * 10.0 * math.cos(yaw_rad)
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=z_pos,
            yaw_deg=yaw_deg,
            speed=_TARGET_SPEED,
            curvature=-0.06,  # left turn (positive in convention = CCW/left),
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
