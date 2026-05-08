"""FP: TMP vehicle from side road (east-bound) turns right into south-bound lane.

Ego heads north at 50 km/h. TMP vehicle approaches from west, heading east,
then curves right (south) into opposite lane. Position history warm-up: 10
east-bound frames, then curvature transition.  OppositeLaneFilter suppresses.
"""

import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 50.0 / 3.6
_TMP_SPEED = 30.0 / 3.6

EXPECTED = {"max_state": "STANDBY"}


def build() -> list[Frame]:
    frames = []
    # Phase 1: ~10 warm-up frames, TMP heading east (yaw_deg=270 in ETS2 → west-to-east)
    # yaw_deg=270 → fwd_x=-sin(270°)=1, fwd_z=-cos(270°)=0 → moving in +X direction (east)
    # TMP starts west of ego (negative X), heading east, approaching intersection
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

    # Phase 2: curvature transition — TMP turns right (south, into opposite lane)
    # Right turn for east-facing vehicle means increasing curvature toward south
    # curvature > 0 = left (CCW), so right turn = negative curvature
    for i in range(30):
        t = (25 + i) * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        frac = i / 30.0
        yaw_deg = 270.0 - frac * 90.0  # sweeps from east (270) toward north... wait
        # East = 270°, South = 0°, turning right from east means going south
        # yaw_deg in ETS2: 0=South, 90=East-ish... Let's use direct:
        # start=270 (east), after right turn 90° → 270-90=180 (north-facing but going south)
        # Actually turning right from east means heading south = yaw=0 in ETS2
        yaw_deg = 270.0 - frac * 90.0  # 270→180: would be north. Let's reconsider.
        # ETS2 yaw: 0=South,90=West,180=North,270=East (CCW)
        # Right turn from East (270) → South (0 or 360): yaw increases (CCW=left)
        # So right turn from east to south is CW = decreasing yaw: 270→180→90→0
        # But that goes through north. Simpler: right turn from east = yaw 270→360(=0)
        yaw_deg = (270.0 + frac * 90.0) % 360.0  # 270→360=0: east to south
        # Position: turning right from east position; land in x≈3.5 (opposite lane)
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
