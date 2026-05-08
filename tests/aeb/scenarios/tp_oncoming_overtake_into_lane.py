"""TP: oncoming vehicle crosses centerline into ego lane.

Starts in opposite lane (x=3.5), yaw=0 (south-facing). Over ~0.4 s (~12 frames)
it yaws inward and lateral position moves to x=0 — directly in ego's lane.
Sustained crossing, not a single-frame jitter. EXPECTED: BRAKE.
"""

import math
from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 80.0 / 3.6
_CROSS_FRAMES = 12  # ~0.4 s to cross
_N_FRAMES = 50

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    closing = _EGO_SPEED + _TARGET_SPEED
    for i in range(_N_FRAMES):
        t = i * _DT
        distance = 60.0 - closing * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # After 10 warm-up frames, cross into ego lane over 12 frames
        if i < 10:
            x_pos = 3.5
            yaw_deg = 0.0
        else:
            frac = min((i - 10) / float(_CROSS_FRAMES), 1.0)
            x_pos = 3.5 - frac * 3.5  # 3.5 → 0.0
            yaw_deg = -frac * 15.0    # slight inward yaw
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=distance,
            yaw_deg=yaw_deg,
            speed=_TARGET_SPEED,
            curvature=0.0 if i < 10 else 0.01,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
