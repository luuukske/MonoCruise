"""TP: faster vehicle alongside ego cuts in then decelerates.

Ego at 80 km/h. Vehicle starts at (x=3.5, z=4) — adjacent lane, slightly ahead
(alongside, not far ahead). Vehicle initially at 90 km/h, then decelerates at
4 m/s² after merge begins (frame 10) and drifts left at 1.5 m/s. Once the
vehicle drops below ego speed while in ego's lane, ego must brake.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED_INIT = 90.0 / 3.6
_TARGET_SPEED_FLOOR = 50.0 / 3.6
_TARGET_DECEL = 4.0
_LATERAL_RATE = 1.5
_START_X = 3.5
_START_Z = 4.0
_MERGE_START_FRAME = 10
_N_FRAMES = 90

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    target_x = _START_X
    target_z = _START_Z
    target_speed = _TARGET_SPEED_INIT
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        accel = -_TARGET_DECEL if i >= _MERGE_START_FRAME and target_speed > _TARGET_SPEED_FLOOR else 0.0
        target = make_vehicle(
            vid=1, x=target_x, z=target_z,
            yaw_deg=180.0, speed=target_speed, curvature=0.0,
            acceleration=accel,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        if i >= _MERGE_START_FRAME:
            target_x = max(target_x - _LATERAL_RATE * _DT, 0.0)
            target_speed = max(target_speed - _TARGET_DECEL * _DT, _TARGET_SPEED_FLOOR)
        target_z += (target_speed - _EGO_SPEED) * _DT
        if target_z < 1.0:
            break
    return frames
