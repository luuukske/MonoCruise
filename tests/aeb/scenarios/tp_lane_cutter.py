"""TP: vehicle merging into ego lane from right at close range."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT
import math

_EGO_SPEED = 70.0 / 3.6
_N_FRAMES = 90

EXPECTED = {
    "max_state": "WARN",
    "t_warn_max": 3.0,
    "t_brake_max": None,
    "must_be_suppressed_by": None,
}


def build() -> list[Frame]:
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        # Merging vehicle starts right of ego, moving left at 45° yaw offset
        # At yaw_norm=0.5 ego faces +Z. Target yaw ~170 deg → mostly North, slightly west.
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # Target moves from right side into ego lane
        x_pos = max(3.0 - t * 1.5, 0.0)
        z_pos = 25.0 - _EGO_SPEED * t * 0.5
        if z_pos < 2.0:
            break
        target = make_vehicle(
            vid=1,
            x=x_pos,
            z=z_pos,
            yaw_deg=175.0,  # mostly co-directional, slight inward lean
            speed=_EGO_SPEED * 0.9,
            curvature=-0.01,  # turning left (into ego lane),
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
