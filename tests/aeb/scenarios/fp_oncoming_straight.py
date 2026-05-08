"""FP: oncoming vehicle in opposite lane, straight road.

Vehicle at x=2.5m — in OPPOSITE_OR_OUTER lane (d_abs=2.5 > lane_half_width=1.95).
Arc corridors overlap geometrically (2.5 < ego_hw+target_hw+margin=2.9m), so AEB
would trigger without OppositeLaneFilter. With it, the oncoming evasion logic
(Fix B) lets the vehicle follow the road and clear ego's arc.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 80.0 / 3.6
_TARGET_SPEED = 70.0 / 3.6
_N_FRAMES = 90

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
        distance = 70.0 - closing * t
        if distance < 3.0:
            break
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        # x=2.5m → d_abs=2.5 > 1.95 → OPPOSITE_OR_OUTER lane.
        # Arc corridor overlap since 2.5 < 1.15+1.25+0.5=2.9 — geometrically collides.
        target = make_vehicle(
            vid=1,
            x=2.5,
            z=distance,
            yaw_deg=0.0,
            speed=_TARGET_SPEED,
            curvature=0.0,
            noise_seed=i,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
    return frames
