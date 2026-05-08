"""FP: MP queue of stationary vehicles at corner entry, ego straight."""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 54.0 / 3.6  # m/s
_N_FRAMES = 60

EXPECTED = {
    "max_state": "STANDBY",
    "t_warn_max": None,
    "t_brake_max": None,
    "must_be_suppressed_by": "CornerEntryStationaryFilter",
}


def build() -> list[Frame]:
    # Left-curving road continuation: R≈60 m, curve starts at z=10.
    # Vehicle 1 sits in Lane.EGO (the critical MP-queue case);
    # vehicles 2-3 are progressively deeper into the curve.
    frames = []
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5,
                       speed=_EGO_SPEED, steer=0.0)
        # s_curve=10 -> theta~9.5deg, lat~0.83 m -> Lane.EGO
        v1 = make_vehicle(vid=1, x=-0.83, z=19.95, yaw_deg=170.5,
                          speed=0.0, is_tmp=True, noise_seed=i)
        # s_curve=20 -> theta~19deg, lat~3.31 m -> Lane.OPPOSITE_OR_OUTER
        v2 = make_vehicle(vid=2, x=-3.31, z=29.63, yaw_deg=160.9,
                          speed=0.0, is_tmp=True, noise_seed=i + 100)
        # s_curve=30 -> theta~28.6deg, lat~7.35 m
        v3 = make_vehicle(vid=3, x=-7.35, z=38.77, yaw_deg=151.4,
                          speed=0.0, is_tmp=True, noise_seed=i + 200)
        frames.append(Frame(ego=ego, vehicles=[v1, v2, v3], t=t))
    return frames
