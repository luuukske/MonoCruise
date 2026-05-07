"""TP: lead vehicle hard-brakes from 90 km/h to 30 km/h at 8 m/s².

Co-directional lead in ego lane, initial gap 25 m. Lead z is updated using
the *relative* speed (lead - ego), so the gap actually closes when ego is
faster than lead. Standard rear-end TP.
"""

from tests.aeb.harness import Frame, EgoState, make_vehicle, _DT

_EGO_SPEED = 90.0 / 3.6
_LEAD_SPEED_INIT = 90.0 / 3.6
_LEAD_SPEED_FINAL = 30.0 / 3.6
_DECEL = 8.0
_INITIAL_GAP = 25.0
_BRAKE_START = 20
_N_FRAMES = 90

EXPECTED = {"max_state": "BRAKE"}


def build() -> list[Frame]:
    frames = []
    lead_speed = _LEAD_SPEED_INIT
    lead_z = _INITIAL_GAP
    for i in range(_N_FRAMES):
        t = i * _DT
        ego = EgoState(x=0.0, y=0.0, z=0.0, yaw_norm=0.5, speed=_EGO_SPEED)
        accel = -_DECEL if i >= _BRAKE_START and lead_speed > _LEAD_SPEED_FINAL else 0.0
        target = make_vehicle(
            vid=1, x=0.0, z=lead_z,
            yaw_deg=180.0, speed=lead_speed, curvature=0.0,
            acceleration=accel,
        )
        frames.append(Frame(ego=ego, vehicles=[target], t=t))
        if i >= _BRAKE_START:
            lead_speed = max(lead_speed - _DECEL * _DT, _LEAD_SPEED_FINAL)
        lead_z += (lead_speed - _EGO_SPEED) * _DT
        if lead_z < 1.0:
            break
    return frames
