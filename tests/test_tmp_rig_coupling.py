"""A TMP trailer record and its locked tractor as one rig in radar: the shared crash latch.

See core/radar/README.md §7, "Rig coupling". The rig drives toward -z at yaw 0, the trailer
centre 9.5 m behind the tractor, and goes through the real reader at 20 Hz.
"""
from __future__ import annotations

from core.radar.reader import TrafficReader
from core.radar.tmp_sdk import FLAG_LINEAR, FLAG_PLACEMENT, KIND_TRAILER, SdkBody, SdkFrame
from core.radar.traffic import Position, Quaternion, Size, Vehicle

DT = 0.05
TRACTOR, TRAILER = 10, 11
HITCH_M = 9.5


def _body(vid: int, x: float, z: float, is_trailer: bool) -> Vehicle:
    size = Size(2.5, 4.0, 13.6 if is_trailer else 6.0)
    return Vehicle(Position(x, 0.0, z), Quaternion(1.0, 0.0, 0.0, 0.0), size, 0.0, 0.0, 0, [],
                   vid, True, is_trailer)


class Rig:
    """Tractor and trailer record moving together; ``lateral`` puts the trailer out of lock."""

    def __init__(self, speed: float, lateral: float = 0.0, with_tractor: bool = True):
        self.reader = TrafficReader()
        self.t = 0.0
        self.z = 0.0
        self.speed = speed
        self.lateral = lateral
        self.with_tractor = with_tractor
        self.tractor: Vehicle | None = None
        self.trailer: Vehicle | None = None

    def step(self, decel: float = 0.0, sdk_trailer_speed: float | None = None) -> None:
        self.t += DT
        self.speed = max(0.0, self.speed - decel * DT)
        self.z -= self.speed * DT
        bodies = [_body(TRAILER, self.lateral, self.z + HITCH_M, True)]
        if self.with_tractor:
            bodies.append(_body(TRACTOR, 0.0, self.z, False))
        sdk = None
        if sdk_trailer_speed is not None:
            body = SdkBody(self.lateral, 0.0, self.z + HITCH_M, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, -sdk_trailer_speed, 0.0, 0.0, 0.0, -1.0, 0, 0.0,
                           KIND_TRAILER, FLAG_PLACEMENT | FLAG_LINEAR, 0)
            sdk = SdkFrame(self.t, True, (body,))
        vehicles, _ = self.reader._smooth_and_build(bodies, self.t, 0.0, 0.0, 500.0, 0.0, sdk)
        by_id = {v.id: v for v in vehicles}
        self.tractor, self.trailer = by_id.get(TRACTOR), by_id[TRAILER]

    def run(self, seconds: float, decel: float = 0.0) -> None:
        for _ in range(round(seconds / DT)):
            self.step(decel)

    def latch(self, vid: int, seconds: float = 2.0) -> None:
        """Put a crash latch on one body's last frame, as the §7 detector would."""
        prev = self.reader._last_vehicles[vid]
        prev._crash_hold_until = self.t + seconds
        prev._crash_knock_until = self.t + seconds


def test_an_unmatched_trailer_keeps_its_own_speed_on_a_braking_rig():
    # Coupling it to the tractor's short window was measured and dropped (README §7, rig coupling).
    rig = Rig(20.0)
    rig.run(1.5)
    rig.run(1.0, decel=6.0)
    alone = Rig(20.0, with_tractor=False)
    alone.run(1.5)
    alone.run(1.0, decel=6.0)
    assert rig.tractor._raw_brake_active
    assert rig.trailer.speed == alone.trailer.speed


def test_a_tractor_crash_is_the_trailers():
    rig = Rig(20.0)
    rig.run(1.5)
    rig.latch(TRACTOR)
    rig.step()
    assert rig.tractor.crash_confirmed and rig.trailer.crash_confirmed
    assert rig.trailer._crash_knock_until == rig.tractor._crash_knock_until


def test_a_trailer_crash_is_the_tractors():
    rig = Rig(20.0)
    rig.run(1.5)
    rig.latch(TRAILER)
    rig.step()
    assert rig.trailer.crash_confirmed and rig.tractor.crash_confirmed


def test_an_unlocked_neighbour_shares_nothing():
    rig = Rig(20.0, lateral=5.0)
    rig.run(1.5)
    rig.latch(TRACTOR)
    rig.step()
    assert rig.tractor.crash_confirmed and not rig.trailer.crash_confirmed


def test_the_shared_latch_expires_with_the_source():
    rig = Rig(20.0)
    rig.run(1.5)
    rig.latch(TRACTOR, seconds=0.5)
    rig.run(1.0)
    assert not rig.tractor.crash_confirmed and not rig.trailer.crash_confirmed
