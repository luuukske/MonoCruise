"""Position-driven coverage for the hard-brake raw-speed transient."""

from __future__ import annotations

from core.radar.traffic import Position, Quaternion, Size, Vehicle

DT = 0.0501


def _vehicle(
    z: float,
    speed: float,
    *,
    is_tmp: bool = True,
    is_trailer: bool = False,
) -> Vehicle:
    return Vehicle(
        Position(0.0, 0.0, z),
        Quaternion(1.0, 0.0, 0.0, 0.0),
        Size(2.5, 3.0, 6.0),
        speed,
        0.0,
        0,
        [],
        1,
        is_tmp,
        is_trailer,
    )


def _step(
    prev: Vehicle,
    t_now: float,
    z: float,
    speed: float,
    *,
    is_tmp: bool = True,
    ego_speed: float = 0.0,
) -> Vehicle:
    cur = _vehicle(z, speed, is_tmp=is_tmp)
    cur.update_from_last(prev, t_now, 0.0, 0.0, 100.0, ego_speed)
    return cur


def _seed_cruise(speed: float, duration_s: float = 1.5) -> tuple[Vehicle, float, float]:
    t_now = 0.0
    z = 0.0
    prev = _vehicle(z, speed)
    prev.time = t_now
    for _ in range(int(duration_s / DT)):
        t_now += DT
        z -= speed * DT
        prev = _step(prev, t_now, z, speed)
    return prev, t_now, z


def test_low_speed_hard_brake_reaches_aeb_speed_bound():
    prev, t_now, z = _seed_cruise(4.0)
    brake_onset = t_now
    true_speed = 4.0
    active_at = None
    accel_at_04 = None

    while true_speed > 0.0:
        next_speed = max(0.0, true_speed - 4.0 * DT)
        z -= 0.5 * (true_speed + next_speed) * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed)
        if prev._raw_brake_active and active_at is None:
            active_at = t_now
        if accel_at_04 is None and t_now - brake_onset >= 0.4:
            accel_at_04 = prev.acceleration
        true_speed = next_speed

    stop_time = t_now
    stop_samples: list[tuple[float, float, float]] = []
    for _ in range(int(0.5 / DT) + 1):
        t_now += DT
        prev = _step(prev, t_now, z, 0.0)
        stop_samples.append((t_now, prev._raw_speed or 0.0, prev.speed))

    assert active_at is not None
    assert active_at - brake_onset <= 0.3
    assert accel_at_04 is not None and accel_at_04 <= -2.0
    deadline = [sample for sample in stop_samples if sample[0] - stop_time <= 0.4 + DT]
    assert abs(deadline[-1][1]) < 0.4
    assert abs(deadline[-1][2]) < 0.5
    assert min(speed for _, _, speed in stop_samples) > -0.5


def test_hard_brake_has_no_rebound_and_tracks_launch():
    prev, t_now, z = _seed_cruise(4.0)
    true_speed = 4.0
    while true_speed > 0.0:
        next_speed = max(0.0, true_speed - 4.0 * DT)
        z -= 0.5 * (true_speed + next_speed) * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed)
        true_speed = next_speed

    stopped: list[Vehicle] = []
    for _ in range(int(1.3 / DT)):
        t_now += DT
        prev = _step(prev, t_now, z, 0.0)
        stopped.append(prev)
    first_settled = next(i for i, v in enumerate(stopped) if abs(v.speed) < 0.5)
    assert all(abs(v.speed) < 0.5 for v in stopped[first_settled:])

    launch_speed = 0.0
    launch_samples: list[Vehicle] = []
    for _ in range(int(0.8 / DT)):
        next_speed = launch_speed + 2.0 * DT
        z -= 0.5 * (launch_speed + next_speed) * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed)
        launch_samples.append(prev)
        launch_speed = next_speed
    assert launch_samples[-1].speed > 1.0


def test_tmp_cruise_wobble_never_enters_transient():
    t_now = 0.0
    z = 0.0
    prev = _vehicle(z, 4.0)
    prev.time = t_now
    active_seen = False
    speeds: list[float] = []
    period = 3.0
    for _ in range(int(12.0 / DT)):
        t_now += DT
        phase = (t_now % period) / period
        true_speed = 4.0 + 0.5 * (1.0 - 2.0 * phase)
        z -= true_speed * DT
        prev = _step(prev, t_now, z, true_speed)
        active_seen = active_seen or prev._raw_brake_active
        speeds.append(prev.speed)

    assert active_seen is False
    tail = speeds[-int(3.0 / DT):]
    assert max(tail) - min(tail) < 0.8


def test_low_speed_packet_stall_without_brake_ramp_does_not_enter():
    prev, t_now, z = _seed_cruise(4.0)
    for _ in range(int(0.4 / DT)):
        t_now += DT
        prev = _step(prev, t_now, z, 4.0)
        assert prev._raw_brake_active is False


def test_high_speed_lag_freeze_resets_transient():
    prev, t_now, z = _seed_cruise(8.0)
    prev._raw_brake_active = True
    prev._raw_brake_confirm_frames = 2
    t_now += DT
    cur = _step(prev, t_now, z, 8.0, ego_speed=8.0)

    assert cur._lag_since is not None
    assert cur._raw_brake_active is False
    assert cur._raw_brake_confirm_frames == 0


def test_subframe_copies_transient_and_reanchor_resets_it():
    prev, t_now, z = _seed_cruise(4.0)
    prev._raw_brake_active = True
    prev._raw_brake_confirm_frames = 2

    subframe = _step(prev, t_now + 0.01, z - 0.04, 4.0)
    assert subframe._raw_brake_active is True
    assert subframe._raw_brake_confirm_frames == 2

    reanchored = _vehicle(z - 0.04, 4.0)
    reanchored.update_from_last(
        subframe, subframe.time - 0.1, 0.0, 0.0, 100.0, 0.0,
    )
    assert reanchored._raw_brake_active is False
    assert reanchored._raw_brake_confirm_frames == 0


def test_confirmed_crash_feeds_acc_the_same_estimate_as_aeb(monkeypatch):
    """A confirmed crash must not leave ACC on the long window. The split routes ACC to the long
    position window, but a crashed vehicle has to reach both consumers through the same
    unfiltered estimate, so the crash bypass pins the ACC input back to the AEB one."""
    from core.radar import traffic as traffic_mod

    prev, t_now, z = _seed_cruise(12.0)

    def _force_crash(self, prev_v, t):
        self.crash_confirmed = True

    monkeypatch.setattr(
        traffic_mod.Vehicle, "_tmp_apply_crash_rotation_jerk", _force_crash,
    )

    true_speed = 12.0
    for _ in range(int(0.8 / DT)):
        next_speed = max(0.0, true_speed - 8.0 * DT)
        z -= 0.5 * (true_speed + next_speed) * DT
        t_now += DT
        prev = _step(prev, t_now, z, next_speed)
        true_speed = next_speed

    assert prev.crash_confirmed is True
    # Both consumers see the crash decelerating, neither is held back.
    assert prev.acceleration < -2.0
    assert prev.acc_accel < -2.0


def test_trailer_does_not_select_its_local_transient():
    prev, t_now, z = _seed_cruise(4.0)
    prev.is_trailer = True
    prev._raw_brake_active = True
    next_speed = 4.0 - 4.0 * DT
    z -= 0.5 * (4.0 + next_speed) * DT
    t_now += DT
    cur = _vehicle(z, next_speed, is_trailer=True)
    cur.update_from_last(prev, t_now, 0.0, 0.0, 100.0, 0.0)

    assert cur._raw_brake_active is False
    assert cur._raw_brake_confirm_frames == 0
