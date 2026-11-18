"""Brake-end release on ACC's lane: once the short fit shows a lead stopped braking, the long accel
fit lets go toward it. Release only, never above zero, never on AEB's lane. core/radar/README.md §7."""
from __future__ import annotations

import pytest

from core.radar import traffic
from core.radar.traffic import (
    _accel_from_speed_history, _brake_end_release, _smooth_vehicle_kinematics,
)

DT = 0.05


def _history(brake_s: float, after_s: float, after_accel: float = 0.0, v0: float = 22.0):
    """Speed samples: steady, a 2.4 m/s^2 brake for brake_s, then after_accel for after_s."""
    out, t, v = [], 0.0, v0
    for _ in range(int(1.0 / DT)):
        out.append((t, v))
        t += DT
    for _ in range(int(brake_s / DT)):
        v -= 2.4 * DT
        out.append((t, v))
        t += DT
    for _ in range(int(after_s / DT)):
        v += after_accel * DT
        out.append((t, v))
        t += DT
    return out


def _long(history):
    return _accel_from_speed_history(history, 1.5)


def test_a_lead_that_stopped_braking_lets_the_long_fit_go():
    h = _history(1.0, 0.5)
    assert _long(h) < -1.0, "the fixture must still read a brake on the long fit"
    assert _brake_end_release(h, _long(h)) == pytest.approx(0.0, abs=0.05)


def test_a_lead_still_braking_keeps_the_long_fit():
    h = _history(1.5, 0.0)
    assert _brake_end_release(h, _long(h)) == _long(h)


def test_the_release_never_adds_braking_or_pull():
    for after in (-0.1, 0.0, 0.8, 2.0):
        h = _history(1.0, 0.5, after_accel=after)
        out = _brake_end_release(h, _long(h))
        assert _long(h) <= out <= 0.0
    assert _brake_end_release(_history(0.0, 1.0, 0.8), 0.5) == 0.5


def test_a_netcode_snap_is_not_a_release():
    h = _history(1.0, 0.0)
    t = h[-1][0]
    h += [(t + DT * (i + 1), h[-1][1] + 0.4 * (i + 1)) for i in range(8)]  # 8 m/s^2 jump
    assert _brake_end_release(h, _long(h)) == _long(h)


def test_share_zero_disables(monkeypatch):
    monkeypatch.setattr(traffic, "_ACC_RELEASE_SHARE", 0.0)
    h = _history(1.0, 0.5)
    assert _brake_end_release(h, _long(h)) == _long(h)


def test_aeb_lane_is_untouched():
    """Same raw speeds with and without the release: AEB's outputs are bit-identical."""
    speeds = [v for _, v in _history(1.0, 1.0)]
    states = {}
    for release in (False, True):
        state = [None] * 7 + [False, 0.0]
        outs = []
        for i, v in enumerate(speeds):
            out = _smooth_vehicle_kinematics(v, v, 100.0 + i * DT, DT, *state, acc_sdk=release)
            state = [out[0], out[1], out[3], out[4], out[5], out[6], out[7], out[8], out[9]]
            outs.append(out)
        states[release] = outs
    for off, on in zip(states[False], states[True]):
        assert off[:4] == on[:4]
    assert any(on[5] > off[5] + 0.3 for off, on in zip(states[False], states[True])), \
        "the ACC lane must let go sooner"
