"""Hold FSM crawl follow: speed keeping behind a moving lead is not a stop, and never a rollback.

See core/acc/ACC_ARCHITECTURE.md §10.3 and core/sending_thread/README.md."""
from __future__ import annotations

import math

import pytest

from core.cruise_control_thread.thread import CruiseControlThread
from core.longitudinal.base import LongOutput
from core.sending_thread import hold_controller as hc
from core.sending_thread.hold_controller import (
    STATE_HOLDING, STATE_ROLLING, STATE_STOPPING, HoldController,
)

DT = 0.01
BRAKE_CAP_MS2 = 10.0
BRAKE_LAG_S = 0.3
BRAKE_DEAD_S = 0.12


def _hold() -> HoldController:
    return HoldController(lambda d: min(1.0, max(0.0, d) / BRAKE_CAP_MS2))


def _pitch_norm(rad: float) -> float:
    return rad / (2.0 * math.pi)


def _tick(h: HoldController, kmh: float, cmd: float, crawl: bool, pitch_rad: float = 0.0,
          gear: int = 1):
    return h.update(speed_kmh=kmh, gear=gear, pitch_norm=_pitch_norm(pitch_rad),
                    commanded_accel_ms2=cmd, gasval=0.0, opdgasval=0.0, offset=0.0,
                    park_brake=False, aeb_active=False, dt=DT, crawl_follow=crawl)


def _settle(h: HoldController, kmh: float, cmd: float, crawl: bool, n: int = 50):
    out = None
    for _ in range(n):
        out = _tick(h, kmh, cmd, crawl)
    return out


def test_without_crawl_follow_a_zero_command_still_stops():
    assert _settle(_hold(), 1.5, 0.0, crawl=False).state == STATE_STOPPING


def test_speed_keeping_behind_a_crawl_is_not_a_stop():
    for cmd in (0.0, -0.1, -0.25):
        assert _settle(_hold(), 1.5, cmd, crawl=True).state == STATE_ROLLING, cmd


def test_a_real_brake_request_still_stops():
    assert _settle(_hold(), 1.5, -0.35, crawl=True).state == STATE_STOPPING


def test_below_the_crawl_floor_it_stops():
    assert _settle(_hold(), 0.9, -0.05, crawl=True).state == STATE_STOPPING


def test_rolling_back_while_crawl_following_stops():
    assert _settle(_hold(), -0.3, -0.05, crawl=True).state in (STATE_STOPPING, STATE_HOLDING)


def test_a_fast_slowdown_is_captured_while_the_brake_can_still_build():
    h, kmh = _hold(), 1.9
    for _ in range(300):
        out = _tick(h, kmh, -0.1, crawl=True)
        if out.state == STATE_STOPPING:
            break
        kmh -= 0.6 * DT * 3.6
    assert out.state == STATE_STOPPING
    # Dead time plus lag of a slow trailer brake, core/sending_thread/README.md.
    assert (kmh / 3.6) / 0.6 >= 0.5


def _hill(grade_rad: float, grade_err: float, cmd: float, crawl: bool) -> float:
    """Truck slowing into a crawl uphill, the mapper under-reading the grade. Lowest speed."""
    h = _hold()
    g_along = hc.GRAVITY_MS2 * math.sin(grade_rad)
    v, brake, low = 1.0, 0.0, 1.0
    queue = [0.0] * int(BRAKE_DEAD_S / DT)
    for _ in range(int(10.0 / DT)):
        out = _tick(h, v * 3.6, cmd, crawl, grade_rad)
        queue.append(BRAKE_CAP_MS2 * out.brake_pedal)
        brake += (queue.pop(0) - brake) * (1.0 - math.exp(-DT / BRAKE_LAG_S))
        drive = cmd + g_along * (1.0 - grade_err) - g_along
        if v > 1e-6:
            v = max(0.0, v + (drive - brake) * DT)
        elif drive < -brake:
            v = v + (drive + brake) * DT
        low = min(low, v)
    return low


@pytest.mark.parametrize("grade_rad", [0.03, 0.05, 0.10, 0.20, 0.25])
@pytest.mark.parametrize("grade_err", [0.0, 0.3, 0.6, 1.0])
def test_crawl_follow_never_rolls_back_more_than_the_plain_capture(grade_rad, grade_err):
    """The capture it defers must still come early enough for the hold brake to build."""
    crawl, plain = _hill(grade_rad, grade_err, -0.2, True), _hill(grade_rad, grade_err, -0.2, False)
    assert crawl >= plain - 1e-4, (crawl, plain)


@pytest.mark.parametrize("grade_rad", [0.03, 0.05, 0.10, 0.20, 0.25])
@pytest.mark.parametrize("grade_err", [0.0, 0.3])
def test_crawl_follow_never_rolls_back_with_a_sane_grade_estimate(grade_rad, grade_err):
    assert _hill(grade_rad, grade_err, -0.2, True) > -hc._HOLD_ROLLBACK_DEADBAND_MS


def _out(wanted, active=True):
    return LongOutput(wanted, active)


class _Acc:
    def __init__(self, crawl: bool) -> None:
        self.crawl_follow = crawl


def test_the_flag_is_only_sent_while_acc_sets_the_command():
    T = CruiseControlThread
    assert T._crawl_follow(_Acc(True), _out(-0.1), -0.1, True, "cc")
    assert not T._crawl_follow(_Acc(False), _out(-0.1), -0.1, True, "cc")
    assert not T._crawl_follow(_Acc(True), _out(-0.1), -0.1, False, "cc")
    assert not T._crawl_follow(_Acc(True), _out(-0.1), -0.1, True, "limiter")
    assert not T._crawl_follow(_Acc(True), _out(0.4), 0.1, True, "cc")
    assert not T._crawl_follow(_Acc(True), _out(None, False), -0.1, True, "cc")
