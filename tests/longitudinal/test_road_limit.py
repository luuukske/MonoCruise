"""Road speed limit from the SDK: the global cap and the set-speed follow.

Driven through CruiseControlThread.loop() against fake siblings. The driver's own
global limit must never be rewritten, and a road with no posted limit must never
disengage CC. See core/cruise_control_thread/README.md.
"""
from __future__ import annotations

import math
import struct

import pytest

from core.cruise_control_thread.road_limit import road_limit_kmh
from core.cruise_control_thread.thread import CruiseControlThread
from core.settings import Settings
from core.speed_units import MPH_TO_KMH
from core.thread_management.registry import registry
from tests.longitudinal.harness import (
    FakeThread,
    pedal_data,
    sending_data,
    telemetry_data,
)


def _sdk_ms(kmh: float) -> float:
    """What the SDK hands over: the limit in m/s as a 32-bit float."""
    return struct.unpack("f", struct.pack("f", kmh / 3.6))[0]


@pytest.fixture
def rig(monkeypatch):
    """CruiseControlThread in cruise mode at 100 km/h, no global limit, both options off."""
    tel = FakeThread("telemetry_thread", telemetry_data(speed=100.0 / 3.6))
    pedal = FakeThread("main_pedal_thread", pedal_data())
    sending = FakeThread("sending_thread", sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)

    settings = Settings.instance()
    for key, value in (
        ("cc_mode", "Cruise control"),
        ("last_game", 1),
        ("global_speed_limit_kmh", None),
        ("acc_enabled", False),
        ("autospeedlimit_variable", False),
        ("autospeedtarget_variable", False),
    ):
        monkeypatch.setattr(settings, key, value)

    thread = CruiseControlThread()
    thread.running = True
    yield thread, tel, settings

    for name in ("telemetry_thread", "main_pedal_thread", "sending_thread"):
        registry.unregister(name)


def _drive(thread, tel, posted_kmh: float) -> None:
    tel.data.set(speedLimit=_sdk_ms(posted_kmh))
    thread.loop()


@pytest.mark.parametrize("kmh", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
def test_sdk_float_rounds_to_the_posted_kmh(kmh):
    """int() truncation read 80 as 79 and 130 as 129."""
    assert road_limit_kmh(_sdk_ms(kmh)) == float(kmh)


@pytest.mark.parametrize("raw", [0.0, -1.0, math.nan, math.inf, None, "x"])
def test_no_posted_limit_reads_as_none(raw):
    assert road_limit_kmh(raw) is None


def test_ats_limit_lands_on_the_mph_grid(monkeypatch):
    monkeypatch.setattr(Settings.instance(), "last_game", 2)
    assert road_limit_kmh(_sdk_ms(55 * MPH_TO_KMH)) == pytest.approx(55 * MPH_TO_KMH)


def test_cap_alone_activates_the_limiter_at_the_posted_limit(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedlimit_variable", True)
    _drive(thread, tel, 80)
    assert thread._limiter_ctrl.active is True
    assert thread._limiter_ctrl.target_speed_kmh == 80.0


def test_cap_takes_the_lower_of_road_and_driver_limit(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedlimit_variable", True)
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)
    _drive(thread, tel, 120)
    assert thread._limiter_ctrl.target_speed_kmh == 90.0
    _drive(thread, tel, 60)
    assert thread._limiter_ctrl.target_speed_kmh == 60.0


def test_cap_never_rewrites_the_drivers_global_limit(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedlimit_variable", True)
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)
    for posted in (50, 0, 120):
        _drive(thread, tel, posted)
    monkeypatch.setattr(settings, "autospeedlimit_variable", False)
    _drive(thread, tel, 50)
    assert settings.global_speed_limit_kmh == 90.0
    assert thread._limiter_ctrl.target_speed_kmh == 90.0


def test_cap_off_ignores_the_road(rig):
    thread, tel, _ = rig
    _drive(thread, tel, 50)
    assert thread._limiter_ctrl.active is False


def test_road_without_a_limit_releases_the_road_cap(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedlimit_variable", True)
    _drive(thread, tel, 80)
    _drive(thread, tel, 0)
    assert thread._limiter_ctrl.active is False


def test_follow_moves_the_set_speed_to_each_new_limit(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedtarget_variable", True)
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(100.0)
    _drive(thread, tel, 80)
    assert thread._cc_ctrl.target_speed_kmh == 80.0
    _drive(thread, tel, 110)
    assert thread._cc_ctrl.target_speed_kmh == 110.0
    assert thread._cc_ctrl.enabled is True


def test_road_without_a_limit_keeps_cc_engaged(rig, monkeypatch):
    """Disabling CC there would also drop ACC mid-follow, without a word."""
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedtarget_variable", True)
    thread._cc_ctrl.enable()
    _drive(thread, tel, 80)
    _drive(thread, tel, 0)
    assert thread._cc_ctrl.enabled is True
    assert thread._cc_ctrl.target_speed_kmh == 80.0
    assert thread.data.active is True


def test_flicker_through_no_limit_keeps_the_drivers_adjustment(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedtarget_variable", True)
    thread._cc_ctrl.enable()
    _drive(thread, tel, 80)
    thread._cc_ctrl.set_target_kmh(85.0)
    _drive(thread, tel, 0)
    _drive(thread, tel, 80)
    assert thread._cc_ctrl.target_speed_kmh == 85.0


def test_follow_off_leaves_the_set_speed_alone(rig):
    thread, tel, _ = rig
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(100.0)
    _drive(thread, tel, 80)
    assert thread._cc_ctrl.target_speed_kmh == 100.0


def test_follow_stays_under_the_drivers_global_limit(rig, monkeypatch):
    thread, tel, settings = rig
    monkeypatch.setattr(settings, "autospeedtarget_variable", True)
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)
    thread._cc_ctrl.enable()
    _drive(thread, tel, 120)
    assert thread._cc_ctrl.target_speed_kmh == 90.0
