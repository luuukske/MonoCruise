"""Pedal setup for one stick axis that rests in the middle. See the pedal README."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

try:
    import pygame  # noqa: F401
except ImportError:
    # CI has no pygame (requirements-test.txt); the config tick never opens a joystick.
    _stub = ModuleType("pygame")
    _stub.joystick = SimpleNamespace(get_count=lambda: 0, Joystick=None)
    sys.modules["pygame"] = _stub

from core.main_pedal_thread.pedal_axes import PedalAxes
from core.main_pedal_thread.thread import MainPedalThread
from core.settings import Settings
from tests.test_pedal_axes import _press

_KEYS = (
    "device", "gasaxis", "brakeaxis", "gas_inverted", "brake_inverted", "shared_pedal_axis",
)


@pytest.fixture
def restored_pedal_settings():
    before = {key: getattr(Settings, key) for key in _KEYS}
    yield
    Settings.save(values=before)


def _axis_kwargs() -> dict:
    return dict(
        gas_axis=Settings.gasaxis,
        brake_axis=Settings.brakeaxis,
        gas_inverted=Settings.gas_inverted,
        brake_inverted=Settings.brake_inverted,
        shared=Settings.shared_pedal_axis,
    )


def test_a_sweep_of_a_middle_axis_saves_both_pedals(restored_pedal_settings):
    thread = MainPedalThread()
    thread.data.pedal_config_stage = "brake"
    _press(thread._pconf_taps, "joy", 0, 0.0, 0.4)
    _press(thread._pconf_taps, "joy", 1, 0.0, 0.7)
    _press(thread._pconf_taps, "joy", 1, 0.7, -0.7)

    thread._tick_pedal_config()
    assert thread.data.pedal_config_stage == "gas"
    assert thread._pconf_brake_axis == 1
    assert thread._pconf_brake_negative is False

    thread._tick_pedal_config()
    assert Settings.shared_pedal_axis is True
    assert Settings.gasaxis == 1 and Settings.brakeaxis == 1
    assert Settings.brake_inverted is False
    assert Settings.gas_inverted is True
    assert thread.data.pedal_config_result["shared_pedal_axis"] is True

    axes = PedalAxes()
    axes.on_motion(1, 0.0, **_axis_kwargs())
    assert (axes.gas, axes.brake) == (0.0, 0.0)
    axes.on_motion(1, 0.4, **_axis_kwargs())
    assert (axes.gas, axes.brake) == (0.0, 0.4)
    axes.on_motion(1, -1.0, **_axis_kwargs())
    assert (axes.gas, axes.brake) == (1.0, 0.0)


def test_the_gas_tap_can_come_after_the_brake(restored_pedal_settings):
    thread = MainPedalThread()
    thread.data.pedal_config_stage = "brake"
    _press(thread._pconf_taps, "joy", 1, 0.0, -0.5)
    thread._tick_pedal_config()
    assert thread.data.pedal_config_stage == "gas"
    thread._tick_pedal_config()
    assert thread.data.pedal_config_stage == "gas"

    _press(thread._pconf_taps, "joy", 0, 0.0, 0.9)
    _press(thread._pconf_taps, "joy", 1, 0.0, 0.45)
    thread._tick_pedal_config()
    assert Settings.shared_pedal_axis is True
    assert Settings.brake_inverted is True
    assert Settings.gas_inverted is False
    assert Settings.gasaxis == Settings.brakeaxis == 1


def test_separate_pedals_still_save_two_axes(restored_pedal_settings):
    Settings.shared_pedal_axis = True
    thread = MainPedalThread()
    thread.data.pedal_config_stage = "brake"
    _press(thread._pconf_taps, "wheel", 0, 0.0, 1.0)
    _press(thread._pconf_taps, "wheel", 0, 1.0, -1.0)
    _press(thread._pconf_taps, "pedals", 2, -1.0, -0.7)
    thread._tick_pedal_config()
    assert thread._pconf_brake_axis == 2
    assert thread._pconf_brake_negative is None

    _press(thread._pconf_taps, "pedals", 2, -0.7, 1.0)
    _press(thread._pconf_taps, "wheel", 0, 0.0, 0.8)
    _press(thread._pconf_taps, "pedals", 1, -1.0, 0.4)
    thread._tick_pedal_config()
    assert Settings.shared_pedal_axis is False
    assert Settings.brakeaxis == 2 and Settings.gasaxis == 1
    assert Settings.brake_inverted is False
    assert Settings.gas_inverted is False

    axes = PedalAxes()
    axes.on_motion(1, 0.0, **_axis_kwargs())
    axes.on_motion(2, -1.0, **_axis_kwargs())
    assert (axes.gas, axes.brake) == (0.5, 0.0)
