"""Invariant: global_speed_limit_kmh caps the CC set-speed, re-checked every tick.

In cruise mode CruiseController.step() re-clamps. In limiter mode step() never
runs, so the orchestrator re-applies the target itself; without that, tightening
the limit mid-session leaves a stale higher button target capping the truck
above the new limit. Both paths are covered here.
"""
from __future__ import annotations

import pytest

import core.longitudinal.cc as cc_mod
from core.longitudinal.cc import CruiseController
from tests.longitudinal.harness import make_ctx


class _FakeSettings:
    global_speed_limit_kmh = None
    cc_kp = 0.3
    cc_ki = 0.02
    cc_kd = 0.0
    cc_integral_clamp = 2.0
    cc_accel_min_ms2 = -2.0
    cc_accel_max_ms2 = 1.5


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    _FakeSettings.global_speed_limit_kmh = None
    monkeypatch.setattr(cc_mod, "Settings", _FakeSettings)
    return _FakeSettings


@pytest.fixture
def cc():
    ctrl = CruiseController()
    ctrl.enable()
    return ctrl


def test_set_target_is_clamped_to_the_global_limit(cc, fake_settings):
    fake_settings.global_speed_limit_kmh = 80.0
    cc.set_target_kmh(120.0)
    assert cc.target_speed_kmh == 80.0


def test_button_increments_cannot_walk_past_the_global_limit(cc, fake_settings):
    fake_settings.global_speed_limit_kmh = 80.0
    cc.set_target_kmh(78.0)
    for _ in range(10):
        cc.change_target_kmh(5)
    assert cc.target_speed_kmh == 80.0


def test_step_reclamps_a_stale_target_when_the_limit_tightens(cc):
    """Cruise mode: CruiseController.step() owns the re-clamp."""
    cc.set_target_kmh(120.0)
    assert cc.target_speed_kmh == 120.0

    _FakeSettings.global_speed_limit_kmh = 70.0
    cc.step(make_ctx(90.0 / 3.6))
    assert cc.target_speed_kmh == 70.0


def test_reapplying_the_target_reclamps_it(cc):
    """Limiter mode: step() never runs, so the orchestrator re-applies the target.

    Mirrors CruiseControlThread's limiter branch, which calls
    set_target_kmh(target_speed_kmh) every tick for exactly this reason.
    """
    cc.set_target_kmh(120.0)
    _FakeSettings.global_speed_limit_kmh = 70.0

    cc.set_target_kmh(cc.target_speed_kmh)
    assert cc.target_speed_kmh == 70.0


def test_clearing_the_global_limit_does_not_restore_the_old_target(cc):
    """Clamping is destructive by design: the user re-sets the speed themselves."""
    _FakeSettings.global_speed_limit_kmh = 70.0
    cc.set_target_kmh(120.0)
    _FakeSettings.global_speed_limit_kmh = None
    cc.step(make_ctx(60.0 / 3.6))
    assert cc.target_speed_kmh == 70.0
