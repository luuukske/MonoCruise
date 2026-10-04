"""Hazard trial confirmation, bounded duration and restoration."""
from core.sending_thread.hazard_flash import HazardFlash


def test_does_not_toggle_before_telemetry_confirmation():
    trial = HazardFlash.begin(0, True, True)
    assert trial.step(0.1, True, True, True) is None
    assert trial.step(0.2, False, False, True) is None
    assert trial.step(0.3, False, True, True) is None
    assert trial.step(0.39, False, True, True) is None
    assert trial.step(0.41, False, True, True) is True


def test_missing_confirmation_aborts_and_restores():
    trial = HazardFlash.begin(0, True, True)
    assert trial.step(1.01, True, False, True) is True
    assert not trial.active
    assert trial.step(2, True, True, True) is None


def test_switching_off_restores_original_off_state():
    trial = HazardFlash.begin(0, False, False)
    assert trial.step(0.1, True, False, False) is False
    assert not trial.active


def test_five_second_limit_restores_even_when_confirmed():
    trial = HazardFlash.begin(0, True, True)
    trial.changed = 4.8
    assert trial.step(5, False, True, True) is True
    assert not trial.active


def test_sender_trial_presses_release_and_restore(monkeypatch):
    import threading
    from types import SimpleNamespace
    from core.settings import Settings
    from core.sending_thread import thread as module

    Settings.save({"experimental_hazard_flash": True, "hazards_variable": True})
    sender = object.__new__(module.SendingThread)
    sender._lock = threading.Lock()
    sender._hazard_flash = None
    sender._hazard_flash_request = True
    sender._hazard_wanted = None
    sender._hazard_phase = "idle"
    sender._hazard_duration = module.HAZARD_PRESS_DURATION
    sender._hazard_retriggers = 0
    controller = SimpleNamespace(flasher4way=False)
    current = False
    previous = False
    rising = 0
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    for i in range(650):
        clock[0] = i / 100
        sender._tick_hazards(controller, current, connected=True)
        if controller.flasher4way and not previous:
            current = not current
            rising += 1
        previous = controller.flasher4way
    assert rising > 4
    assert current is False
    assert sender._hazard_flash is None
    assert controller.flasher4way is False
