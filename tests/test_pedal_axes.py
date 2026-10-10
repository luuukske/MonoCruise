"""Pedal axes and the pedal-connect tap detector, driven by SDL motion events only.

SDL reports 0.0 for an axis it has not heard from since the device was opened.
Reading the gas axis on a brake event gave half throttle (issue #12), and a rest
baseline read at open made pedal taps misdetected or inverted. See
core/main_pedal_thread/README.md.
"""

from __future__ import annotations

from core.main_pedal_thread.pedal_axes import (
    PEDAL_TAP_THRESHOLD,
    TAP_THRESHOLD,
    PedalAxes,
    TapDetector,
    normalise_axis,
)

AXES = dict(gas_axis=1, brake_axis=2, gas_inverted=False, brake_inverted=False)


def test_brake_event_never_moves_the_gas():
    axes = PedalAxes()
    axes.on_motion(2, 0.6, **AXES)
    assert axes.brake == normalise_axis(0.6, False)
    assert axes.gas == 0.0


def test_each_pedal_follows_its_own_axis():
    axes = PedalAxes()
    axes.on_motion(1, -1.0, **AXES)
    axes.on_motion(2, 1.0, **AXES)
    axes.on_motion(0, 0.3, **AXES)  # steering on the same device
    assert (axes.gas, axes.brake) == (0.0, 1.0)


def test_inverted_axis_and_reset():
    axes = PedalAxes()
    axes.on_motion(1, -0.5, gas_axis=1, brake_axis=2, gas_inverted=True, brake_inverted=False)
    assert axes.gas == 0.75
    axes.reset()
    assert (axes.gas, axes.brake) == (0.0, 0.0)


def test_only_live_axes_are_re_read():
    """A missed event self-heals once the axis has reported; an unheard one stays 0."""
    axes = PedalAxes()
    sdl = {1: 0.0, 2: 0.0}  # SDL's 0.0 for axes it has not heard from
    axes.refresh(sdl.get, **AXES)
    assert (axes.gas, axes.brake) == (0.0, 0.0)

    axes.on_motion(2, 0.4, **AXES)
    sdl[2] = -1.0  # released, and that event never reached us
    axes.refresh(sdl.get, **AXES)
    assert (axes.gas, axes.brake) == (0.0, 0.0)


def test_brake_going_live_is_reported_once():
    """A brake held through a reconnect jumps from 0: the stomp test must skip it."""
    axes = PedalAxes()
    axes.on_motion(2, 0.0, **AXES)
    assert axes.take_brake_went_live() is True
    axes.on_motion(2, 0.2, **AXES)
    assert axes.take_brake_went_live() is False
    axes.reset()
    axes.on_motion(2, 0.0, **AXES)
    assert axes.take_brake_went_live() is True


def _press(det: TapDetector, guid: str, axis: int, start: float, end: float, steps: int = 6):
    for i in range(steps + 1):
        det.on_motion(guid, axis, start + (end - start) * i / steps)


def test_tap_is_measured_from_the_first_event_not_from_zero():
    """Rest at -1.0: a 0.0 baseline would call it pressed and inverted at once."""
    det = TapDetector()
    det.on_motion("pedals", 2, -1.0)
    assert det.take_hit() is None
    _press(det, "pedals", 2, -1.0, 0.2)
    hit = det.take_hit()
    assert hit is not None and hit.axis == 2 and hit.inverted is False


def test_inverted_pedal_is_detected_as_inverted():
    det = TapDetector()
    _press(det, "pedals", 1, 1.0, 0.0)
    hit = det.take_hit()
    assert hit is not None and hit.inverted is True


def test_small_motion_is_not_a_tap():
    det = TapDetector()
    _press(det, "wheel", 0, 0.0, TAP_THRESHOLD * 0.9)
    assert det.take_hit() is None


def test_the_axis_that_moved_most_wins():
    det = TapDetector()
    _press(det, "pedals", 1, -1.0, -0.6)   # brushed the gas
    _press(det, "pedals", 2, -1.0, 0.5)    # pressed the brake
    hit = det.take_hit()
    assert hit is not None and hit.axis == 2


def test_gas_stage_ignores_the_brake_and_other_devices():
    det = TapDetector()
    det.restrict("pedals", skip_axis=2)
    _press(det, "pedals", 2, -1.0, 1.0)
    _press(det, "wheel", 0, -1.0, 1.0)
    assert det.take_hit() is None
    _press(det, "pedals", 1, -1.0, 0.5)
    hit = det.take_hit()
    assert hit is not None and (hit.guid, hit.axis) == ("pedals", 1)


def test_light_pedal_tap_counts_but_the_same_move_off_centre_does_not():
    """A light load-cell tap: a pedal rests at an end, a steering axis does not."""
    move = (PEDAL_TAP_THRESHOLD + TAP_THRESHOLD) / 2
    det = TapDetector()
    _press(det, "wheel", 0, 0.0, move)
    assert det.take_hit() is None
    _press(det, "pedals", 2, -1.0, -1.0 + move)
    hit = det.take_hit()
    assert hit is not None and hit.axis == 2 and hit.pedal_like


def test_a_pedal_beats_a_bigger_steering_move():
    det = TapDetector()
    _press(det, "wheel", 0, 0.0, 0.9)
    _press(det, "pedals", 2, -1.0, -0.6)
    hit = det.take_hit()
    assert hit is not None and (hit.guid, hit.axis) == ("pedals", 2)


def test_rest_end_decides_inversion_and_survives_a_full_stomp():
    det = TapDetector()
    det.on_motion("pedals", 1, 0.95)  # rests at +1: pressing lowers it
    _press(det, "pedals", 1, 0.95, -1.0)
    _press(det, "pedals", 1, -1.0, 0.95)
    hit = det.take_hit()
    assert hit is not None and hit.inverted is True


def test_summary_says_which_devices_never_reported():
    det = TapDetector()
    _press(det, "wheel", 0, 0.0, 0.1)
    text = det.summary({"wheel": "Wheel", "pedals": "Pedals"})
    assert "'Pedals': no axis events" in text
    assert "'Wheel': 7 events, max 0.10 on axis 0" in text
