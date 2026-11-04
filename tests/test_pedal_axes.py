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
    split_pedal,
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


def test_separate_pedals_keep_the_end_to_end_map():
    """Shared-axis support must not move the map every other pedal uses."""
    assert normalise_axis(-1.0, False) == 0.0
    assert normalise_axis(0.0, False) == 0.5
    assert normalise_axis(1.0, False) == 1.0
    assert normalise_axis(-1.0, True) == 1.0
    assert normalise_axis(1.0, True) == 0.0
    assert normalise_axis(0.25, False) == 0.625


def test_split_pedal_is_one_half_and_the_halves_never_overlap():
    assert split_pedal(0.0, False) == 0.0
    assert split_pedal(1.0, False) == 1.0
    assert split_pedal(-1.0, False) == 0.0
    assert split_pedal(0.5, False) == 0.5
    assert split_pedal(-0.5, True) == 0.5
    assert split_pedal(-1.0, True) == 1.0
    assert split_pedal(1.2, False) == 1.0
    for step in range(-10, 11):
        raw = step / 10
        gas = split_pedal(raw, True)
        brake = split_pedal(raw, False)
        assert gas == 0.0 or brake == 0.0
        assert 0.0 <= gas <= 1.0 and 0.0 <= brake <= 1.0


def test_the_shared_flag_does_not_remap_separate_axes():
    """A stale flag, or the same axis without the flag, keeps the old map."""
    separate = PedalAxes()
    separate.on_motion(
        1, 0.0, gas_axis=1, brake_axis=2, gas_inverted=False, brake_inverted=False, shared=True,
    )
    assert separate.gas == 0.5

    same_axis = PedalAxes()
    same_axis.on_motion(
        1, 0.0, gas_axis=1, brake_axis=1, gas_inverted=False, brake_inverted=False, shared=False,
    )
    assert (same_axis.gas, same_axis.brake) == (0.5, 0.5)


def test_a_shared_axis_splits_around_the_middle():
    axes = PedalAxes()
    kw = dict(gas_axis=1, brake_axis=1, gas_inverted=True, brake_inverted=False, shared=True)
    axes.on_motion(0, 0.8, **kw)  # steering on the same stick
    assert (axes.gas, axes.brake) == (0.0, 0.0)
    assert axes.take_brake_went_live() is False

    axes.on_motion(1, 0.0, **kw)
    assert (axes.gas, axes.brake) == (0.0, 0.0)
    assert axes.take_brake_went_live() is True
    axes.on_motion(1, 1.0, **kw)
    assert (axes.gas, axes.brake) == (0.0, 1.0)
    assert axes.take_brake_went_live() is False
    axes.on_motion(1, -0.5, **kw)
    assert (axes.gas, axes.brake) == (0.5, 0.0)

    reads = []

    def read(axis):
        reads.append(axis)
        return -1.0

    axes.refresh(read, **kw)
    assert reads == [1]
    assert (axes.gas, axes.brake) == (1.0, 0.0)


def test_a_shared_axis_is_not_re_read_before_it_has_reported():
    axes = PedalAxes()
    kw = dict(gas_axis=1, brake_axis=1, gas_inverted=True, brake_inverted=False, shared=True)
    reads = []
    axes.refresh(lambda axis: reads.append(axis) or 0.4, **kw)
    assert reads == []
    assert (axes.gas, axes.brake) == (0.0, 0.0)


def test_an_end_rest_pedal_is_not_shared_when_it_crosses_the_middle():
    det = TapDetector()
    _press(det, "pedals", 2, -1.0, 1.0)
    hit = det.take_hit()
    assert hit is not None and hit.pedal_like and not hit.combined and hit.inverted is False
    det.restrict("pedals", skip_axis=2)
    _press(det, "pedals", 2, 1.0, -1.0)
    assert det.take_hit() is None


def test_steering_across_both_stops_does_not_steal_a_pedal_tap():
    """A wheel that rests in the middle used to become an inverted pedal at full lock."""
    det = TapDetector()
    _press(det, "wheel", 0, 0.0, 1.0)
    _press(det, "wheel", 0, 1.0, -1.0)
    _press(det, "pedals", 2, -1.0, -0.8)
    hit = det.take_hit()
    assert hit is not None and hit.axis == 2 and hit.pedal_like and not hit.combined


def test_middle_rest_survives_a_press_to_the_stop():
    det = TapDetector()
    det.on_motion("joy", 1, 0.1)
    assert det.take_hit() is None
    _press(det, "joy", 1, 0.1, 1.0)
    _press(det, "joy", 1, 1.0, 0.1)
    hit = det.take_hit()
    assert hit is not None and hit.combined and not hit.pedal_like
    assert hit.inverted is False and hit.negative_half is False


def test_just_outside_the_middle_can_still_adopt_an_end():
    det = TapDetector()
    _press(det, "joy", 1, 0.11, 1.0)
    _press(det, "joy", 1, 1.0, 0.2)
    hit = det.take_hit()
    assert hit is not None and hit.pedal_like and hit.inverted is True and not hit.combined


def test_a_clear_press_of_a_middle_rest_axis_is_a_tap_and_a_small_one_is_not():
    det = TapDetector()
    _press(det, "joy", 1, 0.0, TAP_THRESHOLD)
    assert det.take_hit() is None
    _press(det, "joy", 1, 0.0, TAP_THRESHOLD + 0.01)
    hit = det.take_hit()
    assert hit is not None and hit.combined and hit.axis == 1 and hit.negative_half is False


def test_the_first_direction_of_a_shared_axis_stays_the_brake():
    det = TapDetector()
    _press(det, "joy", 1, 0.0, 0.5)
    _press(det, "joy", 1, 0.5, -0.9)
    hit = det.take_hit()
    assert hit is not None and hit.combined and hit.negative_half is False
    assert hit.magnitude == 0.5


def test_one_sweep_already_holds_the_gas_half():
    det = TapDetector()
    _press(det, "joy", 1, 0.0, -0.6)
    _press(det, "joy", 1, -0.6, 0.6)
    brake = det.take_hit()
    assert brake is not None and brake.negative_half is True
    det.restrict("joy", 1, gas_negative=False)
    gas = det.take_hit()
    assert gas is not None and gas.shared_gas and gas.axis == 1 and gas.negative_half is False


def test_a_small_swing_past_the_middle_is_not_the_gas():
    det = TapDetector()
    _press(det, "joy", 1, 0.0, 0.6)
    _press(det, "joy", 1, 0.6, -0.2)
    assert det.take_hit() is not None
    det.restrict("joy", 1, gas_negative=True)
    assert det.take_hit() is None
    _press(det, "joy", 1, -0.2, -0.45)
    gas = det.take_hit()
    assert gas is not None and gas.shared_gas and gas.negative_half is True


def test_the_gas_half_beats_steering_and_loses_to_a_real_pedal():
    det = TapDetector()
    _press(det, "joy", 1, 0.0, 0.6)
    assert det.take_hit() is not None
    det.restrict("joy", 1, gas_negative=True)
    _press(det, "joy", 0, 0.0, 0.95)
    _press(det, "joy", 1, 0.0, -0.4)
    _press(det, "joy", 2, -1.0, -0.6)
    hit = det.take_hit()
    assert hit is not None and hit.pedal_like and hit.axis == 2 and not hit.shared_gas

    det = TapDetector()
    _press(det, "joy", 1, 0.0, 0.6)
    det.take_hit()
    det.restrict("joy", 1, gas_negative=True)
    _press(det, "joy", 0, 0.0, 0.95)
    _press(det, "joy", 1, 0.0, -0.4)
    hit = det.take_hit()
    assert hit is not None and hit.shared_gas and hit.axis == 1 and hit.guid == "joy"
