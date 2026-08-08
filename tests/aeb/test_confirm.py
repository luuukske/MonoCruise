"""OccupancyConfirm lapse-tolerant streak tests (risk / engage / warn windows)."""
from __future__ import annotations

from core.aeb.confirm import OccupancyConfirm
from core.aeb.calibration import DEFAULT as CAL


_DT = 1.0 / 30.0  # 30 Hz, one frame ~= 33 ms


def _new(window_s: float) -> OccupancyConfirm:
    return OccupancyConfirm(
        CAL.aeb_confirm_occupancy, CAL.aeb_confirm_max_gap_frames, window_s,
    )


def _first_confirm_frame(pattern: list[bool], window_s: float) -> int | None:
    """Feed a per-frame qualification pattern at 30 Hz and return the index of
    the first frame at which the streak confirms, or None if it never does."""
    oc = _new(window_s)
    for i, q in enumerate(pattern):
        oc.observe(i * _DT, q)
        if oc.confirmed(i * _DT):
            return i
    return None


def test_solid_streak_confirms_after_window_elapses():
    window = 0.20
    # Continuous qualification: must NOT confirm before the window elapses
    # (~6 frames), must confirm shortly after.
    oc = _new(window)
    for i in range(4):
        oc.observe(i * _DT, True)
    assert not oc.confirmed(3 * _DT)  # 0.10 s in, window not elapsed
    frame = _first_confirm_frame([True] * 12, window)
    assert frame is not None
    assert 5 <= frame <= 7  # 0.20 s / 0.033 s ~= 6


def test_single_dropout_does_not_restart_window():
    # One missed frame mid-streak must not restart the confirm window (hard-reset regression).
    window = 0.20
    pattern = [True, True, True, False, True, True, True, True, True]
    frame = _first_confirm_frame(pattern, window)
    assert frame is not None
    assert frame <= 7  # confirmed on time despite the frame-3 dropout


def test_two_isolated_dropouts_tolerated():
    # Two non-consecutive dropouts stay within occupancy and gap tolerance.
    window = 0.20
    pattern = [True, True, False, True, False, True, True, True, True, True]
    frame = _first_confirm_frame(pattern, window)
    assert frame is not None
    assert frame <= 8


def test_long_gap_drops_streak():
    # More than aeb_confirm_max_gap_frames (2) consecutive misses drops the
    # streak: first_mono clears, so the wait restarts from the next qualify.
    window = 0.20
    oc = _new(window)
    for i in range(5):
        oc.observe(i * _DT, True)
    assert oc.tracking
    for i in range(5, 8):  # 3 consecutive misses
        oc.observe(i * _DT, False)
    assert not oc.tracking  # dropped
    # A single qualify afterwards must not immediately confirm.
    oc.observe(8 * _DT, True)
    assert not oc.confirmed(8 * _DT)


def test_sparse_blips_never_confirm():
    # Sparse qualify pattern stays below aeb_confirm_occupancy; must never confirm.
    window = 0.20
    pattern = ([True, False, False] * 20)
    assert _first_confirm_frame(pattern, window) is None


def test_reset_clears_state():
    oc = _new(0.20)
    for i in range(8):
        oc.observe(i * _DT, True)
    assert oc.confirmed(7 * _DT)
    oc.reset()
    assert not oc.tracking
    assert not oc.confirmed(8 * _DT)


def test_warn_confirms_before_oblique_engage_on_shared_stream():
    # Oblique warn window must confirm at least 0.1 s before oblique engage on same stream.
    stream = [True] * 24  # covers engage_confirm_oblique_s=0.40 at 30 Hz
    warn_frame = _first_confirm_frame(stream, CAL.aeb_warn_confirm_oblique_s)
    engage_frame = _first_confirm_frame(stream, CAL.aeb_engage_confirm_oblique_s)
    assert warn_frame is not None and engage_frame is not None
    assert warn_frame < engage_frame
    assert (engage_frame - warn_frame) * _DT >= 0.10 - 1e-9


def test_calibration_preserves_warn_lead_invariant():
    # The documented ordering guarantee is a constraint on the constants.
    assert (CAL.aeb_warn_confirm_oblique_s
            <= CAL.aeb_engage_confirm_oblique_s - 0.1 + 1e-9)


def test_window_can_shrink_and_confirm_earlier():
    # Shorter near-certain engage window can confirm immediately after oblique streak built.
    oc = _new(CAL.aeb_engage_confirm_oblique_s)
    for i in range(4):  # ~0.10 s of qualification
        oc.observe(i * _DT, True)
    assert not oc.confirmed(3 * _DT)
    oc.window_s = CAL.aeb_engage_confirm_s  # geometry became near-certain
    assert oc.confirmed(3 * _DT)
