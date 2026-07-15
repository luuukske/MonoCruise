"""Unit tests for OccupancyConfirm: the lapse-tolerant confirm streak that
backs the three AEB engagement/warn windows (risk, engage, warn).

The bug these guard: the old hard-reset streaks restarted their whole confirm
clock (or dropped a per-target id) on a single unqualified frame, so per-frame
detection flicker turned one solid threat into repeated full restarts. These
tests pin the two design constraints:
  (a) tolerate isolated 1-2 frame dropouts inside an otherwise-solid streak,
  (b) still reject a signal that is mostly absent.
"""

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
    # A solid streak with one missed frame in the middle still confirms on
    # schedule. The hard-reset predecessor would have restarted the 0.20 s
    # clock at the dropout and delayed the brake by ~0.10 s.
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
    # qualify / miss / miss forever: gap_run maxes at 2 (never resets) but the
    # qualified fraction is ~1/3, below aeb_confirm_occupancy. Must never fire
    # no matter how long it runs.
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
    # Invariant: oblique warn must precede oblique engagement by >= 0.1 s.
    # Warn qualifies on a superset of the engage frames; the worst case for
    # the ordering is an identical stream. Even then, the shorter warn window
    # confirms at least 0.1 s earlier.
    stream = [True] * 12
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
    # The engage gate re-sets window_s each frame (near-certain vs oblique).
    # A streak that has been observed for 0.10 s under the oblique window is
    # not yet confirmed, but flipping to the shorter near-certain window
    # confirms it immediately.
    oc = _new(CAL.aeb_engage_confirm_oblique_s)
    for i in range(4):  # ~0.10 s of qualification
        oc.observe(i * _DT, True)
    assert not oc.confirmed(3 * _DT)
    oc.window_s = CAL.aeb_engage_confirm_s  # geometry became near-certain
    assert oc.confirmed(3 * _DT)
