"""Synthetic-trace tests for the shared speed/accel filter chain.

Covers the acc_speed anti-oscillation changes (trend-gated fast tau,
standstill latch) and the faster front-end accel filter. Feeds
_smooth_vehicle_kinematics directly; no game or shared memory involved.
See core/radar/AGENTS.md §7.
"""

from __future__ import annotations

import math

from core.radar.traffic import _smooth_vehicle_kinematics

DT = 0.05  # full-update cadence (s)


def _run_chain(raw_speeds: list[float], dt: float = DT):
    """Run the chain over a raw-speed trace, carrying state like update_from_last."""
    t = 100.0
    speed_ema = accel = acc_speed = None
    history = None
    standstill = False
    release_s = 0.0
    out = []
    for raw in raw_speeds:
        (speed_ema, accel, speed_corr, acc_speed, history,
         standstill, release_s) = _smooth_vehicle_kinematics(
            raw, t, dt, speed_ema, accel, acc_speed, history,
            standstill, release_s,
        )
        out.append((speed_corr, acc_speed, accel, standstill))
        t += dt
    return out


def _bounce(amp: float, freq_hz: float, dur_s: float, dt: float = DT) -> list[float]:
    n = int(dur_s / dt)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i * dt) for i in range(n)]


def _ramp(v0: float, a: float, dur_s: float, dt: float = DT) -> list[float]:
    n = int(dur_s / dt)
    return [v0 + a * i * dt for i in range(n)]


def _crash_stop_trace() -> list[float]:
    """Cruise, hard decel to 0, then a crash-style rock around standstill."""
    trace = [8.0] * int(2.0 / DT)
    trace += _ramp(8.0, -4.0, 2.0)
    trace += _bounce(1.0, 1.0, 4.0)
    return trace


def test_standstill_bounce_latches_acc_speed_to_zero():
    out = _run_chain(_crash_stop_trace())
    tail = out[-int(1.5 / DT):]
    assert all(acc_speed == 0.0 for _, acc_speed, _, _ in tail)
    assert all(standstill for _, _, _, standstill in tail)


def test_standstill_bounce_keeps_aeb_speed_responsive():
    out = _run_chain(_crash_stop_trace())
    tail = out[-int(2.0 / DT):]
    # The AEB-facing speed_corr must still show the raw oscillation.
    assert max(abs(speed_corr) for speed_corr, _, _, _ in tail) > 0.5


def test_bounce_does_not_release_latch():
    trace = _crash_stop_trace()
    trace += _bounce(1.0, 1.0, 3.0)
    out = _run_chain(trace)
    # Once latched, a continuing +-1 m/s 1 Hz rock never releases the latch.
    latched_from = next(i for i, (_, _, _, ss) in enumerate(out) if ss)
    assert all(ss for _, _, _, ss in out[latched_from:])


def test_hard_brake_ramp_tracks_with_bounded_lag():
    trace = [25.0] * int(2.0 / DT)
    trace += _ramp(25.0, -6.0, 25.0 / 6.0)
    trace += [0.0] * int(2.5 / DT)
    out = _run_chain(trace)
    ramp_start = int(2.0 / DT)
    ramp_end = ramp_start + int(25.0 / 6.0 / DT)
    # After the trend gate opens (~0.6 s), acc_speed rides the ramp closely.
    settled = out[ramp_start + int(0.6 / DT):ramp_end]
    assert max(abs(acc - corr) for corr, acc, _, _ in settled) < 1.0
    # After the stop it settles and latches to exactly 0.
    assert out[-1][1] == 0.0
    assert out[-1][3] is True


def test_launch_releases_latch_and_tracks():
    trace = _crash_stop_trace()
    trace += _ramp(0.0, 1.5, 2.5)
    out = _run_chain(trace)
    assert out[len(_crash_stop_trace()) - 1][3] is True
    # Raw crosses 0.6 m/s at 0.4 s into the launch; release needs 0.5 s more.
    end_corr, end_acc, _, end_ss = out[-1]
    assert end_ss is False
    assert end_acc > 0.5 * end_corr


def test_convoy_sawtooth_attenuated_at_cruise():
    # TMP reconciliation shape: slow drift down, snap back up, zero mean.
    # Each drift phase looks like a real ramp on the short trend window; the
    # consistency factor must keep the feed-forward shut so acc_speed stays flat.
    period = 3.0
    n = int(24.0 / DT)
    trace = [22.0 + 0.8 * (1.0 - 2.0 * ((i * DT % period) / period)) for i in range(n)]
    out = _run_chain(trace)
    tail = out[-int(10.0 / DT):]
    corr_dev = max(abs(corr - 22.0) for corr, _, _, _ in tail)
    acc_dev = max(abs(acc - 22.0) for _, acc, _, _ in tail)
    # Without the consistency factor this rode at ~0.8x of the input wobble.
    assert acc_dev < 0.30
    assert acc_dev < 0.4 * corr_dev


def test_cruise_ripple_still_attenuated():
    n = int(6.0 / DT)
    trace = [25.0 + 0.5 * math.sin(2.0 * math.pi * 1.0 * i * DT) for i in range(n)]
    out = _run_chain(trace)
    tail = out[-int(3.0 / DT):]
    corr_dev = max(abs(corr - 25.0) for corr, _, _, _ in tail)
    acc_dev = max(abs(acc - 25.0) for _, acc, _, _ in tail)
    assert acc_dev < 0.3 * corr_dev


def test_accel_estimate_reacts_within_half_second():
    trace = [20.0] * int(2.0 / DT)
    trace += _ramp(20.0, -5.0, 2.0)
    out = _run_chain(trace)
    at_half_s = out[int(2.0 / DT) + int(0.5 / DT)]
    # Old front end (fit 0.80, ema 0.40) read -1.91 here; fit 0.70 / ema 0.45
    # reads -2.40. Threshold sits between so a filtering regression fails.
    assert at_half_s[2] < -2.2
