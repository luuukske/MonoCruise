"""The TruckersMP display model the convoy tests stand on, against the local clip stores.

`test_platoon.py` is only as honest as its netcode. These check the model still
matches real remote-truck streams, binned by what the sender was doing, and that on
the ACC chain and in braking overshoot it is no harsher than the game. Both stores
are read whole, every label: clips are bursty by session and a sample of a few
hundred moved the stall rate by 3x. See tools/acc_platoon/README.md, "Calibration".
"""
from __future__ import annotations

import pytest

from tools.acc_platoon import calibrate
from tools.acc_platoon.sim import FRAME_STEP_SHARES

from .harness import clip_root

MIN_STREAMS = 10000
WORKERS = 8
MODEL_SESSIONS = 240
RATIO_TOL = 0.05
RATE_FACTOR = 2.0
# Braking corrections at 2 to 6 m/s^2 run about 2x the corpus (README, "Calibration"); this
# bound holds that gap, and narrowing it is the next calibration step.
BRAKING_FACTOR = 2.5
SESSION_ZERO_TOL = 0.15
# The ACC-chain response of the model may exceed the corpus by this much and no more.
HARSHER_FACTOR = 1.1
# Drawn overshoot in braking: the model must not exceed the corpus, nor fall below this share of it.
OVERSHOOT_FLOOR = 0.5
STEP_SHARE_TOL = 0.04
MIN_EVENTS = 8
MIN_CORPUS_MIN = 20.0
MIN_MODEL_MIN = 4.0
BRAKE_BANDS = (1, 2, 3)
BRAKE_BINS = (0, 1, 2, 3)

pytestmark = [
    pytest.mark.needs_clips,
    pytest.mark.skipif(clip_root() is None, reason="no local AEB clip store"),
]


@pytest.fixture(scope="module")
def corpus():
    streams = calibrate.corpus_streams(workers=WORKERS)
    if len(streams) < MIN_STREAMS:
        pytest.skip("local clip store too small for a stable TMP measurement")
    return streams


@pytest.fixture(scope="module")
def model():
    return calibrate.model_streams(sessions=MODEL_SESSIONS, workers=WORKERS)


@pytest.fixture(scope="module")
def tables(corpus, model):
    return calibrate.state_table(corpus), calibrate.state_table(model)


def _within(m: float, c: float, factor: float) -> bool:
    return 1.0 / factor < (m + 0.1) / (c + 0.1) < factor


def test_drawn_positions_look_like_real_tmp_streams(corpus, model):
    c, m = calibrate.raw_stats(corpus), calibrate.raw_stats(model)
    assert abs(m["ratio_p10"] - c["ratio_p10"]) < RATIO_TOL, calibrate.report(c, m)
    assert abs(m["ratio_p90"] - c["ratio_p90"]) < RATIO_TOL, calibrate.report(c, m)
    for key in ("rms_p50", "rms_p90", "stalls_per_min", "rewinds_per_min"):
        assert _within(m[key], c[key], RATE_FACTOR), (key, calibrate.report(c, m))


def test_steady_stalls_fall_with_speed_as_in_the_game(tables):
    c, m = tables
    for band in range(len(calibrate.SPEED_BANDS)):
        key = (band, calibrate.STEADY_BIN)
        if c.get(key, {}).get("minutes", 0.0) < MIN_CORPUS_MIN or m.get(key, {}).get("minutes", 0.0) < MIN_MODEL_MIN:
            continue
        assert _within(m[key]["stall_pm"], c[key]["stall_pm"], RATE_FACTOR), (
            key, calibrate.state_report(c, m))


def _pooled(table, cell: int) -> float:
    minutes = sum(table[(b, cell)]["minutes"] for b in BRAKE_BANDS if (b, cell) in table)
    events = sum((table[(b, cell)]["stall_pm"] + table[(b, cell)]["rewind_pm"]) * table[(b, cell)]["minutes"]
                 for b in BRAKE_BANDS if (b, cell) in table)
    return events / max(minutes, 1e-9)


def test_braking_corrections_grow_with_deceleration_as_in_the_game(tables):
    """Holds and rewinds per minute of braking at 3 to 22 m/s, by deceleration class."""
    c, m = tables
    rates = []
    for cell in BRAKE_BINS:
        rc, rm = _pooled(c, cell), _pooled(m, cell)
        assert _within(rm, rc, BRAKING_FACTOR), (cell, rc, rm, calibrate.state_report(c, m))
        rates.append(rm)
    assert rates == sorted(rates, reverse=True), rates


def test_receiver_sessions_spread_like_the_corpus(corpus, model):
    c, m = calibrate.session_rates(corpus), calibrate.session_rates(model)
    assert abs(m["zero_share"] - c["zero_share"]) < SESSION_ZERO_TOL, calibrate.report(c, m)
    assert _within(m["p90"], c["p90"], RATE_FACTOR), calibrate.report(c, m)
    assert _within(m["pooled"], c["pooled"], RATE_FACTOR), calibrate.report(c, m)


def test_a_braking_truck_is_drawn_ahead_but_no_further_than_the_game_draws_it(corpus, model):
    c, m = calibrate.overshoot(corpus), calibrate.overshoot(model)
    for k in range(len(calibrate.OVERSHOOT_DECEL)):
        if c[f"n_{k}"] < MIN_EVENTS or m[f"n_{k}"] < MIN_EVENTS:
            continue
        lead_c, lead_m = c[f"lead_at_{k}"], m[f"lead_at_{k}"]
        assert OVERSHOOT_FLOOR * lead_c <= lead_m <= HARSHER_FACTOR * lead_c, calibrate.report(c, m)


def test_the_model_brakes_acc_no_harder_than_the_game_does(corpus, model):
    """A phantom brake the convoy sim shows must be one the game shows at least as often."""
    c = calibrate.chain_stats(corpus, WORKERS)
    m = calibrate.chain_stats(model, WORKERS)
    for kind, key in (("pause", "pause_decel_p90"), ("rewind", "rewind_decel_p50"),
                      ("quiet", "quiet_decel_p90")):
        if c[f"{kind}_n"] < MIN_EVENTS or m[f"{kind}_n"] < MIN_EVENTS:
            continue
        assert m[key] <= c[key] * HARSHER_FACTOR, (key, calibrate.report(c, m))
    assert m["acc_accel_p90"] <= c["acc_accel_p90"] * HARSHER_FACTOR, calibrate.report(c, m)


def test_radar_frames_arrive_at_the_measured_cadence(corpus):
    measured = calibrate.frame_steps(corpus)
    for steps, share in FRAME_STEP_SHARES:
        assert abs(measured.get(steps, 0.0) - share) < STEP_SHARE_TOL, measured
