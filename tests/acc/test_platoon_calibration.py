"""The TruckersMP display model the convoy tests stand on, against the local clip store.

`test_platoon.py` is only as honest as its netcode. These check the model still
matches real remote-truck streams, and that on the ACC chain it is no harsher than
the game. The whole store is read: clips are bursty by session and a sample of a few
hundred moved the pause rate by 3x. See tools/acc_platoon/README.md, "Calibration"."""
from __future__ import annotations

import pytest

from tools.acc_platoon import calibrate

from .harness import clip_root

MIN_STREAMS = 1000
WORKERS = 8
RATIO_TOL = 0.05
RATE_FACTOR = 2.0
# The ACC-chain response of the model may exceed the corpus by this much and no more.
HARSHER_FACTOR = 1.1
MIN_EVENTS = 8

pytestmark = [
    pytest.mark.needs_clips,
    pytest.mark.skipif(clip_root() is None, reason="no local AEB clip store"),
]


@pytest.fixture(scope="module")
def corpus():
    streams = calibrate.corpus_streams(clip_root(), workers=WORKERS)
    if len(streams) < MIN_STREAMS:
        pytest.skip("local clip store too small for a stable TMP measurement")
    return streams


@pytest.fixture(scope="module")
def model():
    return calibrate.model_streams()


def test_drawn_positions_look_like_real_tmp_streams(corpus, model):
    c, m = calibrate.raw_stats(corpus), calibrate.raw_stats(model)
    assert abs(m["ratio_p10"] - c["ratio_p10"]) < RATIO_TOL, calibrate.report(c, m)
    assert abs(m["ratio_p90"] - c["ratio_p90"]) < RATIO_TOL, calibrate.report(c, m)
    for key in ("rms_p50", "rms_p90", "stalls_per_min", "rewinds_per_min"):
        assert 1.0 / RATE_FACTOR < m[key] / c[key] < RATE_FACTOR, (key, calibrate.report(c, m))


def test_the_model_brakes_acc_no_harder_than_the_game_does(corpus, model):
    """A phantom brake the convoy sim shows must be one the game shows at least as often."""
    c, m = calibrate.artefact_response(corpus), calibrate.artefact_response(model)
    for kind, key in (("pause", "pause_decel_p90"), ("rewind", "rewind_decel_p50"),
                      ("quiet", "quiet_decel_p90")):
        if c[f"{kind}_n"] < MIN_EVENTS:
            continue
        assert m[key] <= c[key] * HARSHER_FACTOR, (key, calibrate.report(c, m))
