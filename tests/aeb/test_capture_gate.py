"""Capture gate: who records, with what cap, and that contributors skip background negatives."""
from __future__ import annotations

import pytest

from core.aeb import capture as capture_mod
from core.aeb.recorder import AEBClipRecorder
from core.aeb.clip_store import AsyncClipWriter, ClipStore


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    """A capture module reset between tests, with the store pointed at tmp_path."""
    monkeypatch.setattr(capture_mod, "_recorder", None)
    monkeypatch.setattr(capture_mod, "_writer", None)
    monkeypatch.setattr(capture_mod, "_initialized", False)

    made: dict = {}
    real_store = ClipStore

    def _store(*args, **kwargs):
        made["max_bytes"] = kwargs.get("max_bytes")
        return real_store(root=tmp_path, **kwargs)

    monkeypatch.setattr(capture_mod, "ClipStore", _store)
    monkeypatch.setattr(capture_mod.AsyncClipWriter, "start", lambda self: None)
    yield made
    capture_mod._recorder = None
    capture_mod._writer = None
    capture_mod._initialized = False


def _gate(monkeypatch, *, debug: bool, contributing: bool) -> None:
    from core.settings import Settings

    # The metaclass proxies class attribute reads to the singleton, so patching
    # the class itself would be silently ignored.
    monkeypatch.setattr(Settings.instance(), "debug", debug, raising=False)
    monkeypatch.setattr(
        "core.aeb.intake_policy.contribution_enabled", lambda: contributing
    )


def test_neither_debug_nor_contributing_records_nothing(fresh, monkeypatch):
    _gate(monkeypatch, debug=False, contributing=False)
    assert capture_mod.get_recorder() is None


def test_opting_in_starts_the_recorder_without_debug(fresh, monkeypatch):
    _gate(monkeypatch, debug=False, contributing=True)
    assert capture_mod.get_recorder() is not None


def test_debug_still_records_without_opting_in(fresh, monkeypatch):
    _gate(monkeypatch, debug=True, contributing=False)
    assert capture_mod.get_recorder() is not None


def test_a_contributor_gets_the_smaller_cap(fresh, monkeypatch):
    _gate(monkeypatch, debug=False, contributing=True)
    capture_mod.get_recorder()
    assert fresh["max_bytes"] == capture_mod._CONTRIBUTOR_MAX_BYTES


def test_debug_keeps_the_full_cap(fresh, monkeypatch):
    """The debug store is the working corpus, not a staging area."""
    _gate(monkeypatch, debug=True, contributing=True)
    capture_mod.get_recorder()
    assert fresh["max_bytes"] is None      # ClipStore default, 500 MB


def test_a_contributor_does_not_capture_background_negatives(fresh, monkeypatch):
    _gate(monkeypatch, debug=False, contributing=True)
    assert capture_mod.get_recorder().capture_tn is False


def test_debug_still_captures_background_negatives(fresh, monkeypatch):
    _gate(monkeypatch, debug=True, contributing=True)
    assert capture_mod.get_recorder().capture_tn is True


def _recorder(capture_tn: bool) -> AEBClipRecorder:
    return AEBClipRecorder(AsyncClipWriter(ClipStore()), capture_tn=capture_tn)


@pytest.mark.parametrize("source", ["shadow_near", "random"])
def test_tn_triggers_are_refused_when_disabled(source):
    assert _recorder(False).trigger(source, at=100.0) == "ignored"
    assert _recorder(True).trigger(source, at=100.0) == "started"


@pytest.mark.parametrize("source", ["auto_engagement", "auto_crash", "manual"])
def test_real_triggers_still_fire_when_tn_is_disabled(source):
    assert _recorder(False).trigger(source, at=100.0) == "started"


def test_a_refused_tn_does_not_even_fold_into_a_pending_clip(fresh):
    """Rejected before the fold, so it never reaches also_triggered either."""
    rec = _recorder(False)
    assert rec.trigger("auto_engagement", at=100.0) == "started"
    assert rec.trigger("shadow_near", at=100.1) == "ignored"
    assert rec._pending.also_triggered == []


def test_a_tn_still_folds_for_a_debug_user(fresh):
    rec = _recorder(True)
    rec.trigger("auto_engagement", at=100.0)
    assert rec.trigger("shadow_near", at=100.1) == "folded"
    assert "shadow_near" in rec._pending.also_triggered
