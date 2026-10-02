"""Intake policy: opt-in gating, fail-closed refusal, throttling, and the kill switch."""
from __future__ import annotations

import json

import pytest

from core.aeb import intake_policy as ip
from core.settings import Settings

_OPEN = {
    "policy_version": 3,
    "accepting": True,
    "min_client_version": "1.0.0",
    "min_schema_version": 3,
    "max_clip_bytes": 2 * 1024 * 1024,
    "refresh_hours": 12,
}


@pytest.fixture()
def opted_in(monkeypatch):
    """Opted in under the current consent text, with no policy cached yet."""
    monkeypatch.setattr(ip, "contribution_enabled", lambda: True)
    Settings.save(values={"aeb_intake_policy_json": "", "aeb_intake_checked": 0.0})
    yield
    Settings.save(values={"aeb_intake_policy_json": "", "aeb_intake_checked": 0.0})


def _blocked(policy, *, clip_bytes=100_000, client_version="1.1.0", schema_version=3):
    return ip.upload_blocked_reason(
        policy, clip_bytes=clip_bytes,
        client_version=client_version, schema_version=schema_version,
    )


def test_nothing_uploads_without_a_policy(opted_in):
    """Fail closed: never having reached the server must not mean 'go ahead'."""
    assert _blocked(None) == "no intake policy"


def test_an_open_policy_allows_the_upload(opted_in):
    assert _blocked(ip.IntakePolicy.from_json(_OPEN)) is None


def test_accepting_false_is_the_kill_switch(opted_in):
    policy = ip.IntakePolicy.from_json({**_OPEN, "accepting": False})
    assert _blocked(policy) == "intake closed"


def test_defaults_refuse_everything():
    """An empty document must not read as permission."""
    assert ip.IntakePolicy.from_json({}).accepting is False


def test_oversized_and_stale_clips_are_refused(opted_in):
    policy = ip.IntakePolicy.from_json(_OPEN)
    assert _blocked(policy, clip_bytes=policy.max_clip_bytes + 1) == "clip too large"
    assert _blocked(policy, schema_version=2) == "clip schema too old"


def test_an_old_client_is_refused(opted_in):
    policy = ip.IntakePolicy.from_json({**_OPEN, "min_client_version": "2.0.0"})
    assert _blocked(policy, client_version="1.9.0") == "client too old"
    assert _blocked(policy, client_version="2.0.0") is None


def test_a_bare_release_floor_refuses_every_preview_of_that_release(opted_in):
    """PEP 440 orders a prerelease before its release, so a floor of "1.1.0"
    turns away 1.1.0-preview.N. That is correct ordering and the wrong policy: a
    floor written as a bare release number while the channel ships prereleases
    refuses the entire fleet, silently, with "client too old" at debug level.
    Found by an end-to-end run against the real endpoint, so the server's policy
    floor names a prerelease. Do not "fix" this in the client."""
    policy = ip.IntakePolicy.from_json({**_OPEN, "min_client_version": "1.1.0"})
    assert _blocked(policy, client_version="1.1.0-preview.14") == "client too old"

    admits_previews = ip.IntakePolicy.from_json({**_OPEN, "min_client_version": "1.1.0-preview.14"})
    assert _blocked(admits_previews, client_version="1.1.0-preview.14") is None
    assert _blocked(admits_previews, client_version="1.1.0") is None


def test_the_running_client_would_not_be_refused_by_its_own_version(opted_in):
    from core.version import __version__

    policy = ip.IntakePolicy.from_json({**_OPEN, "min_client_version": __version__})
    assert _blocked(policy, client_version=__version__) is None


def test_an_unparseable_version_does_not_satisfy_the_floor(opted_in):
    policy = ip.IntakePolicy.from_json({**_OPEN, "min_client_version": "2.0.0"})
    assert _blocked(policy, client_version="not-a-version") == "client too old"


def test_opting_out_blocks_upload_whatever_the_policy_says(monkeypatch):
    monkeypatch.setattr(ip, "contribution_enabled", lambda: False)
    assert _blocked(ip.IntakePolicy.from_json(_OPEN)) == "not opted in"


def test_no_fetch_thread_starts_when_not_opted_in(monkeypatch):
    """A user who never opted in makes no request to the server at all."""
    monkeypatch.setattr(ip, "contribution_enabled", lambda: False)
    monkeypatch.setattr(ip, "_fetch_policy_text", _must_not_run)
    assert ip.start_policy_fetch() is None


def test_a_fetch_caches_the_raw_document(opted_in, monkeypatch):
    monkeypatch.setattr(ip, "_fetch_policy_text", lambda: json.dumps(_OPEN))
    policy = ip._run_fetch()

    assert policy is not None and policy.accepting is True
    assert ip.cached_policy().policy_version == 3
    assert float(Settings.aeb_intake_checked) > 0.0


def test_unknown_fields_survive_the_cache_round_trip(opted_in, monkeypatch):
    """The raw text is stored so a future server field is not silently dropped."""
    monkeypatch.setattr(ip, "_fetch_policy_text", lambda: json.dumps({**_OPEN, "future": 42}))
    ip._run_fetch()
    assert json.loads(Settings.aeb_intake_policy_json)["future"] == 42


def test_a_second_fetch_inside_the_window_is_throttled(opted_in, monkeypatch):
    monkeypatch.setattr(ip, "_fetch_policy_text", lambda: json.dumps(_OPEN))
    ip._run_fetch()
    monkeypatch.setattr(ip, "_fetch_policy_text", _must_not_run)
    assert ip._run_fetch().policy_version == 3


def test_a_broken_response_keeps_the_previous_policy(opted_in, monkeypatch):
    monkeypatch.setattr(ip, "_fetch_policy_text", lambda: json.dumps(_OPEN))
    ip._run_fetch()
    Settings.save(values={"aeb_intake_checked": 0.0})       # let it try again

    monkeypatch.setattr(ip, "_fetch_policy_text", lambda: "not json at all")
    with pytest.raises(json.JSONDecodeError):
        ip._run_fetch()
    assert ip.cached_policy().policy_version == 3


def test_a_non_object_document_is_rejected(opted_in, monkeypatch):
    monkeypatch.setattr(ip, "_fetch_policy_text", lambda: "[1, 2, 3]")
    with pytest.raises(ValueError):
        ip._run_fetch()
    assert ip.cached_policy() is None


def _must_not_run():
    raise AssertionError("the network seam must not be reached here")
