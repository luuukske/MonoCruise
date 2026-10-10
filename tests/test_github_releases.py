"""Release lists must survive the GitHub REST API rate limit (shared VPN exit addresses).

The updater and the boot update check fall back to the public releases feed. See
"GitHub releases" in shared/README.md. Nothing here touches the network.
"""
from __future__ import annotations

import pytest
from requests.structures import CaseInsensitiveDict

from shared import github_releases as gr
from tests._updater_loader import load_updater  # noqa: F401  (puts updater/ on sys.path)

import github_api  # noqa: E402

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en-US">
  <entry>
    <id>tag:github.com,2008:Repository/1/v1.2.0-preview.1</id>
    <link rel="alternate" type="text/html" href="https://github.com/o/r/releases/tag/v1.2.0-preview.1"/>
    <title>MonoCruise v1.2.0-preview.1</title>
    <content type="html">&lt;h3&gt;Fixed&lt;/h3&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;ACC&lt;/strong&gt;: brakes &amp;amp; releases on time.&lt;/li&gt;
&lt;/ul&gt;</content>
  </entry>
  <entry>
    <id>tag:github.com,2008:Repository/1/v1.1.0</id>
    <link rel="alternate" type="text/html" href="https://github.com/o/r/releases/tag/v1.1.0"/>
    <title>MonoCruise v1.1.0</title>
    <content type="html">&lt;p&gt;Stable.&lt;/p&gt;</content>
  </entry>
</feed>"""

_API_RELEASES = [{"id": 7, "tag_name": "v1.2.0", "prerelease": False}]


class _Response:
    def __init__(self, status: int, *, headers=None, text: str = "", payload=None):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers or {})
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


_RATE_LIMITED = _Response(
    403,
    headers={"x-ratelimit-remaining": "0"},
    text='{"message":"API rate limit exceeded for 203.0.113.7."}',
)


def _serve(monkeypatch, *, api, feed):
    """Answer the API and the feed URL with the given responses (or raise them)."""
    calls: list[str] = []

    def _get(url, **_kwargs):
        calls.append(url)
        response = api if "api.github.com" in url else feed
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(gr.requests, "get", _get)
    return calls


def test_rate_limit_detection():
    assert gr.is_rate_limited(_RATE_LIMITED)
    assert gr.is_rate_limited(_Response(429, headers={"Retry-After": "60"}))
    assert not gr.is_rate_limited(_Response(403, text="Forbidden"))
    assert not gr.is_rate_limited(_Response(200))


def test_failure_reason_never_quotes_the_address():
    reason = gr.describe_failure(_RATE_LIMITED)
    assert "rate limit" in reason
    assert "203.0.113.7" not in reason


def test_the_api_is_used_while_it_answers(monkeypatch):
    calls = _serve(monkeypatch, api=_Response(200, payload=_API_RELEASES), feed=None)

    assert gr.fetch_releases("o", "r", timeout=1) == _API_RELEASES
    assert len(calls) == 1


def test_a_rate_limited_api_falls_back_to_the_feed(monkeypatch):
    calls = _serve(monkeypatch, api=_RATE_LIMITED, feed=_Response(200, text=_FEED))

    releases = gr.fetch_releases("o", "r", timeout=1)

    assert calls[1] == "https://github.com/o/r/releases.atom"
    assert [r["tag_name"] for r in releases] == ["v1.2.0-preview.1", "v1.1.0"]
    assert [r["prerelease"] for r in releases] == [True, False]
    preview = releases[0]
    assert preview["id"] == "v1.2.0-preview.1"
    assert preview["name"] == "MonoCruise v1.2.0-preview.1"
    assert preview["assets"] == [{
        "name": "Update-v1.2.0-preview.1.zip",
        "browser_download_url":
            "https://github.com/o/r/releases/download/v1.2.0-preview.1/Update-v1.2.0-preview.1.zip",
    }]
    assert preview["body"] == "### Fixed\n\n- **ACC**: brakes & releases on time."


def test_an_unreachable_api_also_falls_back(monkeypatch):
    _serve(
        monkeypatch,
        api=gr.requests.ConnectionError("reset"),
        feed=_Response(200, text=_FEED),
    )
    assert len(gr.fetch_releases("o", "r", timeout=1)) == 2


def test_both_sources_down_raises(monkeypatch):
    _serve(monkeypatch, api=_RATE_LIMITED, feed=_Response(429))

    with pytest.raises(gr.ReleaseSourceError) as info:
        gr.fetch_releases("o", "r", timeout=1)
    assert "rate limit" in str(info.value)


def test_html_to_markdown_covers_release_note_markup():
    html = (
        "<h2>Added</h2><p>Intro <em>text</em> with <code>code</code> and "
        '<a href="https://x.test/a">a link</a>.</p>'
        "<ol><li>one</li><li>two<ul><li>nested</li></ul></li></ol>"
        "<pre><code>keep   spacing</code></pre><p>it&#39;s done</p>"
    )
    assert gr.html_to_markdown(html) == (
        "## Added\n\n"
        "Intro *text* with `code` and [a link](https://x.test/a).\n\n"
        "1. one\n2. two\n  - nested\n\n"
        "```\nkeep   spacing\n```\n\n"
        "it's done"
    )


def test_updater_uses_the_fallback(monkeypatch):
    _serve(monkeypatch, api=_RATE_LIMITED, feed=_Response(200, text=_FEED))
    api = github_api.GitHubAPI("o", "r")

    assert api.get_latest_release_for_channel("stable")["tag_name"] == "v1.1.0"
    assert api.get_latest_release_for_channel("preview")["tag_name"] == "v1.2.0-preview.1"


def test_updater_reports_both_sources_down(monkeypatch):
    _serve(monkeypatch, api=_RATE_LIMITED, feed=_Response(500))

    with pytest.raises(github_api.GitHubAPIError):
        github_api.GitHubAPI("o", "r").get_releases()


def test_update_check_reads_the_fallback(monkeypatch):
    from core import update_check

    _serve(monkeypatch, api=_RATE_LIMITED, feed=_Response(200, text=_FEED))

    assert update_check._latest_tag_for_channel("preview") == "v1.2.0-preview.1"
    assert update_check._latest_tag_for_channel("stable") == "v1.1.0"


def test_update_check_never_caches_a_missing_channel(monkeypatch):
    """The feed holds 10 releases; no stable among them must not erase the known version."""
    from core import update_check

    only_previews = _FEED.split("<entry>\n    <id>tag:github.com,2008:Repository/1/v1.1.0")[0]
    _serve(monkeypatch, api=_RATE_LIMITED, feed=_Response(200, text=only_previews + "</feed>"))

    with pytest.raises(LookupError):
        update_check._latest_tag_for_channel("stable")
