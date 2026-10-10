"""MonoCruise releases from GitHub, with a fallback past the REST API rate limit.

See shared/README.md, "GitHub releases".
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import quote, unquote

import requests

log = logging.getLogger(__name__)

_API_URL = "https://api.github.com/repos/{owner}/{repo}/releases"
_FEED_URL = "https://github.com/{owner}/{repo}/releases.atom"
_ASSET_URL = "https://github.com/{owner}/{repo}/releases/download/{tag}/{name}"
_ATOM = "{http://www.w3.org/2005/Atom}"
_USER_AGENT = "MonoCruise"


class ReleaseSourceError(Exception):
    """Neither the releases API nor the releases feed produced a release list."""


def is_rate_limited(response) -> bool:
    """True for GitHub's primary or secondary REST API rate limit response."""
    if response.status_code not in (403, 429):
        return False
    headers = response.headers
    if headers.get("X-RateLimit-Remaining") == "0" or "Retry-After" in headers:
        return True
    return "rate limit" in (response.text or "").lower()


def describe_failure(response) -> str:
    """Loggable reason for a failed GitHub response. Never quotes the body: it names the IP."""
    if is_rate_limited(response):
        return "GitHub API rate limit reached for this network address (common on VPNs)"
    return f"GitHub returned HTTP {response.status_code}"


def fetch_releases(owner: str, repo: str, *, timeout: float) -> list[dict]:
    """Published releases, newest first: the API, else the releases feed (newest 10 only)."""
    try:
        return _fetch_api(owner, repo, timeout)
    except ReleaseSourceError as api_error:
        log.info("releases API unavailable (%s); trying the releases feed", api_error)
        try:
            releases = _fetch_feed(owner, repo, timeout)
        except ReleaseSourceError as feed_error:
            raise ReleaseSourceError(f"{api_error}; releases feed: {feed_error}") from feed_error
        log.info("loaded %d releases from the releases feed", len(releases))
        return releases


def _get(url: str, timeout: float, accept: str):
    try:
        return requests.get(
            url, headers={"User-Agent": _USER_AGENT, "Accept": accept}, timeout=timeout
        )
    except requests.RequestException as exc:
        raise ReleaseSourceError(f"cannot reach GitHub: {exc}") from exc


def _fetch_api(owner: str, repo: str, timeout: float) -> list[dict]:
    response = _get(_API_URL.format(owner=owner, repo=repo), timeout, "application/vnd.github+json")
    if response.status_code != 200:
        raise ReleaseSourceError(describe_failure(response))
    try:
        releases = response.json()
    except ValueError as exc:
        raise ReleaseSourceError("GitHub returned an unreadable release list") from exc
    if not isinstance(releases, list):
        raise ReleaseSourceError("GitHub returned an unexpected release list")
    return releases


def _fetch_feed(owner: str, repo: str, timeout: float) -> list[dict]:
    response = _get(_FEED_URL.format(owner=owner, repo=repo), timeout, "application/atom+xml")
    if response.status_code != 200:
        raise ReleaseSourceError(f"HTTP {response.status_code}")
    try:
        return parse_feed(response.text, owner, repo)
    except ET.ParseError as exc:
        raise ReleaseSourceError("unreadable feed") from exc


def parse_feed(xml_text: str, owner: str, repo: str) -> list[dict]:
    """Release dicts shaped like the API's, built from the releases Atom feed."""
    root = ET.fromstring(xml_text)
    releases: list[dict] = []
    for entry in root.iter(f"{_ATOM}entry"):
        tag = _entry_tag(entry)
        if not tag:
            continue
        asset = f"Update-{tag}.zip"
        url = _ASSET_URL.format(owner=owner, repo=repo, tag=quote(tag, safe=""), name=asset)
        releases.append({
            "id": tag,
            "tag_name": tag,
            "name": (entry.findtext(f"{_ATOM}title") or tag).strip(),
            # The feed has no prerelease flag; release.yml publishes every tag with a '-' as one.
            "prerelease": "-" in tag,
            "body": html_to_markdown(entry.findtext(f"{_ATOM}content") or ""),
            "assets": [{"name": asset, "browser_download_url": url}],
        })
    return releases


def _entry_tag(entry) -> str:
    for link in entry.iter(f"{_ATOM}link"):
        href = link.get("href") or ""
        if "/releases/tag/" in href:
            return unquote(href.rsplit("/releases/tag/", 1)[1]).strip("/")
    entry_id = entry.findtext(f"{_ATOM}id") or ""
    return entry_id.rsplit("/", 1)[1] if "/" in entry_id else ""


_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCKS = {"p", "div", "blockquote", "details", "summary", "table"}


class _MarkdownWriter(HTMLParser):
    """The subset of GitHub's rendered release HTML that release notes use, back to markdown."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._end = ""
        self._lists: list[list] = []  # [ordered, next number]
        self._links: list[str | None] = []
        self._pre = 0

    def markdown(self) -> str:
        text = re.sub(r"[ \t]+\n", "\n", "".join(self._parts))
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _emit(self, text: str) -> None:
        if text:
            self._parts.append(text)
            self._end = text[-1]

    def _newline(self) -> None:
        if self._parts and self._end != "\n":
            self._emit("\n")

    def _block(self) -> None:
        if self._parts:
            self._newline()
            self._emit("\n")

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag in _HEADINGS:
            self._block()
            self._emit("#" * _HEADINGS[tag] + " ")
        elif tag in _BLOCKS:
            self._block()
        elif tag == "br":
            self._emit("\n")
        elif tag in ("ul", "ol"):
            if self._lists:
                self._newline()
            else:
                self._block()
            start = attr.get("start") or "1"
            self._lists.append([tag == "ol", int(start) if start.isdigit() else 1])
        elif tag == "li":
            self._newline()
            marker = "- "
            if self._lists and self._lists[-1][0]:
                marker = f"{self._lists[-1][1]}. "
                self._lists[-1][1] += 1
            self._emit("  " * max(len(self._lists) - 1, 0) + marker)
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "pre":
            self._block()
            self._emit("```\n")
            self._pre += 1
        elif tag == "a":
            href = attr.get("href") or None
            self._links.append(href)
            if href:
                self._emit("[")
        elif tag == "img" and attr.get("src"):
            self._emit(f"![{attr.get('alt') or ''}]({attr['src']})")
        elif tag == "video" and attr.get("src"):
            self._block()
            self._emit(attr["src"])
            self._block()
        elif tag == "hr":
            self._block()
            self._emit("---")
            self._block()

    def handle_endtag(self, tag):
        if tag in _HEADINGS or tag in _BLOCKS:
            self._block()
        elif tag in ("ul", "ol"):
            if self._lists:
                self._lists.pop()
            if self._lists:
                self._newline()
            else:
                self._block()
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "pre" and self._pre:
            self._pre -= 1
            self._newline()
            self._emit("```")
            self._block()
        elif tag == "a":
            href = self._links.pop() if self._links else None
            if href:
                self._emit(f"]({href})")

    def handle_data(self, data):
        if self._pre:
            self._emit(data)
            return
        text = re.sub(r"\s+", " ", data)
        if not self._parts or self._end in ("\n", " "):
            text = text.lstrip()
        self._emit(text)


def html_to_markdown(html_text: str) -> str:
    """Markdown for the updater's renderer from a feed entry's rendered HTML."""
    writer = _MarkdownWriter()
    writer.feed(html_text)
    writer.close()
    return writer.markdown()
