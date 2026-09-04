"""ETS2LA SDK over GitHub (list + SHA-verified download). See README.md AV section."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger("sdk")

_OWNER = "ETS2LA"
_REPO = "ETS2LA"
_SUBDIR = "Windows"
_VERSIONS_PATH = "Assets/SDKs"
_REQUEST_TIMEOUT = 20  # seconds
# GitHub requires a User-Agent; a descriptive one also keeps the traffic
# obviously legitimate rather than looking like an anonymous downloader.
_USER_AGENT = "MonoCruise-SDK-Installer"


class SdkSourceError(Exception):
    """The SDK source could not be reached or returned unexpected data."""


class SdkVersionUnsupported(SdkSourceError):
    """The requested game version has no SDK folder in the repository.

    Either direction: a version too new to be published yet, or an old one
    upstream has since pruned.
    """


def version_key(version: str) -> tuple[int, ...]:
    """Sortable key for a dotted version, or () when it is not numeric."""
    parts: list[int] = []
    for chunk in version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            return ()
    return tuple(parts)


def unsupported_reason(version: str, published: Sequence[str]) -> str:
    """Plain reason a game version has no plugin, in whichever direction."""
    key = version_key(version)
    keys = [k for k in (version_key(p) for p in published) if k]
    if not key or not keys:
        return f"no game plugin is published for game version {version}"
    if key > max(keys):
        return f"game version {version} is newer than any published game plugin"
    if key < min(keys):
        return f"the game plugin for game version {version} is no longer published"
    return f"no game plugin is published for game version {version}"


def git_blob_sha(data: bytes) -> str:
    """git-blob SHA-1 of ``data`` (matches the ``sha`` GitHub reports)."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def git_blob_sha_of(path: Path) -> str | None:
    """git-blob SHA of a file on disk, or None if it cannot be read."""
    try:
        return git_blob_sha(path.read_bytes())
    except OSError:
        return None


@dataclass(frozen=True)
class RemoteFile:
    name: str
    sha: str
    size: int
    download_url: str


class SdkSource:
    """The ``Assets/SDKs/<version>/Windows`` folder for one game version."""

    def __init__(self, version: str, *, owner: str = _OWNER, repo: str = _REPO):
        self.version = version
        self.owner = owner
        self.repo = repo
        self._path = f"Assets/SDKs/{version}/{_SUBDIR}"
        self._listing: dict[str, RemoteFile] | None = None

    @property
    def contents_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{self._path}"

    def _headers(self, accept: str) -> dict[str, str]:
        return {"User-Agent": _USER_AGENT, "Accept": accept}

    def list_files(self, *, refresh: bool = False) -> dict[str, RemoteFile]:
        """Remote file listing (one API call, cached). Raises SdkSourceError on failure."""
        if self._listing is not None and not refresh:
            return self._listing

        try:
            response = requests.get(
                self.contents_url,
                headers=self._headers("application/vnd.github+json"),
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise SdkSourceError(f"cannot reach GitHub: {exc}") from exc

        # A missing version folder is a 404 - distinct from an outage or rate
        # limit, so callers can tell "not supported yet" from "couldn't ask".
        if response.status_code == 404:
            raise SdkVersionUnsupported(
                f"no game plugin is published for game version {self.version}"
            )
        if response.status_code != 200:
            raise SdkSourceError(
                f"GitHub returned HTTP {response.status_code} for SDK version {self.version}"
            )

        try:
            entries = response.json()
        except ValueError as exc:
            raise SdkSourceError("GitHub returned an unreadable response") from exc

        if not isinstance(entries, list):
            raise SdkVersionUnsupported(
                f"no game plugin folder for game version {self.version} in the repository"
            )

        listing: dict[str, RemoteFile] = {}
        for entry in entries:
            if entry.get("type") != "file":
                continue
            name = entry.get("name")
            sha = entry.get("sha")
            url = entry.get("download_url")
            if not (name and sha and url):
                continue
            listing[name] = RemoteFile(name, sha, int(entry.get("size", 0)), url)

        self._listing = listing
        return listing

    def list_versions(self) -> list[str]:
        """Game versions published upstream, oldest first. Empty when unknown.

        Only consulted to explain a version that has no folder, so the happy
        path still costs exactly one API call.
        """
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{_VERSIONS_PATH}"
        try:
            response = requests.get(
                url,
                headers=self._headers("application/vnd.github+json"),
                timeout=_REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                return []
            entries = response.json()
        except (requests.RequestException, ValueError):
            log.debug("could not list the published SDK versions", exc_info=True)
            return []
        if not isinstance(entries, list):
            return []
        names = [
            entry.get("name")
            for entry in entries
            if isinstance(entry, dict) and entry.get("type") == "dir"
        ]
        return sorted((n for n in names if n and version_key(n)), key=version_key)

    def download(self, remote: RemoteFile, dest: Path) -> None:
        """Download to dest via temp file; git-blob SHA must match remote."""
        try:
            response = requests.get(
                remote.download_url,
                headers=self._headers("application/octet-stream"),
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.content
        except requests.RequestException as exc:
            raise SdkSourceError(f"failed to download {remote.name}: {exc}") from exc

        actual = git_blob_sha(data)
        if actual != remote.sha:
            raise SdkSourceError(
                f"integrity check failed for {remote.name} "
                f"(expected {remote.sha[:10]}, got {actual[:10]})"
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise SdkSourceError(f"failed to write {remote.name}: {exc}") from exc

        log.debug("fetched %s (%d bytes) for SDK version %s", remote.name, len(data), self.version)
