"""Read the ETS2LA SDK files straight from their public GitHub repository.

The SDKs are published per game version at
``Assets/SDKs/<version>/Windows`` in https://github.com/ETS2LA/ETS2LA
(fetching them client-side is explicitly permitted by the maintainers and the
repository license).

Two-step, deliberately AV-friendly flow:

  * :meth:`SdkSource.list_files` makes a single JSON request to the GitHub
    contents API and returns each file's git-blob SHA. Comparing that SHA to a
    locally installed file tells us whether an update exists without ever
    downloading a binary.
  * :meth:`SdkSource.download` only runs once we have decided a file is missing
    or stale. It streams the file over HTTPS from ``raw.githubusercontent.com``,
    re-computes the git-blob SHA and refuses to write anything whose SHA does
    not match what the API reported.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger("sdk")

_OWNER = "ETS2LA"
_REPO = "ETS2LA"
_SUBDIR = "Windows"
_REQUEST_TIMEOUT = 20  # seconds
# GitHub requires a User-Agent; a descriptive one also keeps the traffic
# obviously legitimate rather than looking like an anonymous downloader.
_USER_AGENT = "MonoCruise-SDK-Installer"


class SdkSourceError(Exception):
    """The SDK source could not be reached or returned unexpected data."""


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
        """Map of ``filename -> RemoteFile`` for the version folder.

        One GitHub API call, cached for the lifetime of this object. Raises
        :class:`SdkSourceError` on any network/HTTP/parse failure so callers can
        tell "no update needed" apart from "couldn't ask".
        """
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

        if response.status_code != 200:
            raise SdkSourceError(
                f"GitHub returned HTTP {response.status_code} for SDK version {self.version}"
            )

        try:
            entries = response.json()
        except ValueError as exc:
            raise SdkSourceError("GitHub returned an unreadable response") from exc

        if not isinstance(entries, list):
            raise SdkSourceError(f"SDK version {self.version} not found in the repository")

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

    def download(self, remote: RemoteFile, dest: Path) -> None:
        """Download ``remote`` to ``dest``, verifying the git-blob SHA.

        Writes to a temporary sibling file and only moves it into place after
        the SHA matches, so a partial or tampered download never lands at
        ``dest``. Raises :class:`SdkSourceError` on failure.
        """
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
