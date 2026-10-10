"""MonoCruise's own builds of upstream game plugins, used until upstream ships the same.

See the "Plugin overrides" section of README.md in this package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .remote import RemoteFile, SdkSourceError, git_blob_sha_of

log = logging.getLogger("sdk")

_BUNDLE_DIR = Path(__file__).resolve().parent / "overrides"


@dataclass(frozen=True)
class PluginOverride:
    """One bundled file that stands in for one upstream file of one game version."""

    version: str
    name: str
    sha: str
    # Upstream git-blob SHAs this file may overwrite. Anything else upstream ships wins.
    replaces: frozenset[str]
    bundle_dir: Path = _BUNDLE_DIR

    @property
    def path(self) -> Path:
        return self.bundle_dir / self.version / self.name

    def remote(self) -> RemoteFile:
        """Listing entry for the installer; no URL, so it can never be downloaded."""
        return RemoteFile(self.name, self.sha, 0, "")

    def verified_path(self) -> Path:
        """The bundled file, only if it still hashes to ``sha``."""
        if git_blob_sha_of(self.path) != self.sha:
            raise SdkSourceError(f"the bundled {self.name} for {self.version} is missing or damaged")
        return self.path


OVERRIDES: tuple[PluginOverride, ...] = ()

# Builds of upstream files we once shipped as overrides. An install still carrying one is
# stale even offline, so the stock file comes back.
RETIRED: dict[tuple[str, str], frozenset[str]] = {
    # The NCZ build of ets2la_plugin.dll, superseded by tmp_plugin/ (monocruise_tmp.dll).
    ("1.61", "ets2la_plugin.dll"): frozenset({"607740aacb072faf2d0e497a4abee00930a95b1c"}),
}


def override_for(version: str, name: str) -> PluginOverride | None:
    for override in OVERRIDES:
        if override.version == version and override.name == name:
            return override
    return None


def overlay(version: str, listing: dict[str, RemoteFile]) -> dict[str, RemoteFile]:
    """Swap in an override where the source still serves a build it replaces."""
    out = dict(listing)
    for override in OVERRIDES:
        if override.version != version:
            continue
        upstream = listing.get(override.name)
        if upstream is not None and upstream.sha in override.replaces:
            out[override.name] = override.remote()
    return out


def replaces_installed(version: str, name: str, installed: Path) -> bool:
    """True when ``installed`` is a build an override replaces, or one of our retired overrides."""
    override = override_for(version, name)
    retired = RETIRED.get((version, name), frozenset())
    if override is None and not retired:
        return False
    sha = git_blob_sha_of(installed)
    return sha in retired or (override is not None and sha in override.replaces)
