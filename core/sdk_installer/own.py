"""Game plugins MonoCruise builds itself and ships inside the app.

See the "MonoCruise's own plugins" section of README.md in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .remote import SdkSourceError, git_blob_sha_of

_BUNDLE_DIR = Path(__file__).resolve().parent / "own"


@dataclass(frozen=True)
class OwnPlugin:
    """One bundled DLL, the same for every game and game version."""

    name: str
    sha: str
    bundle_dir: Path = _BUNDLE_DIR

    @property
    def path(self) -> Path:
        return self.bundle_dir / self.name

    def verified_path(self) -> Path:
        """The bundled file, only if it still hashes to ``sha``."""
        if git_blob_sha_of(self.path) != self.sha:
            raise SdkSourceError(f"the bundled {self.name} is missing or damaged")
        return self.path

    def is_current(self, installed: Path) -> bool:
        return git_blob_sha_of(installed) == self.sha


# Built from tmp_plugin/ (TruckersMP no-collision zone state, core/radar/README.md §18).
OWN_PLUGINS: tuple[OwnPlugin, ...] = (
    OwnPlugin(name="monocruise_tmp.dll", sha="6f4769af30fd5437d5d60f4b561d600bd4314f6c"),
)


def own_plugins() -> tuple[OwnPlugin, ...]:
    return OWN_PLUGINS


def own_plugin(name: str) -> OwnPlugin | None:
    for plugin in OWN_PLUGINS:
        if plugin.name == name:
            return plugin
    return None


def is_own(name: str) -> bool:
    return own_plugin(name) is not None
