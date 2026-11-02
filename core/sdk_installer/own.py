"""Game plugins MonoCruise builds itself and ships inside the app.

See the "MonoCruise's own plugins" section of README.md in this package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.settings import Settings

from .remote import SdkSourceError, git_blob_sha_of

_BUNDLE_DIR = Path(__file__).resolve().parent / "own"


@dataclass(frozen=True)
class OwnPlugin:
    """One bundled DLL, the same for every game and game version.

    ``wanted`` makes it opt-in: an absent copy is installed only while it returns True.
    """

    name: str
    sha: str
    bundle_dir: Path = _BUNDLE_DIR
    label: str = "MonoCruise plugin"
    wanted: Callable[[], bool] | None = field(default=None, compare=False)

    def tracked(self, installed: bool) -> bool:
        """Checked and installed: always, or while wanted, or once installed so it stays current."""
        return self.wanted is None or installed or self.wanted()

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


def _lane_features_on() -> bool:
    """The rumble, the lane keeping assist and LA all act through the FFB plugin."""
    return any(bool(getattr(Settings, key, False)) for key in ("ldw_enabled", "lka_enabled", "la_enabled"))


OWN_PLUGINS: tuple[OwnPlugin, ...] = (
    # Built from tmp_plugin/ (TruckersMP no-collision zone state, core/radar/README.md §18).
    OwnPlugin(name="monocruise_tmp.dll", sha="9f4dbd2477808c65668df4babf7c41506bc9be23", label="TruckersMP plugin"),
    # Built from plugins/ffb/ (lane departure rumble and steering); it patches the wheel's force, so opt-in.
    OwnPlugin(
        name="monocruise_ffb.dll", sha="ff77efccd676abf4247de0d38c81902384e374c2",
        label="Lane departure plugin", wanted=_lane_features_on,
    ),
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
