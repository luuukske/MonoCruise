"""Boot-time GitHub update check (daemon thread). See README.md in this package."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("update_check")

REPO_OWNER = "luuukske"
REPO_NAME = "MonoCruise"

# One release-list GET per fresh check. A VPN exit address shares the API's 60 req/h
# with everyone on it, so shared.github_releases falls back to the releases feed.
_REQUEST_TIMEOUT = 15  # seconds, matches updater/github_api.py

# Spacing between network checks (background checker may relaunch often).
THROTTLE_SECONDS = 30 * 60 * 60


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of the boot update check handed to the callback."""

    update_available: bool  # a newer build than the running one exists on-channel
    latest_version: str     # newest release tag seen (e.g. "v1.1.0-preview.5"), "" if none
    current_version: str    # the running build
    channel: str            # "stable" | "preview"
    fresh: bool             # True: a network query just ran; False: served from cache (throttled)


def _parse(version_text: str):
    """Parse a version string (a leading 'v' is tolerated), or return None."""
    from packaging.version import InvalidVersion, Version

    if not version_text:
        return None
    try:
        return Version(version_text[1:] if version_text.startswith("v") else version_text)
    except (InvalidVersion, ValueError):
        return None


def _is_newer(candidate_tag: str, current_text: str) -> bool:
    """True when *candidate_tag* is a strictly newer version than *current_text*."""
    cand = _parse(candidate_tag)
    cur = _parse(current_text)
    if cand is None or cur is None:
        return False
    return cand > cur


def _latest_tag_for_channel(channel: str) -> str:
    """Newest on-channel release tag from GitHub (newest-first list). Raises on error."""
    from shared.github_releases import fetch_releases

    want_prerelease = channel == "preview"
    for release in fetch_releases(REPO_OWNER, REPO_NAME, timeout=_REQUEST_TIMEOUT):
        if bool(release.get("prerelease", False)) == want_prerelease and release.get("tag_name"):
            return release["tag_name"]
    # The feed fallback only lists the newest 10; never cache "no release" from it.
    raise LookupError(f"no {channel} release in the list GitHub returned")


def update_is_pending() -> bool:
    """True when cached latest_known_version beats running build (no network)."""
    try:
        from core.settings import Settings
        from core.version import __version__

        return _is_newer(Settings.latest_known_version or "", __version__)
    except Exception:
        log.debug("update_is_pending failed", exc_info=True)
        return False


def popup_throttled() -> bool:
    """True if update popup was shown within THROTTLE_SECONDS (uses last_update_popup)."""
    from core.settings import Settings

    last = float(getattr(Settings, "last_update_popup", 0.0) or 0.0)
    return (time.time() - last) < THROTTLE_SECONDS


def mark_popup_shown() -> None:
    """Record that the update popup was just shown, for :func:`popup_throttled`."""
    from core.settings import Settings

    Settings.save(values={"last_update_popup": time.time()})


def _run_check(on_result: Callable[[UpdateCheckResult], None]) -> None:
    from core.settings import Settings
    from core.version import __version__

    channel = getattr(Settings, "update_channel", "stable") or "stable"
    now = time.time()
    last = float(getattr(Settings, "last_update_check", 0.0) or 0.0)
    throttled = (now - last) < THROTTLE_SECONDS

    if throttled:
        latest = getattr(Settings, "latest_known_version", "") or ""
        fresh = False
        log.info(
            "update check throttled (%.1fh since last); using cached %r",
            (now - last) / 3600.0, latest or "(none)",
        )
    else:
        latest = _latest_tag_for_channel(channel)
        fresh = True
        # Advance throttle stamp only after a successful fetch.
        Settings.save(values={"last_update_check": now, "latest_known_version": latest})
        log.info(
            "update check (%s channel): latest=%r current=%s",
            channel, latest or "(none)", __version__,
        )

    on_result(
        UpdateCheckResult(
            update_available=_is_newer(latest, __version__),
            latest_version=latest,
            current_version=__version__,
            channel=channel,
            fresh=fresh,
        )
    )


def start_update_check(
    on_result: Callable[[UpdateCheckResult], None],
) -> threading.Thread:
    """Daemon thread update check; errors swallowed; on_result skipped on failure."""

    def _worker() -> None:
        try:
            _run_check(on_result)
        except Exception:
            log.info("update check failed (offline or GitHub error); skipping", exc_info=True)

    thread = threading.Thread(target=_worker, name="update_check", daemon=True)
    thread.start()
    return thread
