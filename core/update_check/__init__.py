"""Boot-time update check for MonoCruise.

Mirrors ``core.sdk_installer.start_boot_check``: a one-shot daemon thread runs
the (possible) GitHub round-trip off the boot / Qt main thread, so checking for
an update can NEVER add to startup time. Every failure is logged and swallowed;
the app boots identically whether or not GitHub can be reached.

What it does
------------
Once per boot it asks GitHub for the newest release on the user's update
channel (stable vs preview) and compares it to the running build. The result is
handed to a callback (wired in ``monocruise.py``), and the newest-seen version
is cached in ``Settings`` so the UI can keep signalling a pending update
without another network call.

Two surfaces, driven separately:

* Popup (opt-in): fires only when ``Settings.notify_for_updates`` is True AND
  this was a fresh (non-throttled) check that found a newer build. See the
  callback in ``monocruise.py``. Throttled to once per ``THROTTLE_SECONDS`` so
  relaunches (the background checker can start MonoCruise once per game session)
  don't re-nag.
* Banner + update-button tint (always): derived on the Qt main thread from
  :func:`update_is_pending`, so they reflect a pending update on every boot,
  including throttled ones, with zero network cost.

Interaction with the planned auto-close feature
------------------------------------------------
"Auto-close" is a SEPARATE, deliberately-unbuilt feature: when MonoCruise was
launched by the background checker, it will close itself once the game exits, so
a checker-started session cleans up instead of lingering after play.

That feature must NOT close the app when an update is ready. Keep MonoCruise
open so the update signals (banner / update-button tint / popup) stay visible
and the user can act on them; a checker-started session that auto-closed on game
exit would take the update prompt down with it before the user ever saw it.

The gate for that future feature is :func:`update_is_pending`: skip the
auto-close while it returns True. This module never closes the window itself.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("update_check")

REPO_OWNER = "luuukske"
REPO_NAME = "MonoCruise"

# GitHub's unauthenticated releases API. One GET per fresh check; the 60 req/h
# per-IP limit is ample given the throttle below.
_RELEASES_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
_REQUEST_TIMEOUT = 15  # seconds, matches updater/github_api.py

# Minimum spacing between network checks. The background checker can relaunch
# MonoCruise once per game session (several times a day); without this the popup
# would re-fire every launch. 30h means at most one check/popup per ~day even
# with daily play, while still catching updates promptly.
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
    """Newest release tag visible on *channel*, or "" if none. Raises on network error.

    Channel selection mirrors ``updater.github_api.get_releases_for_channel``:
    'preview' sees only prereleases, 'stable' only non-prereleases, keyed on the
    release's ``prerelease`` flag (CI sets it from the tag). GitHub returns
    releases newest-first, so the first on-channel match wins.
    """
    import requests

    resp = requests.get(_RELEASES_URL, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    want_prerelease = channel == "preview"
    for release in resp.json():
        if bool(release.get("prerelease", False)) == want_prerelease:
            return release.get("tag_name") or ""
    return ""


def update_is_pending() -> bool:
    """True when the cached newest-seen release is newer than the running build.

    Pure and cheap: reads ``Settings`` + ``core.version`` only, no network. Safe
    to call from the Qt main thread on a timer. Drives the banner and the
    update-button tint so a pending update shows on every boot, including
    throttled ones.
    """
    try:
        from core.settings import Settings
        from core.version import __version__

        return _is_newer(Settings.latest_known_version or "", __version__)
    except Exception:
        log.debug("update_is_pending failed", exc_info=True)
        return False


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
        # Persist the throttle stamp + newest-seen tag so throttled boots and the
        # banner/button can answer without a network call. A failed fetch raises
        # above and never reaches here, so the stamp only advances on success.
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
    """Run the update check on a daemon thread and hand the result back.

    Keeps the (possible) network call off the boot / Qt main thread. Any error
    is logged and swallowed so a failed check can never affect startup; on
    failure ``on_result`` is not called (the UI simply shows no pending update).
    """

    def _worker() -> None:
        try:
            _run_check(on_result)
        except Exception:
            log.info("update check failed (offline or GitHub error); skipping", exc_info=True)

    thread = threading.Thread(target=_worker, name="update_check", daemon=True)
    thread.start()
    return thread
