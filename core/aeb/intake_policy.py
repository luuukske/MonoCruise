"""Server intake policy for clip contribution: kill switch plus guard rails.

Fetched only for users who opted in, cached to Settings, and fail-closed: with no
policy nothing may upload. See README.md in this package.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

POLICY_URL = "https://ld-tech.org/api/v1/aeb_intake.json"

_REQUEST_TIMEOUT = 15          # seconds, matches core/update_check
_DEFAULT_REFRESH_HOURS = 12.0
# A policy document is a few hundred bytes; this only bounds a hostile response.
_MAX_POLICY_BYTES = 64 * 1024


@dataclass(frozen=True)
class IntakePolicy:
    """What the server currently accepts. Defaults refuse everything."""

    accepting: bool = False
    min_client_version: str = ""
    min_schema_version: int = 0
    max_clip_bytes: int = 2 * 1024 * 1024
    refresh_hours: float = _DEFAULT_REFRESH_HOURS
    policy_version: int = 0

    @classmethod
    def from_json(cls, data: dict) -> "IntakePolicy":
        """Parse a policy document, falling back to the refusing defaults per field."""
        base = cls()
        try:
            refresh = float(data.get("refresh_hours", base.refresh_hours))
        except (TypeError, ValueError):
            refresh = base.refresh_hours
        return cls(
            accepting=bool(data.get("accepting", False)),
            min_client_version=str(data.get("min_client_version", "")),
            min_schema_version=_int_or(data.get("min_schema_version"), base.min_schema_version),
            max_clip_bytes=_int_or(data.get("max_clip_bytes"), base.max_clip_bytes),
            refresh_hours=max(0.5, refresh),
            policy_version=_int_or(data.get("policy_version"), base.policy_version),
        )


def _int_or(value: object, fallback: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def contribution_enabled() -> bool:
    """True only when the user opted in under the current consent text."""
    try:
        from core.settings import CONSENT_VERSION, Settings

        return (
            bool(getattr(Settings, "aeb_contribute", False))
            and int(getattr(Settings, "aeb_contribute_consent_version", 0)) >= CONSENT_VERSION
        )
    except Exception:
        logger.debug("could not read the clip contribution opt-in", exc_info=True)
        return False


def cached_policy() -> IntakePolicy | None:
    """Last policy seen, or None when one has never been fetched."""
    try:
        from core.settings import Settings

        raw = str(getattr(Settings, "aeb_intake_policy_json", "") or "")
        if not raw:
            return None
        return IntakePolicy.from_json(json.loads(raw))
    except Exception:
        logger.debug("cached intake policy could not be read", exc_info=True)
        return None


def upload_blocked_reason(
    policy: IntakePolicy | None,
    *,
    clip_bytes: int,
    client_version: str,
    schema_version: int,
) -> str | None:
    """Why this clip may not be uploaded, or None when it may. Fail-closed."""
    if not contribution_enabled():
        return "not opted in"
    if policy is None:
        return "no intake policy"
    if not policy.accepting:
        return "intake closed"
    if schema_version < policy.min_schema_version:
        return "clip schema too old"
    if clip_bytes > policy.max_clip_bytes:
        return "clip too large"
    if policy.min_client_version and _older_than(client_version, policy.min_client_version):
        return "client too old"
    return None


def _older_than(current: str, minimum: str) -> bool:
    """True when *current* is a strictly older version than *minimum*."""
    from packaging.version import InvalidVersion, Version

    def parse(text: str):
        try:
            return Version(text[1:] if text.startswith("v") else text)
        except (InvalidVersion, ValueError, AttributeError):
            return None

    cur, low = parse(current), parse(minimum)
    if cur is None or low is None:
        # An unparseable version must not silently satisfy a floor.
        return True
    return cur < low


def _fetch_policy_text() -> str:
    """GET the policy document. Seam for tests; nothing else should call it."""
    import requests

    resp = requests.get(POLICY_URL, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text[:_MAX_POLICY_BYTES]


def _throttled(now: float) -> bool:
    from core.settings import Settings

    last = float(getattr(Settings, "aeb_intake_checked", 0.0) or 0.0)
    cached = cached_policy()
    hours = cached.refresh_hours if cached is not None else _DEFAULT_REFRESH_HOURS
    return (now - last) < hours * 3600.0


def _run_fetch() -> IntakePolicy | None:
    from core.settings import Settings

    if not contribution_enabled():
        logger.debug("clip contribution is off; intake policy not fetched")
        return None
    now = time.time()
    if _throttled(now):
        return cached_policy()

    text = _fetch_policy_text()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("intake policy is not an object")
    policy = IntakePolicy.from_json(parsed)
    # Stamp only after a successful parse, so a bad response retries next boot.
    Settings.save(values={"aeb_intake_policy_json": text, "aeb_intake_checked": now})
    logger.info(
        "intake policy v%d: accepting=%s", policy.policy_version, policy.accepting
    )
    return policy


def start_policy_fetch() -> threading.Thread | None:
    """Daemon-thread policy refresh. Returns None when the user has not opted in."""
    if not contribution_enabled():
        return None

    def _worker() -> None:
        try:
            _run_fetch()
        except Exception:
            logger.info("intake policy fetch failed; keeping the cached one", exc_info=True)

    thread = threading.Thread(target=_worker, name="aeb_intake_policy", daemon=True)
    thread.start()
    return thread
