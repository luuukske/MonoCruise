"""Python lock switch interval from ``Settings.gil_switch_ms``. See README.md, "GIL switch interval"."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("main")

MIN_MS = 0.1
MAX_MS = 50.0


def apply_gil_switch(ms) -> bool:
    """Set the interval to ``ms`` milliseconds; an unusable value keeps Python's own and says so."""
    try:
        value = float(ms)
    except (TypeError, ValueError):
        value = float("nan")
    if not MIN_MS <= value <= MAX_MS:
        log.warning("gil_switch_ms %r unusable, keeping %.1f ms", ms, sys.getswitchinterval() * 1000.0)
        return False
    sys.setswitchinterval(value / 1000.0)
    log.info("GIL switch interval %.1f ms", sys.getswitchinterval() * 1000.0)
    return True
