from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    """
    Process-wide configuration loaded once in main.py.

    Threads receive a reference to this dataclass (or specific fields)
    as needed; no worker ever touches the config file directly.
    """

    # Thread loop intervals (seconds)
    telemetry_loop_interval: float = 0.05
    cruise_loop_interval: float = 0.05
    updater_loop_interval: float = 60.0
    ui_loop_interval: float = 0.05
    watchdog_loop_interval: float = 0.2
    monitor_loop_interval: float = 1.0

    # Watchdog settings
    heartbeat_timeout: float = 1.0

    # Behaviour flags
    auto_restart_workers: bool = True

    @staticmethod
    def load(path: str | Path | None = None) -> "Settings":
        """
        Load settings from a JSON file if present, falling back to
        defaults.  Unknown keys are ignored, missing keys use defaults.
        """
        if path is None:
            return Settings()

        p = Path(path)
        if not p.is_file():
            return Settings()

        try:
            raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # On parse errors, fall back to safe defaults.
            return Settings()

        valid_field_names = {f.name for f in fields(Settings)}
        kwargs = {k: v for k, v in raw.items() if k in valid_field_names}
        return Settings(**kwargs)


__all__ = ["Settings"]

