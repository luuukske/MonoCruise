"""
Settings — plain typed dataclass, loaded once in main.py.

No thread reads the config file itself; all threads receive the same
`settings` instance (read-only) via constructor injection or a module-level
reference imported here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib     import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


@dataclass
class Settings:
    debug:          bool  = True
    log_level:      str   = "INFO"

    # Watchdog
    auto_restart:   bool  = True

    # Example worker settings
    telemetry_port: int   = 9341
    loop_interval:  float = 0.05

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        if not path.exists():
            return cls()
        with path.open() as fh:
            data = json.load(fh)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
