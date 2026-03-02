"""
Settings — plain typed dataclass, loaded once in main.py.

No thread reads the config file itself; all threads receive the same
`settings` instance (read-only) via constructor injection or a module-level
reference imported here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib     import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


@dataclass
class Settings:
    # General settings
    debug: bool = False

    # UI position/appearance
    panel_x: int = None
    panel_y: int = None
    cc_panel_scaling: float = None
    show_cc_ui: bool = True
    hide_button_action: bool = False
    bar_variable: bool = True

    # User Input
    device: object = None
    gasaxis: int = None
    brakeaxis: int = None

    # Configuration
    gas_exponent_variable: float = None
    brake_exponent_variable: float = None
    weight_adjustment: bool = False
    polling_rate: int = 100 # fps

    # OPD
    max_opd_brake_variable: float = 0.0
    opd_mode_variable: int = 0
    offset_variable: float = 0.0

    # Safety & warnings
    hazards_variable: bool = True
    autodisable_hazards: bool = True
    horn_variable: bool = False
    airhorn_variable: bool = False
    autostart_variable: bool = True
    AEB_enabled: bool = False

    # Cruise/ACC/Custom buttons
    cc_dec_button: object = None
    cc_inc_button: object = None
    cc_start_button: object = None
    cc_mode: str = "ACC"
    acc_enabled: object = None
    long_increments: int = 1
    short_increments: int = 5
    long_press_reset: bool = True

    _saved_state: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _public_fields(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if not k.startswith("_")}

    def load(self) -> None:
        self._saved_state = {}
        if not CONFIG_PATH.exists():
            return
        with CONFIG_PATH.open() as fh:
            data = json.load(fh)
        for k, v in data.items():
            if k in self.__dataclass_fields__ and not k.startswith("_"):
                setattr(self, k, v)
                self._saved_state[k] = v

    def save(self, values: dict | None = None):
        """Write settings to disk only when values have changed since the last load or save.

        If *values* is provided, those fields are applied to the instance before
        the dirty check, allowing a partial update in a single call.
        """
        if values:
            for k, v in values.items():
                if k in self.__dataclass_fields__ and not k.startswith("_"):
                    setattr(self, k, v)
        current = self._public_fields()
        if current == self._saved_state:
            return
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w") as fh:
            json.dump(current, fh, indent=2)
        self._saved_state = dict(current)
