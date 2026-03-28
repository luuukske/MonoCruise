"""
Settings — plain typed dataclass, loaded once in main.py.

No thread reads the config file itself; all threads receive the same
`settings` instance (read-only) via constructor injection or a module-level
reference imported here.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib     import Path

from core.thread_management.registry import registry

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class _SingletonMeta(type):
    _instances: dict[type, object] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def __getattribute__(cls, name):
        # Forward class-level field access (Settings.some_field) to the singleton instance.
        if not name.startswith("_"):
            try:
                dataclass_fields = super().__getattribute__("__dataclass_fields__")
                if name in dataclass_fields:
                    instance = cls()
                    with instance._state_lock:
                        return getattr(instance, name)
            except AttributeError:
                pass
        return super().__getattribute__(name)


@dataclass
class Settings(metaclass=_SingletonMeta):
    # General settings
    debug: bool = True
    last_game: int = 1 # 1 = ETS2, 2 = ATS

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
    brake_inverted: bool = False
    gas_inverted: bool = False

    # Pedal configuration
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

    # PID tuning (cruise_control_thread)
    cc_kp: float = 0.35
    cc_ki: float = 0.08
    cc_kd: float = 0.15
    cc_integral_clamp: float = 3.0
    cc_accel_max_ms2: float = 1.8
    cc_accel_min_ms2: float = -4.0

    # AccelToPedalMapper tuning
    mapper_reference_mass_kg: float = 20000.0
    mapper_accel_scale_ms2: float = 3.5
    mapper_brake_divisor: float = 7.0
    mapper_brake_power: float = 2.5
    mapper_weight_span_tons: float = 12.7
    mapper_weight_strength: float = 0.27

    # Adaptive brake efficiency
    brake_efficiency_learning: bool = True
    brake_efficiency_alpha: float = 0.05
    brake_efficiency_warn_ratio: float = 0.75

    _saved_state: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    @staticmethod
    def _normalize_increment(value: object) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return 1

    def _public_fields(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if not k.startswith("_")}

    @classmethod
    def instance(cls) -> "Settings":
        return cls()

    @classmethod
    def load(cls) -> None:
        self = cls.instance()
        with self._state_lock:
            self._saved_state = {}
            if not CONFIG_PATH.exists():
                return
            with CONFIG_PATH.open() as fh:
                data = json.load(fh)
            for k, v in data.items():
                if k in self.__dataclass_fields__ and not k.startswith("_"):
                    if k in {"short_increments", "long_increments"}:
                        v = self._normalize_increment(v)
                    setattr(self, k, v)
                    self._saved_state[k] = v

    @classmethod
    def save(cls, values: dict | None = None):
        """Write settings to disk only when values have changed since the last load or save.

        If *values* is provided, those fields are applied to the instance before
        the dirty check, allowing a partial update in a single call.
        """
        self = cls.instance()
        with self._state_lock:
            if values:
                for k, v in values.items():
                    if k in self.__dataclass_fields__ and not k.startswith("_"):
                        setattr(self, k, v)

                    if v == "polling_rate":
                        for thread in registry.get_all_threads():
                            if hasattr(thread, "update_polling_rate"):
                                thread.update_polling_rate()

            current = self._public_fields()
            if current == self._saved_state:
                return
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CONFIG_PATH.open("w") as fh:
                json.dump(current, fh, indent=2, sort_keys=True)
            self._saved_state = dict(current)
            return
