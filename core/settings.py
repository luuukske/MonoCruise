"""
Settings: plain typed dataclass, loaded once in main.py.

No thread reads the config file itself; all threads receive the same
`settings` instance (read-only) via constructor injection or a module-level
reference imported here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import MISSING, dataclass, field
from pathlib     import Path

from core.thread_management.registry import registry

# Frozen (PyInstaller): __file__ resolves inside the bundle's _internal dir,
# but config.json must live next to MonoCruise.exe in the install root, where
# the standalone updater expects to find it (see updater/updater.py).
if getattr(sys, "frozen", False):
    CONFIG_PATH = Path(sys.executable).resolve().parent / "config.json"
else:
    CONFIG_PATH = Path(__file__).parent.parent / "config.json"
BACKUP_PATH = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")

_log = logging.getLogger("settings")


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
    debug: bool = False
    last_game: int = 1 # 1 = ETS2, 2 = ATS

    # Update channel: "stable" (default) or "preview" (opt into prereleases).
    # The updater reads this from config.json to decide which releases to offer.
    update_channel: str = "stable"

    # Update notifications.
    # notify_for_updates: when True (default) a NOTICE popup fires at boot the
    # moment an update is found (see core/update_check). When False, no popup;
    # the banner + update-button tint still signal a pending update.
    notify_for_updates: bool = True
    # Boot update-check cache (not user knobs): last_update_check gates the 30h
    # network throttle; latest_known_version is the newest release tag seen on
    # the active channel, so the banner/button can show a pending update on
    # throttled boots with no network. Both are written by core/update_check.
    last_update_check: float = 0.0
    latest_known_version: str = ""

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
    weight_adjustment: bool = True
    polling_rate: int = 100

    # OPD
    max_opd_brake_variable: float = 0.04
    opd_mode_variable: int = 0
    offset_variable: float = 0.25

    # Safety & warnings
    hazards_variable: bool = True
    autodisable_hazards: bool = True
    horn_variable: bool = True
    airhorn_variable: bool = True
    autostart_variable: bool = True
    AEB_enabled: bool = False
    # Debug AEB clip capture: grab a screen thumbnail per clip for tagging context.
    aeb_capture_screenshots: bool = True

    # Cruise/ACC/Custom buttons
    cc_dec_button: object = None
    cc_inc_button: object = None
    cc_start_button: object = None
    acc_dist_inc_button: object = None  # raises gap (toward farthest, level 4)
    acc_dist_dec_button: object = None  # lowers gap (toward closest, level 1)
    cc_mode: str = "Cruise control"
    acc_enabled: object = None
    acc_gap_level: int = 2  # 1=closest, 4=farthest. Drives ACC headway and cc_panel lines.
    long_increments: int = 1
    short_increments: int = 5
    long_press_reset: bool = True

    # PID tuning (cruise_control_thread): defaults aligned with config.json
    cc_kp: float = 0.5
    cc_ki: float = 0.0
    cc_kd: float = 0.2
    cc_integral_clamp: float = 3.0
    cc_accel_max_ms2: float = 1.0
    cc_accel_min_ms2: float = -1.0

    # PID tuning for SpeedLimiter: independent of CC so each can be tuned separately.
    limiter_kp: float = 1.3
    limiter_ki: float = 0.0
    limiter_kd: float = 0.2
    limiter_integral_clamp: float = 3.0
    limiter_accel_min_ms2: float = -1.0

    # Global speed limiter target (km/h).
    # Dual role: (1) clamps the CC set-speed so CC never commands above this value;
    # (2) when cc_mode is "Speed limiter", activates SpeedLimiter unconditionally
    # regardless of whether the CC FSM is engaged: the truck is always capped.
    # None disables both effects.
    global_speed_limit_kmh: float | None = None

    # AccelToPedals tuning: split gas (PID) / brake (feedforward + trim PI) architecture.
    # Weight baselines and smoothing constants stay fixed in code.
    mapper_accel_scale_ms2: float = 1.0       # baseline max gas accel for capacity estimate
    mapper_brake_scale_ms2: float = 6.5       # baseline max brake decel for capacity estimate
    mapper_rolling_resistance: float = 0.07   # rolling resistance coefficient for road load

    # Adaptive brake efficiency
    brake_efficiency_learning: bool = True
    brake_efficiency_alpha: float = 0.05
    brake_efficiency_warn_ratio: float = 0.75

    # PedalCapacityTracker: persisted estimates (0 = use baseline on next startup)
    pedal_capacity_max_brake_ms2: float = 11.457
    # Legacy global accel scalar: kept as the cold-start seed source for the
    # shape-function anchor below before any new sample has been taken.
    pedal_capacity_max_accel_ms2: float = 2.124
    # Shape-function anchor gain (m/s² at gas=1.0, mass-normalized, at the
    # anchor gear). One scalar parameterizes the whole per-gear gain curve
    # via G(gear) = anchor * ratio^(anchor_gear - gear). See pedal_capacity.py.
    pedal_capacity_accel_anchor_gain_ms2: float = 0.0
    # Per-gear-step ratio for the shape function (1.0 = flat, >1.0 = lower
    # gear has more gain). Learned online via log-space regression: falls
    # back to a sensible default until enough cross-gear samples accumulate.
    # 0 = use the in-code default on next startup.
    pedal_capacity_accel_ratio_step: float = 0.0

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

    @staticmethod
    def _normalize_channel(value: object) -> str:
        if isinstance(value, str) and value.lower() in {"stable", "preview"}:
            return value.lower()
        _log.warning("invalid update_channel %r; falling back to 'stable'", value)
        return "stable"

    def _public_fields(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if not k.startswith("_")}

    @staticmethod
    def _dataclass_field_default(f) -> object:
        if f.default_factory is not MISSING:
            return f.default_factory()
        if f.default is not MISSING:
            return f.default
        return None

    @classmethod
    def instance(cls) -> "Settings":
        return cls()

    @staticmethod
    def _quarantine_corrupt_file(path: Path, reason: str) -> Path | None:
        """Move a corrupt config out of the way so the user can recover it.

        Returns the quarantine path on success, None on failure. Never deletes
        data: only renames.
        """
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            dest = path.with_suffix(path.suffix + f".corrupt-{ts}")
            shutil.move(str(path), str(dest))
            _log.error("config %s (%s); moved to %s", path.name, reason, dest.name)
            return dest
        except Exception:
            _log.exception("failed to quarantine corrupt config at %s", path)
            return None

    @classmethod
    def _try_read_config(cls, path: Path) -> tuple[dict | None, str | None]:
        """Read and parse a config file. Returns (data, error). data is None on failure."""
        try:
            with path.open() as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return None, "missing"
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON ({exc.msg} line {exc.lineno})"
        except OSError as exc:
            return None, f"read error ({exc})"
        if not isinstance(raw, dict):
            return None, f"top-level JSON is {type(raw).__name__}, expected object"
        return raw, None

    @classmethod
    def load(cls) -> None:
        self = cls.instance()
        with self._state_lock:
            public_keys = [k for k in self.__dataclass_fields__ if not k.startswith("_")]

            data: dict | None = None
            source: str = "defaults"
            file_was_recoverable = False

            if CONFIG_PATH.exists():
                data, err = cls._try_read_config(CONFIG_PATH)
                if data is not None:
                    source = "config.json"
                else:
                    # Primary corrupt: try the backup before quarantining.
                    _log.error("config.json unreadable: %s", err)
                    if BACKUP_PATH.exists():
                        backup_data, backup_err = cls._try_read_config(BACKUP_PATH)
                        if backup_data is not None:
                            data = backup_data
                            source = "config.json.bak"
                            file_was_recoverable = True
                            _log.warning("recovered settings from %s", BACKUP_PATH.name)
                        else:
                            _log.error("config.json.bak also unreadable: %s", backup_err)
                    cls._quarantine_corrupt_file(CONFIG_PATH, err or "unreadable")
            elif BACKUP_PATH.exists():
                # No primary file at all: last-resort recover from backup.
                backup_data, backup_err = cls._try_read_config(BACKUP_PATH)
                if backup_data is not None:
                    data = backup_data
                    source = "config.json.bak (primary missing)"
                    file_was_recoverable = True
                    _log.warning("config.json missing; recovered from %s", BACKUP_PATH.name)
                else:
                    _log.error("config.json missing and backup unreadable: %s", backup_err)

            if data is None:
                data = {}

            missing_from_file = any(k not in data for k in public_keys)

            for k in public_keys:
                if k in data:
                    v = data[k]
                    if k in {"short_increments", "long_increments"}:
                        v = self._normalize_increment(v)
                    elif k == "update_channel":
                        v = self._normalize_channel(v)
                    setattr(self, k, v)
                else:
                    f = self.__dataclass_fields__[k]
                    setattr(self, k, cls._dataclass_field_default(f))

            had_complete_file = (source.startswith("config.json")) and not missing_from_file
            self._saved_state = self._public_fields() if had_complete_file else {}

            _log.info(
                "settings loaded from %s (missing_keys=%s, recovered=%s)",
                source, missing_from_file, file_was_recoverable,
            )

            if not had_complete_file:
                # Persist merged result so the file gains any new fields. Safe by
                # construction: existing keys in `data` were preserved above, so
                # this only ever *adds* defaults for fields the file didn't have.
                cls.save(force=True)

    @classmethod
    def _atomic_write(cls, payload: dict) -> None:
        """Write *payload* to CONFIG_PATH atomically, keeping a .bak of the previous file.

        Sequence:
          1. Write to a temp file in the same directory.
          2. fsync the temp file so the bytes hit the disk.
          3. Copy the existing config.json to config.json.bak (if it exists and parses
             as a non-empty dict): this is our last-known-good snapshot.
          4. os.replace(temp, config.json): atomic on POSIX and on Windows for same-volume moves.

        Any failure aborts before touching the original file.
        """
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Step 1+2: write temp file and fsync.
        fd, tmp_name = tempfile.mkstemp(
            prefix=CONFIG_PATH.name + ".",
            suffix=".tmp",
            dir=str(CONFIG_PATH.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync isn't critical to correctness; ignore on filesystems that don't support it.
                    pass

            # Step 3: snapshot the current file as backup *only if* it is itself a valid dict.
            # Never overwrite a good backup with a corrupt primary.
            if CONFIG_PATH.exists():
                snapshot, err = cls._try_read_config(CONFIG_PATH)
                if snapshot is not None and snapshot:
                    try:
                        shutil.copy2(str(CONFIG_PATH), str(BACKUP_PATH))
                    except OSError:
                        _log.exception("failed to update %s", BACKUP_PATH.name)
                elif err:
                    _log.warning("not refreshing %s: current %s is %s",
                                 BACKUP_PATH.name, CONFIG_PATH.name, err)

            # Step 4: atomic replace.
            os.replace(str(tmp_path), str(CONFIG_PATH))
        except Exception:
            # Clean up temp file if anything went wrong before replace.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

    @classmethod
    def save(cls, values: dict | None = None, *, force: bool = False):
        """Write settings to disk when values have changed since the last load or save.

        If *values* is provided, those fields are applied to the instance before
        the dirty check, allowing a partial update in a single call.

        If *force* is True, always write the full public field set (used after load
        when the config file was missing keys so defaults are persisted).

        Writes are atomic (temp-file + os.replace) and the previous config.json is
        rotated to config.json.bak before each replace, so a crash mid-write or a
        bad payload can never leave the user without a recoverable file.
        """
        self = cls.instance()
        with self._state_lock:
            if values:
                for k, v in values.items():
                    if k in self.__dataclass_fields__ and not k.startswith("_"):
                        setattr(self, k, v)

                    if k == "polling_rate":
                        for thread in registry.get_all_threads():
                            if hasattr(thread, "update_polling_rate"):
                                thread.update_polling_rate()

            current = self._public_fields()
            if not force and current == self._saved_state:
                return

            try:
                cls._atomic_write(current)
            except Exception:
                _log.exception("failed to persist settings; on-disk config left untouched")
                return

            self._saved_state = dict(current)
            return

