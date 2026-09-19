"""Read transmission and brake intensity for the selected ETS2/ATS profile.

File layout, key names, and AV constraints: README.md.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from core.sdk_installer.game_paths import GAME_CONFIG

log = logging.getLogger(__name__)

_GAME_FOLDERS = {
    "ets2": "Euro Truck Simulator 2",
    "ats": "American Truck Simulator",
}

_TRANS_BY_G: dict[int, tuple[str, str]] = {
    0: ("arcade", "arcade (simple automatic)"),
    1: ("sequential", "sequential"),
    2: ("h-shifter", "h-shifter"),
    3: ("automatic", "automatic (realistic automatic)"),
}

_TRANS_BY_SHIFTER = {
    "arcade": 0,
    "manual": 1,
    "hshifter": 2,
    "automatic": 3,
}

_USET_RE = re.compile(r'uset\s+(\S+)\s+"([^"]*)"')
_SELECTED_RE = re.compile(r"New profile selected:\s+'([^']*)'")
_HEX_DIR_RE = re.compile(
    r"(?:steam_profiles|/steam/profiles|/profiles)/([0-9A-Fa-f]+)"
)
_HEX_NAME_RE = re.compile(r"^[0-9A-Fa-f]{2,}$")
_CONST_RE = re.compile(
    r'config_lines\[\d+\]:\s+"constant\s+(c_\S+)\s+([-+0-9.eE]+)"'
)
_BRAKE_INTENSITY_MIN = 1.0 / 3.0
_BRAKE_INTENSITY_MAX = 3.0


@dataclass(frozen=True)
class SelectedProfileSettings:
    """Transmission and braking intensity for one selected profile."""

    game: str
    profile_hex: str | None
    profile_name: str | None
    store: str | None
    identity_source: str
    g_trans: int | None
    transmission: str | None
    transmission_ui: str | None
    live_shifter_type: str | None
    g_adaptive_shift: float | None
    adaptive: str | None
    g_brake_intensity: float | None
    brake_intensity_slider_pct: float | None
    c_brake_dz: float | None
    trans_from: str | None
    brake_from: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_selected_profile(
    game: str | None = None,
    *,
    user_dir: Path | None = None,
    steam_remotes: list[Path] | None = None,
    live_shifter_type: str | None | object = ...,
) -> SelectedProfileSettings:
    """Return settings for the profile the game last selected.

    ``user_dir`` / ``steam_remotes`` are test seams. Omit them in production.
    Pass ``live_shifter_type`` to inject telemetry; omit to read SCS SHM.
    """
    resolved_game = game or _detect_game()
    root = user_dir if user_dir is not None else _find_user_dir(resolved_game)
    remotes = (
        steam_remotes
        if steam_remotes is not None
        else _steam_remote_profile_dirs(resolved_game)
    )
    log_text = _read_text(root / "game.log.txt") if root else ""
    hex_id, name, ident_src = _selected_identity(root, log_text)
    store, folder = _profile_folder(root, hex_id) if root and hex_id else (None, None)

    values, origins = _merge_profile_values(folder, remotes, hex_id, log_text)
    if live_shifter_type is ...:
        shifter = _read_live_shifter_type()
    else:
        shifter = live_shifter_type  # type: ignore[assignment]

    g_trans, trans_from = _resolve_g_trans(values, origins, shifter)
    trans, trans_ui = _TRANS_BY_G.get(g_trans, (None, None)) if g_trans is not None else (None, None)

    adaptive_raw = _as_float(values.get("g_adaptive_shift"))
    brake = _as_float(values.get("g_brake_intensity"))
    deadzone = _as_float(values.get("c_brake_dz"))
    return SelectedProfileSettings(
        game=resolved_game,
        profile_hex=hex_id,
        profile_name=name or (decode_profile_hex(hex_id) if hex_id else None),
        store=store,
        identity_source=ident_src,
        g_trans=g_trans,
        transmission=trans,
        transmission_ui=trans_ui,
        live_shifter_type=shifter,
        g_adaptive_shift=adaptive_raw,
        adaptive=_adaptive_label(adaptive_raw),
        g_brake_intensity=brake,
        brake_intensity_slider_pct=_brake_slider_pct(brake),
        c_brake_dz=deadzone,
        trans_from=trans_from,
        brake_from=origins.get("g_brake_intensity"),
    )


def decode_profile_hex(hex_name: str) -> str | None:
    """UTF-8 profile name encoded as the folder's hex id, or None if invalid."""
    if not _HEX_NAME_RE.match(hex_name) or len(hex_name) % 2:
        return None
    try:
        return bytes.fromhex(hex_name).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def parse_uset(text: str) -> dict[str, str]:
    """Parse prism3d ``uset name "value"`` lines."""
    return {m.group(1): m.group(2) for m in _USET_RE.finditer(text)}


def parse_controls_constants(text: str) -> dict[str, str]:
    """Parse ``constant c_*`` entries from a plaintext controls.sii."""
    return {m.group(1): m.group(2) for m in _CONST_RE.finditer(text)}


def _detect_game() -> str:
    shifter_game = _telemetry_game()
    if shifter_game:
        return shifter_game
    newest: tuple[float, str] | None = None
    for game_type in GAME_CONFIG:
        path = _find_user_dir(game_type)
        if path is None:
            continue
        log_path = path / "game.log.txt"
        try:
            mtime = log_path.stat().st_mtime if log_path.is_file() else path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, game_type)
    return newest[1] if newest else "ets2"


def _find_user_dir(game: str) -> Path | None:
    folder = _GAME_FOLDERS.get(game)
    if not folder:
        return None
    candidates: list[Path] = []
    docs = _documents_dir()
    if docs is not None:
        candidates.append(docs / folder)
    home = Path.home()
    candidates.append(home / "Documents" / folder)
    candidates.append(home / "OneDrive" / "Documents" / folder)
    if sys.platform != "win32":
        candidates.append(home / ".local" / "share" / folder)
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_dir():
            existing.append(path)
    if not existing:
        return None
    return max(existing, key=_user_dir_recency)


def _user_dir_recency(path: Path) -> float:
    log_path = path / "game.log.txt"
    try:
        if log_path.is_file():
            return log_path.stat().st_mtime
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _documents_dir() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        folderid = _GUID(
            0xFDD39AD0,
            0x238F,
            0x46AF,
            (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
        )
        get_path = ctypes.windll.shell32.SHGetKnownFolderPath
        get_path.argtypes = [
            ctypes.POINTER(_GUID),
            wintypes.DWORD,
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]
        get_path.restype = ctypes.c_long
        buf = ctypes.c_wchar_p()
        if get_path(ctypes.byref(folderid), 0, None, ctypes.byref(buf)) != 0:
            return None
        value = buf.value
        ctypes.windll.ole32.CoTaskMemFree(buf)
        return Path(value) if value else None
    except (OSError, AttributeError, ValueError):
        log.debug("could not resolve the Documents folder", exc_info=True)
        return None


def _steam_remote_profile_dirs(game: str) -> list[Path]:
    cfg = GAME_CONFIG.get(game)
    if not cfg:
        return []
    appid = cfg["steam_id"]
    found: list[Path] = []
    for userdata in _steam_userdata_roots():
        profiles = userdata / appid / "remote" / "profiles"
        if profiles.is_dir():
            found.append(profiles)
    return found


def _steam_userdata_roots() -> list[Path]:
    roots: list[Path] = []
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"
            )
            install = winreg.QueryValueEx(key, "InstallPath")[0]
            winreg.CloseKey(key)
            roots.append(Path(install) / "userdata")
        except OSError:
            pass
        roots.extend(
            [
                Path(r"C:/Program Files (x86)/Steam/userdata"),
                Path("C:/Program Files/Steam/userdata"),
            ]
        )
    else:
        home = Path.home()
        roots.extend(
            [
                home / ".steam" / "steam" / "userdata",
                home / ".local" / "share" / "Steam" / "userdata",
            ]
        )
    out: list[Path] = []
    seen: set[Path] = set()
    for path in roots:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _selected_identity(
    root: Path | None, log_text: str
) -> tuple[str | None, str | None, str]:
    hex_id = None
    name = None
    if log_text:
        hex_matches = _HEX_DIR_RE.findall(log_text)
        if hex_matches:
            hex_id = hex_matches[-1]
        name_matches = _SELECTED_RE.findall(log_text)
        if name_matches:
            name = name_matches[-1]
        if hex_id:
            decoded = decode_profile_hex(hex_id)
            if name is None:
                name = decoded
            return hex_id, name, "game_log"
    if root is None:
        return None, name, "missing"
    newest = _newest_profile_hex(root)
    if newest:
        return newest, name or decode_profile_hex(newest), "mtime"
    return None, name, "missing"


def _newest_profile_hex(root: Path) -> str | None:
    best: tuple[float, str] | None = None
    for store in ("steam_profiles", "profiles"):
        base = root / store
        if not base.is_dir():
            continue
        try:
            children = list(base.iterdir())
        except OSError:
            log.debug("could not list a profile store")
            continue
        for child in children:
            if not child.is_dir() or not _HEX_NAME_RE.match(child.name):
                continue
            recency = _profile_recency(child)
            if best is None or recency > best[0]:
                best = (recency, child.name)
    return best[1] if best else None


def _profile_folder(root: Path, hex_id: str) -> tuple[str | None, Path | None]:
    matches: list[tuple[float, str, Path]] = []
    for store in ("steam_profiles", "profiles"):
        folder = root / store / hex_id
        if not folder.is_dir():
            # Steam writes mixed-case hex; compare case-insensitively.
            parent = root / store
            if parent.is_dir():
                try:
                    for child in parent.iterdir():
                        if child.is_dir() and child.name.lower() == hex_id.lower():
                            folder = child
                            break
                except OSError:
                    pass
        if folder.is_dir():
            matches.append((_profile_recency(folder), store, folder))
    if not matches:
        return None, None
    matches.sort(reverse=True)
    _, store, folder = matches[0]
    return store, folder


def _profile_recency(folder: Path) -> float:
    newest = 0.0
    try:
        for path in folder.iterdir():
            if not path.is_file():
                continue
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return 0.0
    return newest


def _merge_profile_values(
    folder: Path | None,
    remotes: list[Path],
    hex_id: str | None,
    log_text: str,
) -> tuple[dict[str, str], dict[str, str]]:
    dated: list[tuple[float, str, dict[str, str]]] = []
    if folder is not None:
        dated.extend(_files_from_profile_dir(folder))
    if hex_id:
        for remote in remotes:
            remote_folder = _case_dir(remote, hex_id)
            if remote_folder is not None:
                dated.extend(_files_from_profile_dir(remote_folder, label_prefix="steam:"))
    if log_text:
        dated.append((-1.0, "game.log", parse_uset(log_text)))
    values: dict[str, str] = {}
    origins: dict[str, str] = {}
    dated.sort(key=lambda item: item[0])
    for _mtime, label, parsed in dated:
        for key, val in parsed.items():
            values[key] = val
            origins[key] = label
    return values, origins


def _files_from_profile_dir(
    folder: Path, label_prefix: str = ""
) -> list[tuple[float, str, dict[str, str]]]:
    out: list[tuple[float, str, dict[str, str]]] = []
    for name in ("config.cfg", "config_local.cfg", "controls.sii"):
        path = folder / name
        text = _read_text(path)
        if not text:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        parsed = parse_uset(text)
        if name == "controls.sii":
            parsed.update(parse_controls_constants(text))
        out.append((mtime, f"{label_prefix}{name}", parsed))
    return out


def _case_dir(parent: Path, hex_id: str) -> Path | None:
    direct = parent / hex_id
    if direct.is_dir():
        return direct
    try:
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == hex_id.lower():
                return child
    except OSError:
        log.debug("could not list a Steam remote profile store")
    return None


def _resolve_g_trans(
    values: dict[str, str],
    origins: dict[str, str],
    shifter: str | None,
) -> tuple[int | None, str | None]:
    if shifter:
        mapped = _TRANS_BY_SHIFTER.get(shifter.strip().lower())
        if mapped is not None:
            return mapped, "telemetry"
    raw = values.get("g_trans")
    if raw is None:
        return None, None
    try:
        return int(float(raw)), origins.get("g_trans")
    except ValueError:
        return None, origins.get("g_trans")


def _adaptive_label(value: float | None) -> str | None:
    if value is None:
        return None
    samples = (
        (0.0, "off"),
        (1.6667, "power"),
        (3.0, "normal"),
        (10.0, "eco"),
    )
    for target, label in samples:
        if math.isclose(value, target, rel_tol=0.02, abs_tol=0.05):
            return label
    return "custom"


def brake_ui_scale(cvar: float | None) -> float:
    """UI percent/100 from the geometric cvar. 50/100/150% -> 0.5/1/1.5."""
    if cvar is None or cvar <= 0.0:
        return 1.0
    lo, hi = _BRAKE_INTENSITY_MIN, _BRAKE_INTENSITY_MAX
    clamped = min(max(float(cvar), lo), hi)
    pos = math.log(clamped / lo) / math.log(hi / lo)
    return 0.5 + pos


def _brake_slider_pct(value: float | None) -> float | None:
    """In-game Gameplay slider percent: 50 left, 100 centre, 150 right."""
    if value is None or value <= 0:
        return None
    return 100.0 * brake_ui_scale(value)


def _as_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_text(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        data = path.read_bytes()
    except OSError:
        log.debug("could not read a game settings file")
        return ""
    if data[:4] in (b"ScsC", b"BSII"):
        return ""
    return data.decode("utf-8", errors="replace")


def _read_live_shifter_type() -> str | None:
    try:
        import truck_telemetry

        truck_telemetry.init()
        raw = truck_telemetry.get_data()
    except Exception:
        log.debug("live shifter type unavailable", exc_info=True)
        return None
    if not raw.get("sdkActive"):
        return None
    value = raw.get("shifterType")
    if not value:
        return None
    text = value.decode("ascii", errors="ignore") if isinstance(value, bytes) else str(value)
    text = text.strip("\x00").strip()
    return text or None


def _telemetry_game() -> str | None:
    try:
        import truck_telemetry

        truck_telemetry.init()
        raw = truck_telemetry.get_data()
    except Exception:
        return None
    if not raw.get("sdkActive"):
        return None
    game = raw.get("game")
    if game == 1:
        return "ets2"
    if game == 2:
        return "ats"
    return None
