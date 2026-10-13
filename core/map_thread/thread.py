"""Map worker: lane geometry around ego from the installed game's own map files.

Opens the archives one per pass, keeps a lane window of about 1.2 km around
ego, follows ego's lane with ``EgoRouter`` and publishes an immutable
``MapSnapshot``. Consumers never parse or open archives. A missing or stale
snapshot means "no map", which is today's behaviour. See core/map_thread/README.md.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.road_layout.ego_route import EgoRouter, LaneRoute
from core.road_layout.game_map import GameMap, GameMapOpener, LaneWindow
from core.road_layout.target_path import TargetRouter
from core.road_layout.tmp_mods import TmpOverlayOpener
from core.thread_management.base_thread import BaseThread, ThreadData
from core.thread_management.registry import registry

logger = logging.getLogger(__name__)

# Window half size, and how close ego may get to its edge before a rebuild.
_WINDOW_HALF_M = 600.0
_WINDOW_MARGIN_M = 300.0
# Work budget per loop pass; one archive or one sector at a time stays under it.
_BUDGET_S = 0.020
_STALE_S = 0.5
# Routes of other vehicles for AEB's road-bent arcs: range, speed floor and time budget.
_TARGET_RANGE_M = 120.0
_TARGET_BEHIND_M = 10.0
_TARGET_MIN_SPEED_MS = 2.0
_TARGET_BUDGET_S = 0.015

# TruckersMP's map additions stay on this long after the last sign of a server.
_TMP_HOLD_S = 60.0

_GAMES = {1: "ets2", 2: "ats"}


@dataclass(frozen=True)
class MapSnapshot:
    """Map state for one ego pose. ``route`` is None off the map or between windows."""

    status: str
    game: str = ""
    window: LaneWindow | None = None
    route: LaneRoute | None = None
    ego_x: float = 0.0
    ego_z: float = 0.0
    t_mono: float = 0.0
    # Vehicle id -> LaneRoute for moving vehicles near ego (AEB road-bent arcs).
    target_routes: dict = field(default_factory=dict)
    # True while TruckersMP's map additions are layered over the base map.
    tmp: bool = False

    def fresh(self, now: float, max_age_s: float = _STALE_S) -> bool:
        return self.status == "ready" and now - self.t_mono <= max_age_s


@dataclass
class MapThreadData(ThreadData):
    snapshot: MapSnapshot = field(default_factory=lambda: MapSnapshot("off"))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class MapThread(BaseThread):
    loop_interval = 0.05
    max_restarts = 3

    def __init__(self) -> None:
        super().__init__(name="map_thread")
        self.data = MapThreadData()
        self._game = ""
        self._opener: GameMapOpener | None = None
        self._map: GameMap | None = None
        self._base_map: GameMap | None = None
        self._overlay_map: GameMap | None = None
        self._overlay_opener: TmpOverlayOpener | None = None
        self._overlay_failed = False
        self._tmp_on = False
        self._tmp_seen = -math.inf
        self._window: LaneWindow | None = None
        self._pending: list[tuple[int, int]] = []
        self._router = EgoRouter()
        self._targets = TargetRouter()
        self._status = ""
        self._open_failed: set[str] = set()

    def setup(self) -> None:
        logger.debug("map thread setup")

    def loop(self) -> None:
        if not self.running:
            return
        if not _enabled():
            self._drop("off")
            return
        game = _running_game()
        pose = _ego_pose()
        if not game or pose is None:
            self._publish(MapSnapshot("waiting"))
            return
        if game != self._game:
            self._switch(game)
        if self._map is None:
            self._open_step()
            return
        self._tmp_step(time.monotonic())
        x, y, z, fwd_x, fwd_z = pose
        if not self._window_step(x, z):
            return
        route = self._router.step(self._window, x, y, z, fwd_x, fwd_z)
        targets = self._target_routes(x, z, fwd_x, fwd_z)
        self._publish(MapSnapshot(
            "ready", self._game, self._window, route, x, z, time.monotonic(), targets, self._tmp_on,
        ))

    def teardown(self) -> None:
        self._drop("off")
        self._map = None
        self._opener = None

    def _switch(self, game: str) -> None:
        self._game = game
        self._map = None
        self._base_map = None
        self._overlay_map = None
        self._overlay_opener = None
        self._overlay_failed = False
        self._tmp_on = False
        self._window = None
        self._pending = []
        self._router.reset()
        self._opener = None
        if game in self._open_failed:
            return
        root = _game_root(game)
        if root is None:
            self._open_failed.add(game)
            self._log_status(f"no {game} install found, map off")
            return
        self._opener = GameMapOpener(root)
        self._log_status(f"opening {game} map")

    def _open_step(self) -> None:
        if self._opener is None:
            self._publish(MapSnapshot("no_map", self._game))
            return
        started = time.monotonic()
        while time.monotonic() - started < _BUDGET_S and self.running:
            if self._opener.step():
                break
        if not self._opener.done:
            self._publish(MapSnapshot("opening", self._game))
            return
        self._map = self._base_map = self._opener.result
        if self._map is None:
            self._open_failed.add(self._game)
            self._log_status(f"{self._game} map unreadable ({self._opener.error}), map off")
        else:
            self._log_status(f"{self._game} map open")
        self._opener = None

    def _tmp_step(self, now: float) -> None:
        """Layer TruckersMP's map additions over the base map while a server is connected."""
        if _tmp_signal():
            self._tmp_seen = now
        want = (
            self._base_map is not None and not self._overlay_failed and _tmp_enabled()
            and now - self._tmp_seen <= _TMP_HOLD_S
        )
        if want == self._tmp_on:
            return
        if want and self._overlay_map is None and not self._build_overlay():
            return
        self._tmp_on = want
        self._map = self._overlay_map if want else self._base_map
        self._window = None
        self._pending = []
        self._router.reset()
        self._targets = TargetRouter()
        self._log_status("TruckersMP map additions on" if want else "TruckersMP map additions off")

    def _build_overlay(self) -> bool:
        """One budget slice of opening TruckersMP's archives. True once the overlay map exists."""
        if self._overlay_opener is None:
            self._overlay_opener = TmpOverlayOpener(self._game)
        started = time.monotonic()
        try:
            while time.monotonic() - started < _BUDGET_S and self.running:
                if self._overlay_opener.step():
                    break
            if not self._overlay_opener.done:
                return False
            archives = self._overlay_opener.archives
            self._overlay_opener = None
            if archives:
                self._overlay_map = self._base_map.with_overlay(archives)
                self._log_status(f"TruckersMP map archives open ({len(archives)})")
                return True
            self._log_status("no TruckersMP map archives found, base map stays")
        except Exception:
            logger.warning("TruckersMP map archives unreadable, base map stays", exc_info=True)
        self._overlay_opener = None
        self._overlay_failed = True
        return False

    def _window_step(self, x: float, z: float) -> bool:
        """Keep the lane window around ego. False while sectors are still loading."""
        win = self._window
        inside = win is not None and (
            win.box[0] + _WINDOW_MARGIN_M <= x <= win.box[1] - _WINDOW_MARGIN_M
            and win.box[2] + _WINDOW_MARGIN_M <= z <= win.box[3] - _WINDOW_MARGIN_M
        )
        if inside:
            return True
        box = (x - _WINDOW_HALF_M, x + _WINDOW_HALF_M, z - _WINDOW_HALF_M, z + _WINDOW_HALF_M)
        if not self._pending:
            self._pending = self._map.sectors_needed(*box)
        started = time.monotonic()
        while self._pending and time.monotonic() - started < _BUDGET_S and self.running:
            self._map.sector(self._pending.pop(0))
        if self._pending:
            if win is None:
                self._publish(MapSnapshot("loading", self._game))
            return win is not None
        self._window = self._map.lanes_in_box(*box)
        logger.debug("map window rebuilt, %d lanes", len(self._window.lanes))
        return True

    def _target_routes(self, ego_x: float, ego_z: float, fwd_x: float, fwd_z: float) -> dict:
        """Lane routes of moving vehicles near and ahead of ego, nearest first, in the budget."""
        vehicles = _radar_vehicles()
        self._targets.prune(v.id for v in vehicles)
        started = time.monotonic()
        out = {}
        for v in sorted(vehicles, key=lambda v: (v.position.x - ego_x) ** 2 + (v.position.z - ego_z) ** 2):
            if math.hypot(v.position.x - ego_x, v.position.z - ego_z) > _TARGET_RANGE_M:
                break
            if time.monotonic() - started > _TARGET_BUDGET_S:
                break
            if v.is_parked or abs(v.speed) < _TARGET_MIN_SPEED_MS:
                continue
            if (v.position.x - ego_x) * fwd_x + (v.position.z - ego_z) * fwd_z < -_TARGET_BEHIND_M:
                continue
            yaw = v._smooth_yaw if v._smooth_yaw is not None else math.radians(v.rotation.euler()[1])
            if v.speed < 0.0:
                yaw += math.pi
            route = self._targets.route(
                v.id, self._window, v.position.x, v.position.y, v.position.z,
                -math.sin(yaw), -math.cos(yaw),
            )
            if route is not None:
                out[v.id] = route
        return out

    def _publish(self, snap: MapSnapshot) -> None:
        self.data.snapshot = snap
        if snap.status != self._status and snap.status in ("ready", "no_map", "off"):
            logger.info("map %s", snap.status)
        self._status = snap.status

    def _drop(self, status: str) -> None:
        if self._status != status:
            self._router.reset()
        self._publish(MapSnapshot(status))

    def _log_status(self, text: str) -> None:
        logger.info(text)


def map_snapshot(max_age_s: float = _STALE_S) -> MapSnapshot | None:
    """Fresh ``ready`` snapshot from the map worker, or None. Safe from any thread."""
    try:
        thread = registry.get_thread("map_thread")
    except KeyError:
        return None
    try:
        snap = thread.data.snapshot
    except AttributeError:
        return None
    if not isinstance(snap, MapSnapshot) or not snap.fresh(time.monotonic(), max_age_s):
        return None
    return snap


def _enabled() -> bool:
    try:
        from core.settings import Settings
        return bool(getattr(Settings(), "map_data_enabled", True))
    except Exception:
        logger.debug("settings unreadable, map stays off")
        return False


def _tmp_enabled() -> bool:
    try:
        from core.settings import Settings
        return bool(getattr(Settings(), "map_tmp_enabled", True))
    except Exception:
        logger.debug("settings unreadable, TruckersMP map additions stay off")
        return False


def _tmp_signal() -> bool:
    """The plugin reports a server connection, or TruckersMP vehicles are in the traffic buffer."""
    try:
        radar = registry.get_thread("radar_thread")
    except KeyError:
        return False
    try:
        with radar.data._lock:
            return bool(radar.data.tmp_connected or radar.data.tmp_session)
    except AttributeError:
        return False


def _radar_vehicles() -> list:
    try:
        radar = registry.get_thread("radar_thread")
    except KeyError:
        return []
    try:
        with radar.data._lock:
            return list(radar.data.vehicles)
    except AttributeError:
        return []


def _running_game() -> str:
    try:
        tel = registry.get_thread("telemetry_thread")
    except KeyError:
        return ""
    try:
        if not tel.is_alive():
            return ""
        with tel.data._lock:
            if not tel.data.is_connected:
                return ""
            raw = int(getattr(tel.data, "game", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return ""
    return _GAMES.get(raw, "")


def _ego_pose() -> tuple[float, float, float, float, float] | None:
    try:
        radar = registry.get_thread("radar_thread")
    except KeyError:
        return None
    try:
        if not radar.is_alive():
            return None
        with radar.data._lock:
            if radar.data.paused:
                return None
            x, y, z = radar.data.ego_x, radar.data.ego_y, radar.data.ego_z
            yaw = radar.data.ego_yaw_rad
    except AttributeError:
        return None
    if x == 0.0 and z == 0.0:
        return None
    return x, y, z, -math.sin(yaw), -math.cos(yaw)


def _game_root(game: str) -> Path | None:
    from core.sdk_installer.game_paths import find_game_installations

    try:
        installs = find_game_installations(game)
    except OSError:
        return None
    return Path(installs[0]) if installs else None
