"""Local game map for clip replays: the map worker's snapshot, rebuilt from a clip's poses.

Clips never carry map data. A replay that wants the map reads it from this
machine's own game install, matched to the clip by where its ego drives. A
TruckersMP clip also gets the TruckersMP map additions the launcher lists today,
as the live worker does; a clip from before those mods were current can differ.

    from tools.map_replay import install
    install()   # core.aeb.clip_eval.run_headless now replays with the map
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MAPS: dict = {}
_MATCH_SHARE = 0.6
_MATCH_OFFSET_M = 3.0
_BOX_PAD_M = 300.0
_REFRESH_S = 0.05


def game_map(game: str, tmp: bool = False):
    """The installed map, with the TruckersMP additions layered over it when ``tmp``."""
    if game not in _MAPS:
        from core.road_layout.game_map import GameMap
        try:
            _MAPS[game] = GameMap.open_installed(game)
        except (FileNotFoundError, OSError):
            _MAPS[game] = None
    base = _MAPS[game]
    if not tmp or base is None:
        return base
    key = (game, "tmp")
    if key not in _MAPS:
        from core.road_layout.tmp_mods import TmpOverlayOpener

        opener = TmpOverlayOpener(game)
        while not opener.step():
            pass
        _MAPS[key] = base.with_overlay(opener.archives) if opener.archives else base
    return _MAPS[key]


def is_tmp(clip) -> bool:
    return getattr(clip.metadata, "session_kind", "") == "TMP"


def _poses(clip):
    out = []
    for f in sorted(clip.radar_frames, key=lambda fr: fr.t_mono):
        e = f.ego
        if e.paused:
            continue
        yaw = e.rotationX * 2.0 * math.pi
        out.append((e.coordinateX, e.coordinateY, e.coordinateZ, -math.sin(yaw), -math.cos(yaw)))
    return out


def match_clip(clip):
    """``(game, window)`` whose lanes ego drives on for most of the clip, else ``(None, None)``."""
    from core.road_layout.ego_route import EgoRouter

    poses = _poses(clip)
    if len(poses) < 10:
        return None, None
    xs = [p[0] for p in poses]
    zs = [p[2] for p in poses]
    box = (min(xs) - _BOX_PAD_M, max(xs) + _BOX_PAD_M, min(zs) - _BOX_PAD_M, max(zs) + _BOX_PAD_M)
    sample = poses[:: max(1, len(poses) // 30)]
    for game in ("ets2", "ats"):
        gm = game_map(game, is_tmp(clip))
        if gm is None:
            continue
        window = gm.lanes_in_box(*box)
        router = EgoRouter()
        hits = 0
        for x, y, z, fx, fz in sample:
            route = router.step(window, x, y, z, fx, fz)
            if route is not None and abs(route.offset_m) < _MATCH_OFFSET_M:
                hits += 1
        if hits >= _MATCH_SHARE * len(sample):
            return game, window
    return None, None


class ClipMap:
    """``provider(ego, vehicles)`` for ``clip_eval.REPLAY_MAP_FACTORY``: routes per tick.

    Mirrors the map worker: ego's route, plus a route for every moving vehicle within
    120 m and not well behind, each with its own router so lane hysteresis carries over.
    """

    def __init__(self, clip) -> None:
        from core.road_layout.ego_route import EgoRouter
        from core.road_layout.target_path import TargetRouter

        self.game, self.window = match_clip(clip)
        self._router = EgoRouter()
        self._targets = TargetRouter()
        self._last = None
        self._last_t = None

    def __call__(self, ego, vehicles=(), t=None):
        if self.window is None or ego is None or ego.paused:
            return None
        # The live worker refreshes at 20 Hz; replay does the same.
        if t is not None and self._last_t is not None and 0.0 <= t - self._last_t < _REFRESH_S:
            return self._last
        yaw = ego.rotationX * 2.0 * math.pi
        route = self._router.step(
            self.window, ego.coordinateX, ego.coordinateY, ego.coordinateZ,
            -math.sin(yaw), -math.cos(yaw),
        )
        if route is None:
            return None
        targets = {}
        self._targets.prune(v.id for v in vehicles)
        fx, fz = -math.sin(yaw), -math.cos(yaw)
        for v in vehicles:
            if v.is_parked or abs(v.speed) < 2.0:
                continue
            dx, dz = v.position.x - ego.coordinateX, v.position.z - ego.coordinateZ
            if math.hypot(dx, dz) > 120.0 or dx * fx + dz * fz < -10.0:
                continue
            v_yaw = v._smooth_yaw if v._smooth_yaw is not None else math.radians(v.rotation.euler()[1])
            if v.speed < 0.0:
                v_yaw += math.pi
            t_route = self._targets.route(
                v.id, self.window, v.position.x, v.position.y, v.position.z,
                -math.sin(v_yaw), -math.cos(v_yaw),
            )
            if t_route is not None:
                targets[v.id] = t_route
        self._last = SimpleNamespace(status="ready", route=route, window=self.window, target_routes=targets)
        self._last_t = t
        return self._last


def install() -> None:
    import core.aeb.clip_eval as clip_eval

    clip_eval.REPLAY_MAP_FACTORY = ClipMap


def uninstall() -> None:
    import core.aeb.clip_eval as clip_eval

    clip_eval.REPLAY_MAP_FACTORY = None
