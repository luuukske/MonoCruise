"""TMP trailer to tractor: the SDK rig link against the pairings radar consumers use, over the clip store.

    python tools/tmp_trailer_link_probe.py                 local and contributed clips
    python tools/tmp_trailer_link_probe.py --root <dir>    one directory
    python tools/tmp_trailer_link_probe.py --json          machine-readable

Replays the radar chain on every clip carrying ``tmp_sdk_buf`` (plugin 1.0.3+), so each
TMP record has the SDK body ``tmp_sdk.attach`` paired with it. A trailer record and a
tractor record whose bodies share ``rig`` in one write are the pair ``Vehicle::GetTrailer``
reported. Each linked trailer frame is scored for the production lock (``resolve_tractor``:
SDK link, then geometry), the geometry-only sticky lock, the strict acquire alone, and the
nearest-body pairing AEB used until 2026-10-24 (kept here as the baseline). Parked trailers,
which the old pairing also swapped, are counted separately.
core.settings is redirected at a throwaway directory before anything reads it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_METHODS = ("production", "geom_sticky", "strict", "aeb_legacy")
_CONSUMERS = ("acc_lead", "aeb_swap", "aeb_swap_legacy")
_SPEED_BANDS = ((0.0, 0.5), (0.5, 5.0), (5.0, 15.0), (15.0, 99.0))
# "Ahead" is a rough corridor in ego's frame; a speed error past _ERR_MS counts as material.
_AHEAD_M = 80.0
_AHEAD_LAT_M = 6.0
_ERR_MS = 2.0
# A wrong pick this far off the trailer's axis sits in another lane.
_NEXT_LANE_M = 2.0
# The pre-2026-10-24 AEB pairing, kept as the baseline.
_LEGACY_RADIUS_M = 30.0
_LEGACY_HEADING_DOT = 0.9


def _sandbox_settings() -> None:
    import core.settings as settings

    if getattr(settings, "_probe_sandboxed", False):
        return
    settings._probe_sandboxed = True
    sandbox = Path(tempfile.mkdtemp(prefix="monocruise-probe-"))
    real = settings.CONFIG_PATH
    settings.CONFIG_PATH = sandbox / "config.json"
    settings.BACKUP_PATH = settings.CONFIG_PATH.with_suffix(settings.CONFIG_PATH.suffix + ".bak")
    if real.is_file():
        shutil.copy2(str(real), str(settings.CONFIG_PATH))


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _yaw(v) -> float:
    return v._smooth_yaw if v._smooth_yaw is not None else math.radians(v.rotation.euler()[1])


def _articulation_deg(trailer, tractor) -> float:
    return abs(_wrap_deg(math.degrees(_yaw(tractor) - _yaw(trailer))))


def _local(trailer, other) -> tuple[float, float]:
    """(longi, lat) of ``other`` in the trailer's frame, forward positive."""
    fy = _yaw(trailer)
    dx = other.position.x - trailer.position.x
    dz = other.position.z - trailer.position.z
    return dx * -math.sin(fy) + dz * -math.cos(fy), dx * math.cos(fy) + dz * -math.sin(fy)


def _legacy_nearest(trailer, vehicles):
    """Nearest same-heading non-trailer within 30 m, any kind: AEB's pairing before 2026-10-24."""
    tr_yaw = math.radians(trailer.rotation.euler()[1])
    tr_fx, tr_fz = -math.sin(tr_yaw), -math.cos(tr_yaw)
    best, best_d_sq = None, _LEGACY_RADIUS_M * _LEGACY_RADIUS_M
    for other in vehicles:
        if other.id == trailer.id or other.is_trailer:
            continue
        dx = other.position.x - trailer.position.x
        dz = other.position.z - trailer.position.z
        d_sq = dx * dx + dz * dz
        if d_sq >= best_d_sq:
            continue
        o_yaw = math.radians(other.rotation.euler()[1])
        if tr_fx * -math.sin(o_yaw) + tr_fz * -math.cos(o_yaw) < _LEGACY_HEADING_DOT:
            continue
        best_d_sq, best = d_sq, other
    return best


def _sdk_tractor(trailer, tractors_by_rig: dict):
    sdk = trailer.sdk
    if sdk is None or not sdk.is_trailer or sdk.is_local:
        return None
    return tractors_by_rig.get(sdk.rig)


def _ahead_of_ego(v, ego) -> bool:
    eyaw = ego.rotationX * 2.0 * math.pi
    efx, efz = -math.sin(eyaw), -math.cos(eyaw)
    ex, ez = v.position.x - ego.coordinateX, v.position.z - ego.coordinateZ
    return 0.0 < ex * efx + ez * efz < _AHEAD_M and abs(ex * -efz + ez * efx) < _AHEAD_LAT_M


def _where(trailer, pick) -> str:
    longi, lat = _local(trailer, pick)
    if abs(lat) > _NEXT_LANE_M:
        return "next_lane"
    return "behind" if longi < 0.0 else "ahead_in_line"


def _clip_rows(path: str) -> dict | None:
    """Per-trailer-frame verdicts for one clip, or None without SDK frames."""
    _sandbox_settings()
    from core.acc.trailer_lock import resolve_tractor
    from core.aeb.clip_replay import decode_radar_stream
    from core.aeb.clip_schema import Clip
    from core.aeb.thread import _TRAILER_SWAP_SPEED_THRESHOLD_MS
    from core.radar.tmp_sdk import KIND_TRACTOR

    try:
        raw = gzip.decompress(Path(path).read_bytes())
    except Exception:
        return None
    if b'"tmp_sdk_buf"' not in raw:
        return None
    try:
        clip = Clip.from_json_dict(json.loads(raw))
        veh_by_t, ego_by_t, frame_t, _off = decode_radar_stream(clip, with_elevation=False)
    except Exception as exc:
        return {"path": Path(path).name, "error": repr(exc)}

    counts: Counter = Counter()
    by_band: dict[str, Counter] = defaultdict(Counter)
    artic_none: Counter = Counter()
    speed_err: dict[str, list[float]] = defaultdict(list)
    tracks_wrong: dict[str, set[int]] = defaultdict(set)
    tracks_err_ahead: dict[str, set[int]] = defaultdict(set)
    tracks: set[int] = set()
    link_geom: list[tuple[float, float, float]] = []
    prod_cache: dict[int, int] = {}
    geom_cache: dict[int, int] = {}
    last: list[int] = []
    for t in frame_t:
        vehicles = veh_by_t[t]
        # Paused frames repeat the last vehicle objects; score each radar frame once.
        ids = [id(v) for v in vehicles]
        if ids == last:
            continue
        last = ids
        ego = ego_by_t[t]
        tractors_by_rig = {
            v.sdk.rig: v for v in vehicles
            if v.is_tmp and not v.is_trailer and v.sdk is not None and v.sdk.kind == KIND_TRACTOR
        }
        for tr in vehicles:
            if tr.is_trailer and getattr(tr, "is_parked", False) and abs(tr.speed) < _TRAILER_SWAP_SPEED_THRESHOLD_MS:
                counts["parked_trailer_frames"] += 1
                pick = _legacy_nearest(tr, vehicles)
                if pick is not None and abs(pick.speed) > _ERR_MS:
                    counts["parked_legacy_moving"] += 1
                    if _ahead_of_ego(tr, ego):
                        counts["parked_legacy_moving_ahead"] += 1
                        tracks_err_ahead["parked_legacy:faster"].add(tr.id)
                continue
            if not (tr.is_tmp and tr.is_trailer):
                continue
            counts["trailer_frames"] += 1
            # Sticky locks advance every frame for every TMP trailer, as ACC and AEB run them.
            picks = {
                "production": resolve_tractor(tr, vehicles, prod_cache),
                "geom_sticky": resolve_tractor(tr, vehicles, geom_cache, use_sdk=False),
                "strict": resolve_tractor(tr, vehicles, {}, use_sdk=False),
                "aeb_legacy": _legacy_nearest(tr, vehicles),
            }
            if tr.sdk is None:
                counts["trailer_unmatched"] += 1
                continue
            truth = _sdk_tractor(tr, tractors_by_rig)
            if truth is None:
                counts["tractor_unmatched"] += 1
                for m, pick in picks.items():
                    if pick is not None and pick.sdk is not None and pick.sdk.rig != tr.sdk.rig:
                        counts[f"{m}:contradicts_sdk"] += 1
                continue
            counts["linked"] += 1
            tracks.add(tr.id)
            spd = abs(truth.speed)
            band = next(f"{lo:g}-{hi:g}" for lo, hi in _SPEED_BANDS if lo <= spd < hi)
            art = _articulation_deg(tr, truth)
            link_geom.append((*_local(tr, truth), art))
            swap_fires = abs(tr.speed) < _TRAILER_SWAP_SPEED_THRESHOLD_MS
            if swap_fires:
                counts["aeb_swap_frames"] += 1
            ahead = _ahead_of_ego(tr, ego)
            # Speed each consumer hands this trailer against its rig's: positive reads faster.
            consumers = {"acc_lead": picks["production"]}
            if swap_fires:
                consumers["aeb_swap"] = picks["production"]
                consumers["aeb_swap_legacy"] = picks["aeb_legacy"]
            for c, pick in consumers.items():
                if not ahead:
                    continue
                counts[f"{c}:ahead"] += 1
                err = abs(pick.speed if pick is not None else tr.speed) - abs(truth.speed)
                if err > _ERR_MS:
                    counts[f"{c}:ahead_faster"] += 1
                    tracks_err_ahead[f"{c}:faster"].add(tr.id)
                    if pick is not None and pick.id != truth.id:
                        counts[f"{c}:ahead_faster:{_where(tr, pick)}"] += 1
                elif err < -_ERR_MS:
                    counts[f"{c}:ahead_slower"] += 1
                    tracks_err_ahead[f"{c}:slower"].add(tr.id)
            for m, pick in picks.items():
                if pick is None:
                    verdict = "none"
                    if m == "strict":
                        artic_none[min(int(art // 15) * 15, 90)] += 1
                elif pick.id == truth.id:
                    verdict = "agree"
                else:
                    verdict = "wrong"
                    tracks_wrong[m].add(tr.id)
                    speed_err[m].append(abs(pick.speed - truth.speed))
                    counts[f"{m}:wrong:{_where(tr, pick)}"] += 1
                counts[f"{m}:{verdict}"] += 1
                by_band[band][f"{m}:{verdict}"] += 1
    return {
        "path": Path(path).name,
        "counts": dict(counts),
        "by_band": {k: dict(v) for k, v in by_band.items()},
        "artic_none": dict(artic_none),
        "speed_err": dict(speed_err),
        "tracks": len(tracks),
        "tracks_wrong": {m: len(s) for m, s in tracks_wrong.items()},
        "tracks_err_ahead": {m: len(s) for m, s in tracks_err_ahead.items()},
        "link_geom": link_geom,
    }


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "   n/a"


def _quantiles(xs: list[float], qs=(0.5, 0.9, 0.99)) -> list[float]:
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else [math.nan] * len(qs)


def _where_split(counts: Counter, key: str) -> dict[str, int]:
    return {w: counts[f"{key}:{w}"] for w in ("next_lane", "behind", "ahead_in_line")}


def _report(rows: list[dict], as_json: bool) -> None:
    counts: Counter = Counter()
    by_band: dict[str, Counter] = defaultdict(Counter)
    artic: Counter = Counter()
    speed_err: dict[str, list[float]] = defaultdict(list)
    tracks = 0
    tracks_wrong: Counter = Counter()
    clips_wrong: Counter = Counter()
    tracks_ahead: Counter = Counter()
    clips_ahead: Counter = Counter()
    worst: list[tuple[int, str]] = []
    geom: list[tuple[float, float, float]] = []
    errors = [r for r in rows if "error" in r]
    good = [r for r in rows if "error" not in r]
    for r in good:
        counts.update(r["counts"])
        for b, c in r["by_band"].items():
            by_band[b].update(c)
        artic.update({int(k): v for k, v in r["artic_none"].items()})
        for m, xs in r["speed_err"].items():
            speed_err[m].extend(xs)
        tracks += r["tracks"]
        tracks_wrong.update(r["tracks_wrong"])
        for m, n in r["tracks_wrong"].items():
            if n:
                clips_wrong[m] += 1
        tracks_ahead.update(r["tracks_err_ahead"])
        for m, n in r["tracks_err_ahead"].items():
            if n:
                clips_ahead[m] += 1
        n_bad = r["counts"].get("aeb_swap_legacy:ahead_faster", 0)
        if n_bad:
            worst.append((n_bad, r["path"]))
        geom.extend(r["link_geom"])

    linked = counts["linked"]
    out = {
        "clips_with_sdk": len(good), "clip_errors": len(errors),
        "trailer_frames": counts["trailer_frames"], "linked_frames": linked,
        "trailer_unmatched": counts["trailer_unmatched"], "tractor_unmatched": counts["tractor_unmatched"],
        "linked_tracks": tracks,
        "methods": {
            m: {
                "agree": counts[f"{m}:agree"], "wrong": counts[f"{m}:wrong"], "none": counts[f"{m}:none"],
                "wrong_where": _where_split(counts, f"{m}:wrong"),
                "contradicts_sdk_when_tractor_unmatched": counts[f"{m}:contradicts_sdk"],
                "tracks_wrong": tracks_wrong[m], "clips_wrong": clips_wrong[m],
                "wrong_speed_err_p50_p90_p99": _quantiles(speed_err[m]),
            } for m in _METHODS
        },
        "aeb_swap_frames": counts["aeb_swap_frames"],
        "ahead": {
            c: {
                "frames": counts[f"{c}:ahead"],
                "reads_faster_frames": counts[f"{c}:ahead_faster"],
                "reads_faster_where": _where_split(counts, f"{c}:ahead_faster"),
                "reads_slower_frames": counts[f"{c}:ahead_slower"],
                "reads_faster_tracks": tracks_ahead[f"{c}:faster"], "reads_faster_clips": clips_ahead[f"{c}:faster"],
                "reads_slower_tracks": tracks_ahead[f"{c}:slower"], "reads_slower_clips": clips_ahead[f"{c}:slower"],
            } for c in _CONSUMERS
        },
        "parked": {
            "frames": counts["parked_trailer_frames"],
            "legacy_moving_frames": counts["parked_legacy_moving"],
            "legacy_moving_ahead_frames": counts["parked_legacy_moving_ahead"],
            "legacy_moving_ahead_tracks": tracks_ahead["parked_legacy:faster"],
            "legacy_moving_ahead_clips": clips_ahead["parked_legacy:faster"],
        },
        "worst_legacy_clips": [p for _n, p in sorted(worst, reverse=True)[:10]],
        "by_band": {b: dict(c) for b, c in sorted(by_band.items())},
        "strict_none_by_articulation_deg": dict(sorted(artic.items())),
        "link_longi_p1_p50_p99": _quantiles([g[0] for g in geom], (0.01, 0.5, 0.99)),
        "link_abs_lat_p50_p99": _quantiles([abs(g[1]) for g in geom], (0.5, 0.99)),
        "link_articulation_p50_p90_p99": _quantiles([g[2] for g in geom]),
    }
    if as_json:
        print(json.dumps(out, indent=2))
        return
    print(f"{len(good)} clips with SDK frames ({len(errors)} failed to replay)")
    print(f"TMP trailer record frames {counts['trailer_frames']}: SDK link {linked}"
          f" ({_pct(linked, counts['trailer_frames'])}), trailer unmatched {counts['trailer_unmatched']},"
          f" tractor unmatched {counts['tractor_unmatched']}; {tracks} linked trailer tracks")
    print("SDK link geometry (trailer frame): longi p1/p50/p99 "
          + " / ".join(f"{x:.1f}" for x in out["link_longi_p1_p50_p99"])
          + " m, |lat| p50/p99 " + " / ".join(f"{x:.2f}" for x in out["link_abs_lat_p50_p99"])
          + " m, articulation p50/p90/p99 " + " / ".join(f"{x:.1f}" for x in out["link_articulation_p50_p90_p99"])
          + " deg")
    print("\nper linked frame          agree    wrong     none   wrong tracks  wrong clips"
          "  |dv| wrong p50/p90/p99   contradicts SDK (tractor unmatched)")
    for m in _METHODS:
        s = out["methods"][m]
        e = s["wrong_speed_err_p50_p90_p99"]
        print(f"  {m:<20} {_pct(s['agree'], linked)} {_pct(s['wrong'], linked)} {_pct(s['none'], linked)}"
              f"   {s['tracks_wrong']:>5}/{tracks:<6} {s['clips_wrong']:>5}"
              f"        {e[0]:.2f} / {e[1]:.2f} / {e[2]:.2f} m/s   {s['contradicts_sdk_when_tractor_unmatched']}")
    print("\nwhere wrong picks sit (trailer frame): "
          + "; ".join(f"{m} " + ", ".join(f"{k} {v}" for k, v in out["methods"][m]["wrong_where"].items())
                      for m in _METHODS))
    print(f"\nAEB swap fires (trailer |speed| < 0.5 m/s) on {out['aeb_swap_frames']} linked frames")
    print(f"trailer ahead of ego (0-{_AHEAD_M:g} m, |lat| < {_AHEAD_LAT_M:g} m), speed it is given"
          f" off its rig's by more than {_ERR_MS:g} m/s:")
    for c, s in out["ahead"].items():
        where = ", ".join(f"{k} {v}" for k, v in s["reads_faster_where"].items())
        print(f"  {c:<16} frames {s['frames']:>7}  reads faster {_pct(s['reads_faster_frames'], s['frames'])}"
              f" ({s['reads_faster_tracks']} tracks, {s['reads_faster_clips']} clips; wrong pick {where})"
              f"  reads slower {_pct(s['reads_slower_frames'], s['frames'])}"
              f" ({s['reads_slower_tracks']} tracks, {s['reads_slower_clips']} clips)")
    p = out["parked"]
    print(f"\nparked trailers under 0.5 m/s: {p['frames']} frames; the legacy swap handed one a moving"
          f" (> {_ERR_MS:g} m/s) truck's speed on {p['legacy_moving_frames']},"
          f" ahead of ego {p['legacy_moving_ahead_frames']} ({p['legacy_moving_ahead_tracks']} tracks,"
          f" {p['legacy_moving_ahead_clips']} clips). Production never swaps them.")
    print("\nby tractor speed (m/s)    " + "   ".join(f"{m:>18}" for m in _METHODS) + "   (wrong / none)")
    for b, c in out["by_band"].items():
        n = c.get("production:agree", 0) + c.get("production:wrong", 0) + c.get("production:none", 0)
        cells = "   ".join(f"{_pct(c.get(f'{m}:wrong', 0), n)} / {_pct(c.get(f'{m}:none', 0), n)}" for m in _METHODS)
        print(f"  {b:<8} n {n:>7}   {cells}")
    print("\nstrict acquire finds nothing, by articulation (deg): "
          + ", ".join(f"{k}+ {v}" for k, v in out["strict_none_by_articulation_deg"].items()))
    if out["worst_legacy_clips"]:
        print("clips where the legacy swap sped up a trailer ahead most often: " + ", ".join(out["worst_legacy_clips"]))
    for r in errors[:5]:
        print(f"replay failed: {r['path']}: {r['error']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _sandbox_settings()
    from core.aeb.clip_store import contributed_clip_root, default_clip_root

    roots = args.root or [default_clip_root(), contributed_clip_root()]
    paths = sorted(str(p) for r in roots if r.exists() for p in r.glob("*.json.gz"))
    if args.limit:
        paths = paths[: args.limit]
    with ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(_clip_rows, paths, chunksize=4) if r is not None]
    _report(rows, args.json)


if __name__ == "__main__":
    main()
