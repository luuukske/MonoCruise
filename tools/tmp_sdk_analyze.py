"""What does the TruckersMP SDK's vehicle velocity and sample arrival tell lag detection?

Reads a ``tools/tmp_sdk_probe.py record`` file, matches each SDK body to the
ETS2LA TMP vehicle at the same place, replays the production radar chain at
30 Hz and reports (tmp_plugin/README.md, "Experimental vehicle block"):

- ego check: SDK velocity of the local truck against SCS telemetry speed
- network: sample interval and gap statistics per remote truck
- velocity semantics: does the SDK velocity only change when a sample arrives
- lead: on each speed change, when the SDK speed and the radar speed cross the
  midpoint, against a centred position reference
- lag: ETS2LA position stalls, SDK sample gaps and radar freezes side by side

    python tools/tmp_sdk_analyze.py tools/tmp_sdk_runs/<file>.bin [--csv out.csv]

Stdlib only, never imports ``core.settings``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import struct
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.radar import tmp_sdk  # noqa: E402
from core.radar.reader import _TOTAL_FORMAT, TrafficReader  # noqa: E402
from core.radar.tmp_sdk import KIND_LOCAL_TRACTOR, KIND_TRACTOR, SdkFrame, decode_frame  # noqa: E402
from tools.tmp_sdk_probe import Record, read_records  # noqa: E402

RADAR_HZ = 30.0
GAP_MS = 250.0
STALL_SHOWN_MS = 300.0
FRESH_MS = 150.0
STALL_EPS_M = 0.005
STALL_MIN_S = 0.10
MOVING_MS = 2.0
REF_HALF_SPAN_S = 0.25
EVENT_SLOPE_MS2 = 1.0
EVENT_MIN_DV_MS = 2.5
EVENT_SEARCH_S = (-1.5, 3.0)


@dataclass
class Sample:
    """One matched frame of one TMP truck."""

    t: float
    sdk_t: float
    raw_x: float
    raw_z: float
    sdk_x: float
    sdk_z: float
    v_h: float
    v_long: float
    vx: float
    vz: float
    has_v: bool
    age_ms: float
    updates: int
    sampled: bool
    match_m: float
    off_long: float
    off_lat: float
    radar_speed: float = math.nan
    radar_freeze: bool | None = None
    radar_lag_confirmed: bool | None = None
    radar_stall: bool | None = None
    radar_crash: bool | None = None
    ref_speed: float = math.nan


@dataclass
class Run:
    duration_s: float = 0.0
    records: int = 0
    sdk_records: int = 0
    ego: list[tuple[float, float, float]] = field(default_factory=list)
    tracks: dict[int, list[Sample]] = field(default_factory=lambda: defaultdict(list))
    unmatched_sdk: int = 0
    matched_sdk: int = 0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))]


def _fmt_dist(values: list[float], unit: str = "", scale: float = 1.0) -> str:
    if not values:
        return "n/a"
    v = [x * scale for x in values]
    return (f"p10 {_pct(v, 10):.2f}  p50 {_pct(v, 50):.2f}  p90 {_pct(v, 90):.2f}  "
            f"p99 {_pct(v, 99):.2f}  max {max(v):.2f}{unit}  (n={len(v)})")


def _match(frame: SdkFrame, tmp: list) -> list[tuple[object, object, float]]:
    """(SDK tractor, ETS2LA TMP tractor, distance) with the radar's own matcher."""
    tmp_sdk.attach(tmp, frame)
    return [(v.sdk, v, math.hypot(v.sdk.x - v.position.x, v.sdk.z - v.position.z))
            for v in tmp if v.sdk is not None]


def load(records: Iterable[Record], use_sdk: bool = False) -> Run:
    """Replay the radar chain at 30 Hz; ``use_sdk`` hands it the SDK frames as RadarThread does."""
    run = Run()
    reader = TrafficReader()
    last_radar_t = -math.inf
    was_paused = False
    t_first = None
    for rec in records:
        run.records += 1
        t_first = rec.t_s if t_first is None else t_first
        run.duration_s = rec.t_s - t_first
        if rec.paused or not rec.pose_ok:
            was_paused = True
            continue
        if was_paused:
            reader.request_reanchor()
            was_paused = False
        try:
            raw = struct.unpack(_TOTAL_FORMAT, rec.traffic)
        except struct.error:
            continue
        tmp = [v for v in TrafficReader._build_vehicles_from_raw(raw) if v.is_tmp and not v.is_trailer]

        frame = decode_frame(rec.sdk, rec.t_s) if rec.sdk else None
        radar = {}
        if rec.t_s - last_radar_t >= 0.95 / RADAR_HZ:
            t_kin = rec.sim_time_us / 1e6 if rec.sim_time_us > 0 else rec.t_s
            out = reader.replay_frame(
                rec.traffic, rec.parked, rec.ego_x, rec.ego_y, rec.ego_z, rec.ego_speed, t_kin,
                sdk=frame if use_sdk else None,
            )
            last_radar_t = rec.t_s
            if out is not None:
                radar = {int(v.id): v for v in out[0]}

        if frame is None:
            continue
        run.sdk_records += 1
        for s in frame.bodies:
            if s.kind == KIND_LOCAL_TRACTOR and rec.pose_ok and s.has_velocity:
                run.ego.append((rec.ego_speed, s.speed_long(), math.hypot(s.x - rec.ego_x, s.z - rec.ego_z)))

        matches = _match(frame, tmp)
        run.matched_sdk += len(matches)
        run.unmatched_sdk += sum(1 for s in frame.bodies if s.kind == KIND_TRACTOR) - len(matches)
        for s, v, d in matches:
            fx, fz = s.forward()
            dx, dz = v.position.x - s.x, v.position.z - s.z
            smp = Sample(
                t=rec.t_s, sdk_t=frame.read_s,
                raw_x=v.position.x, raw_z=v.position.z, sdk_x=s.x, sdk_z=s.z,
                v_h=s.speed_h, v_long=s.speed_long(), vx=s.vx, vz=s.vz, has_v=s.has_velocity,
                age_ms=s.age_ms, updates=s.updates, sampled=s.sampled, match_m=d,
                off_long=dx * fx + dz * fz, off_lat=dx * -fz + dz * fx,
            )
            rv = radar.get(int(v.id))
            if rv is not None:
                smp.radar_speed = float(rv.speed)
                smp.radar_freeze = rv._lag_since is not None and not rv.lag_confirmed
                smp.radar_lag_confirmed = bool(rv.lag_confirmed)
                smp.radar_stall = bool(rv.netcode_stall)
                smp.radar_crash = bool(rv.crash_confirmed)
            run.tracks[int(v.id)].append(smp)
    for track in run.tracks.values():
        _fill_reference(track)
    return run


def _fill_reference(track: list[Sample]) -> None:
    """Centred speed from ETS2LA raw positions over +-REF_HALF_SPAN_S: hindsight, not causal."""
    j0 = j1 = 0
    n = len(track)
    for i, s in enumerate(track):
        while j0 < i and track[j0].t < s.t - REF_HALF_SPAN_S:
            j0 += 1
        j1 = max(j1, i)
        while j1 + 1 < n and track[j1 + 1].t <= s.t + REF_HALF_SPAN_S:
            j1 += 1
        a, b = track[j0], track[j1]
        if b.t - a.t >= REF_HALF_SPAN_S:
            s.ref_speed = math.hypot(b.raw_x - a.raw_x, b.raw_z - a.raw_z) / (b.t - a.t)


def _segments(track: list[Sample]) -> list[list[Sample]]:
    """Split a track where it was not seen for over 1 s."""
    out, cur = [], []
    for s in track:
        if cur and s.t - cur[-1].t > 1.0:
            out.append(cur)
            cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def report_ego(run: Run) -> list[str]:
    if not run.ego:
        return ["ego: no local-truck SDK velocity (plugin missing, or not driving on TMP)"]
    moving = [(tel, sdk, d) for tel, sdk, d in run.ego if abs(tel) > 2.0]
    err = [sdk - tel for tel, sdk, _ in moving]
    ratio = [sdk / tel for tel, sdk, _ in moving]
    sign_ok = sum(1 for tel, sdk, _ in moving if (tel > 0) == (sdk > 0))
    return [
        "ego check (SDK local truck vs SCS telemetry, moving frames):",
        f"  speed error m/s   {_fmt_dist(err)}",
        f"  speed ratio       {_fmt_dist(ratio)}",
        f"  sign agrees       {sign_ok}/{len(moving)}",
        f"  position offset m {_fmt_dist([d for _, _, d in run.ego])}",
    ]


def _arrivals(seg: list[Sample]) -> list[float]:
    """Distinct sample arrival times (plugin clock) seen in a segment."""
    out: list[float] = []
    last_updates = None
    for s in seg:
        if not s.sampled:
            continue
        if last_updates is None or s.updates != last_updates:
            out.append(s.sdk_t - s.age_ms / 1000.0)
            last_updates = s.updates
    return out


def report_network(run: Run) -> list[str]:
    intervals, rates, ages = [], [], []
    multi = 0
    pairs = 0
    for track in run.tracks.values():
        for seg in _segments(track):
            arr = _arrivals(seg)
            intervals += [(b - a) * 1000.0 for a, b in zip(arr, arr[1:]) if b > a]
            span = seg[-1].sdk_t - seg[0].sdk_t
            if span > 2.0 and seg[-1].sampled and seg[0].sampled:
                rates.append((seg[-1].updates - seg[0].updates) / span)
            ages += [s.age_ms for s in seg if s.sampled]
            for a, b in zip(seg, seg[1:]):
                if a.sampled and b.sampled:
                    pairs += 1
                    multi += (b.updates - a.updates) > 1
    over = {ms: sum(1 for a in ages if a > ms) / max(len(ages), 1) for ms in (150, 250, 500)}
    return [
        "network samples (remote trucks):",
        f"  updates per s       {_fmt_dist(rates)}",
        f"  sample interval ms  {_fmt_dist(intervals)}",
        f"  sample age ms       {_fmt_dist(ages)}",
        "  age over 150/250/500 ms: " + "  ".join(f"{100 * over[k]:.2f}%" for k in (150, 250, 500)),
        f"  frames with >1 new sample: {multi}/{pairs}",
    ]


def report_velocity(run: Run) -> list[str]:
    held = held_changed = fresh = fresh_changed = 0
    steady_err, pos_vs_sdk = [], []
    for track in run.tracks.values():
        for a, b in zip(track, track[1:]):
            if b.t - a.t > 0.2 or not (a.has_v and b.has_v and a.sampled and b.sampled):
                continue
            changed = math.hypot(b.vx - a.vx, b.vz - a.vz) > 0.01
            if b.updates == a.updates:
                held += 1
                held_changed += changed
            else:
                fresh += 1
                fresh_changed += changed
        for s in track:
            if s.has_v and not math.isnan(s.ref_speed):
                pos_vs_sdk.append(s.v_h - s.ref_speed)
                if s.ref_speed > MOVING_MS:
                    steady_err.append(abs(s.v_h - s.ref_speed) / s.ref_speed)
    lines = [
        "SDK velocity semantics:",
        f"  changed between samples (no new packet): {held_changed}/{held} frame pairs",
        f"  changed on a new packet:                 {fresh_changed}/{fresh} frame pairs",
        "  -> mostly unchanged without a packet = network-carried value; changing every frame = client side",
        f"  SDK speed - position speed m/s  {_fmt_dist(pos_vs_sdk)}",
        f"  relative error while moving     {_fmt_dist(steady_err)}",
    ]
    offs_l = [s.off_long for t in run.tracks.values() for s in t]
    offs_t = [s.off_lat for t in run.tracks.values() for s in t]
    lines += [
        "ETS2LA position minus SDK placement, body frame:",
        f"  longitudinal m  {_fmt_dist(offs_l)}",
        f"  lateral m       {_fmt_dist(offs_t)}",
    ]
    return lines


def _crossing(seg: list[Sample], attr: str, level: float, falling: bool, t0: float, t1: float) -> float:
    """First time ``attr`` crosses ``level`` in [t0, t1], linearly interpolated; nan if never."""
    prev = None
    for s in seg:
        if s.t < t0:
            continue
        if s.t > t1:
            break
        v = getattr(s, attr)
        if math.isnan(v):
            continue
        if prev is not None:
            pv, pt = prev
            crossed = (pv > level >= v) if falling else (pv < level <= v)
            if crossed:
                return pt + (s.t - pt) * (level - pv) / (v - pv) if v != pv else s.t
        prev = (v, s.t)
    return math.nan


def _events(seg: list[Sample]) -> list[tuple[float, float, float, float]]:
    """Speed changes on the reference: (t_start, t_end, v_start, v_end)."""
    pts = [(s.t, s.ref_speed) for s in seg if not math.isnan(s.ref_speed)]
    events = []
    i = 0
    while i + 1 < len(pts):
        sign = 0
        j = i
        while j + 1 < len(pts):
            dt = pts[j + 1][0] - pts[j][0]
            slope = (pts[j + 1][1] - pts[j][1]) / dt if dt > 1e-6 else 0.0
            s_now = 1 if slope > EVENT_SLOPE_MS2 else (-1 if slope < -EVENT_SLOPE_MS2 else 0)
            if s_now == 0 or (sign and s_now != sign):
                break
            sign = s_now
            j += 1
        if j > i and abs(pts[j][1] - pts[i][1]) >= EVENT_MIN_DV_MS:
            events.append((pts[i][0], pts[j][0], pts[i][1], pts[j][1]))
        i = max(j, i + 1)
    return events


def report_lead(run: Run) -> list[str]:
    leads: dict[str, list[float]] = defaultdict(list)
    paired: list[float] = []
    for track in run.tracks.values():
        for seg in _segments(track):
            for t0, t1, v0, v1 in _events(seg):
                level = 0.5 * (v0 + v1)
                falling = v1 < v0
                kind = "dec" if falling else "acc"
                lo, hi = t0 + EVENT_SEARCH_S[0], t1 + EVENT_SEARCH_S[1]
                t_ref = _crossing(seg, "ref_speed", level, falling, lo, hi)
                if math.isnan(t_ref):
                    continue
                t_sdk = _crossing(seg, "v_h", level, falling, lo, hi)
                t_radar = _crossing(seg, "radar_speed", level, falling, lo, hi)
                if not math.isnan(t_sdk):
                    leads[f"sdk {kind}"].append(t_sdk - t_ref)
                if not math.isnan(t_radar):
                    leads[f"radar {kind}"].append(t_radar - t_ref)
                if falling and not (math.isnan(t_sdk) or math.isnan(t_radar)):
                    paired.append(t_radar - t_sdk)
    lines = ["speed-change timing vs centred position reference (s; negative = earlier):"]
    for key in ("sdk dec", "radar dec", "sdk acc", "radar acc"):
        lines.append(f"  {key:9s} {_fmt_dist(leads[key])}")
    lines.append(f"  braking: SDK ahead of radar by (s)  {_fmt_dist(paired)}")
    return lines


def _episodes(seg: list[Sample], flag) -> list[tuple[int, int]]:
    out, start = [], None
    for i, s in enumerate(seg):
        if flag(s, i):
            start = i if start is None else start
        elif start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(seg) - 1))
    return out


def report_lag(run: Run) -> list[str]:
    stalls = []
    gaps = []
    confusion = {"freeze+gap": 0, "freeze+fresh": 0, "gap no freeze": 0, "radar frames": 0}
    stop_freezes = 0
    for track in run.tracks.values():
        for seg in _segments(track):
            def raw_still(s, i, seg=seg):
                return i > 0 and math.hypot(s.raw_x - seg[i - 1].raw_x, s.raw_z - seg[i - 1].raw_z) < STALL_EPS_M
            for a, b in _episodes(seg, raw_still):
                a -= 1
                dur = seg[b].t - seg[a].t
                # The centred reference already sags inside the stall, so read it a window earlier.
                before = [s.ref_speed for s in seg[:a + 1]
                          if seg[a].t - 2 * REF_HALF_SPAN_S <= s.t <= seg[a].t - REF_HALF_SPAN_S
                          and not math.isnan(s.ref_speed)]
                v_before = statistics.median(before) if before else seg[a].v_h
                if dur < STALL_MIN_S or v_before < MOVING_MS:
                    continue
                sdk_move = math.hypot(seg[b].sdk_x - seg[a].sdk_x, seg[b].sdk_z - seg[a].sdk_z)
                stalls.append({
                    "dur": dur,
                    "max_age": max(s.age_ms for s in seg[a:b + 1]),
                    "sdk_move_ratio": sdk_move / max(v_before * dur, 1e-6),
                    "v_sdk_mid": seg[(a + b) // 2].v_h,
                    "v_before": v_before,
                    "radar_froze": any(s.radar_freeze for s in seg[a:b + 1]),
                })
            for a, b in _episodes(seg, lambda s, i: s.sampled and s.age_ms > GAP_MS):
                gaps.append({"dur": seg[b].t - seg[a].t + GAP_MS / 1000.0,
                             "max_age": max(s.age_ms for s in seg[a:b + 1]),
                             "radar_froze": any(s.radar_freeze for s in seg[a:b + 1])})
            for s in seg:
                if s.radar_freeze is None or not s.sampled:
                    continue
                confusion["radar frames"] += 1
                gap = s.age_ms > GAP_MS
                if s.radar_freeze and gap:
                    confusion["freeze+gap"] += 1
                elif s.radar_freeze:
                    confusion["freeze+fresh"] += 1
                    stop_freezes += s.age_ms < FRESH_MS and s.has_v and s.v_h < 1.0
                elif gap:
                    confusion["gap no freeze"] += 1
    lines = [
        f"ETS2LA position stalls (moving > {MOVING_MS} m/s, >= {STALL_MIN_S:.2f} s): {len(stalls)}",
        f"  duration s          {_fmt_dist([s['dur'] for s in stalls])}",
        f"  max sample age ms   {_fmt_dist([s['max_age'] for s in stalls])}",
        f"  with an SDK gap >{GAP_MS:.0f} ms: {sum(s['max_age'] > GAP_MS for s in stalls)}",
        f"  SDK placement kept moving (ratio > 0.5): {sum(s['sdk_move_ratio'] > 0.5 for s in stalls)}",
        f"  SDK speed mid-stall / speed before  {_fmt_dist([s['v_sdk_mid'] / s['v_before'] for s in stalls])}",
        f"  radar froze during stall: {sum(s['radar_froze'] for s in stalls)}",
        f"SDK sample gaps (age > {GAP_MS:.0f} ms): {len(gaps)}",
        f"  duration s          {_fmt_dist([g['dur'] for g in gaps])}",
        f"  radar froze during gap: {sum(g['radar_froze'] for g in gaps)}",
        "radar freeze vs SDK sample age (radar frames of matched trucks):",
        "  " + "  ".join(f"{k}: {v}" for k, v in confusion.items()),
        f"  radar freeze while SDK says stopped (< 1 m/s) with a fresh sample: {stop_freezes}",
    ]
    return lines


def gap_drops(run: Run) -> tuple[list[float], list[float]]:
    """Per gap over STALL_SHOWN_MS on a moving truck: how far radar speed left the sender's own range."""
    drops, over = [], []
    for track in run.tracks.values():
        for seg in _segments(track):
            for a, b in _episodes(seg, lambda s, i: s.sampled and s.age_ms > STALL_SHOWN_MS):
                t0, t1 = seg[a].t, seg[b].t
                win = [x for x in seg if t0 - 0.3 <= x.t <= t1 + 1.5 and not math.isnan(x.radar_speed)]
                if not win or max(x.v_h for x in win) <= 3.0:
                    continue
                # Real braking moves the sender's speed too, so only the excess counts.
                drops.append(min(x.v_h for x in win) - min(x.radar_speed for x in win))
                over.append(max(x.radar_speed for x in win) - max(x.v_h for x in win))
    return drops, over


def _crash_onsets(run: Run) -> int:
    """Rising edges of crash_confirmed over radar frames only (other records carry None)."""
    onsets = 0
    for t in run.tracks.values():
        flags = [s.radar_crash for s in t if s.radar_crash is not None]
        onsets += sum(1 for a, b in zip(flags, flags[1:]) if b and not a)
    return onsets


def report_compare(old: Run, new: Run) -> list[str]:
    lines = ["radar chain today vs with SDK speed (core/radar/README.md section 19):"]
    for name, run in (("today", old), ("SDK  ", new)):
        drops, over = gap_drops(run)
        lines.append(
            f"  {name} gaps {len(drops)}: below sender >2 m/s {sum(d > 2 for d in drops)}, >4 {sum(d > 4 for d in drops)}, "
            f"worst {max(drops, default=math.nan):.1f}; overshoot >2 {sum(o > 2 for o in over)}, "
            f"worst {max(over, default=math.nan):.1f}; crash onsets {_crash_onsets(run)}"
        )
    stall_frames = sum(1 for t in new.tracks.values() for s in t if s.radar_stall)
    lines.append(f"  SDK chain: frames flagged netcode stall {stall_frames}")
    for name, run in (("today", old), ("SDK  ", new)):
        lines += [f"  [{name}] {ln.strip()}" for ln in report_lead(run)[1:]]
    return lines


def write_csv(run: Run, path: Path) -> None:
    cols = [f for f in Sample.__dataclass_fields__]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track", *cols])
        for tid, track in run.tracks.items():
            for s in track:
                w.writerow([tid, *(getattr(s, c) for c in cols)])


def analyze(records: Iterable[Record], compare: bool = True) -> tuple[Run, list[str]]:
    """Report on today's radar chain; ``compare`` also replays it with SDK speed."""
    records = list(records)
    run = load(records)
    head = [
        f"duration {run.duration_s:.0f} s, {run.records} records, {run.sdk_records} with SDK data",
        f"TMP trucks matched: {len(run.tracks)} tracks; SDK trucks matched {run.matched_sdk}, "
        f"unmatched {run.unmatched_sdk}",
    ]
    lines = head + [""] + report_ego(run) + [""] + report_network(run) + [""]
    lines += report_velocity(run) + [""] + report_lead(run) + [""] + report_lag(run)
    if compare:
        lines += [""] + report_compare(run, load(records, use_sdk=True))
    return run, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--csv", type=Path, help="per-sample rows for every matched truck")
    args = ap.parse_args()
    run, lines = analyze(read_records(args.recording))
    print("\n".join(lines))
    if args.csv:
        write_csv(run, args.csv)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
