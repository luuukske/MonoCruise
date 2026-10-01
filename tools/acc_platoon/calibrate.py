"""TMP display model against real remote-truck streams from the clip corpus.

Both sides go through the same measurement: raw frame statistics of the drawn
position, then the shipped `Vehicle` filter chain, so the comparison is on what
ACC actually consumes. See tools/acc_platoon/README.md.
"""
from __future__ import annotations

import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from core.radar.traffic import Position, Quaternion, Size, Vehicle

from .netcode import NORMAL, PHYSICS_HZ, SYNC_DELAY_S, NetProfile, TmpStream, TruePath

MIN_STREAM_S: float = 8.0
MIN_SPEED_MS: float = 8.0
MAX_FRAME_GAP_S: float = 0.3
FIT_HALF_S: float = 1.0
STEADY_HALF_S: float = 2.0
STEADY_ACCEL_MS2: float = 0.2
# Artefact response: the window read after an event, and how steady the lead must be around it.
RESPONSE_WINDOW_S: float = 2.0
STEADY_SPEED_TOL_MS: float = 1.5
QUIET_SPACING_S: float = 3.2
# Synthetic ego sits this far behind the drawn truck, at its speed: a typical follow.
SYNTH_EGO_GAP_M: float = 40.0
_FWD = Quaternion(1.0, 0.0, 0.0, 0.0)
_SIZE = Size(2.5, 3.8, 6.0)


@dataclass
class Stream:
    """One remote truck as drawn at each radar frame, projected on its direction of travel."""

    t: list[float]
    s: list[float]
    ego_gap: list[float] = field(default_factory=list)
    ego_speed: list[float] = field(default_factory=list)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))]


def _fit(ts: list[float], ss: list[float], i: int, half: float) -> tuple[float, float, float] | None:
    """Centred least-squares quadratic around frame i: (position, speed, accel)."""
    lo = hi = i
    while lo > 0 and ts[i] - ts[lo - 1] <= half:
        lo -= 1
    while hi < len(ts) - 1 and ts[hi + 1] - ts[i] <= half:
        hi += 1
    if hi - lo < 6:
        return None
    s00 = s01 = s02 = s03 = s04 = r0 = r1 = r2 = 0.0
    for j in range(lo, hi + 1):
        u = ts[j] - ts[i]
        u2 = u * u
        s00 += 1.0
        s01 += u
        s02 += u2
        s03 += u2 * u
        s04 += u2 * u2
        r0 += ss[j]
        r1 += u * ss[j]
        r2 += u2 * ss[j]
    m = [[s00, s01, s02, r0], [s01, s02, s03, r1], [s02, s03, s04, r2]]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(3):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, 4):
                    m[r][k] -= f * m[c][k]
    return m[0][3] / m[0][0], m[1][3] / m[1][1], 2.0 * m[2][3] / m[2][2]


def _along(points: list[tuple[float, float, float]]) -> list[float]:
    ss = [0.0]
    for i in range(1, len(points)):
        j0, j1 = max(0, i - 8), min(len(points) - 1, i + 8)
        ux = points[j1][1] - points[j0][1]
        uz = points[j1][2] - points[j0][2]
        norm = math.hypot(ux, uz)
        if norm < 1e-6:
            ss.append(ss[-1])
            continue
        dx = points[i][1] - points[i - 1][1]
        dz = points[i][2] - points[i - 1][2]
        ss.append(ss[-1] + (dx * ux + dz * uz) / norm)
    return ss


def _clean_runs(rows: list[tuple]) -> list[list[tuple]]:
    runs: list[list[tuple]] = [[rows[0]]]
    for a, b in zip(rows, rows[1:]):
        gap = b[0] - a[0]
        if gap <= 0.0 or gap > MAX_FRAME_GAP_S:
            runs.append([b])
        else:
            runs[-1].append(b)
    return runs


def corpus_streams(root: Path, max_clips: int | None = None, seed: int = 7,
                   workers: int = 1) -> list[Stream]:
    """Moving TMP vehicles from a clip store, one stream per clean run of frames.

    Clips are bursty by session, so a sample misleads; `max_clips=None` reads them all.
    """
    files = sorted(root.glob("*.json.gz"))
    if max_clips is not None:
        files = random.Random(seed).sample(files, min(max_clips, len(files)))
    if workers <= 1:
        return _streams_of(files)
    chunks = [files[k::workers] for k in range(workers)]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return [st for part in pool.map(_streams_of, chunks) for st in part]


def _streams_of(files: list[Path]) -> list[Stream]:
    from core.aeb.clip_store import deserialize_clip
    from core.aeb.clip_timebase import _traffic_kinematics, replay_frames

    out: list[Stream] = []
    for path in files:
        try:
            frames = replay_frames(deserialize_clip(path.read_bytes()))
        except Exception:
            continue
        rows_by_id: dict[int, list[tuple]] = {}
        for f in frames:
            if f.traffic_buf is None or f.ego.paused:
                continue
            kin = _traffic_kinematics(f.traffic_buf)
            for k in kin or ():
                if k.is_tmp:
                    gap = math.hypot(k.position.x - f.ego.coordinateX, k.position.z - f.ego.coordinateZ)
                    rows_by_id.setdefault(k.id, []).append(
                        (f.t_wall, k.position.x, k.position.z, gap, abs(f.ego.speed)))
        for rows in rows_by_id.values():
            for run in _clean_runs(rows):
                if len(run) < 2 or run[-1][0] - run[0][0] < MIN_STREAM_S:
                    continue
                ss = _along(run)
                if (ss[-1] - ss[0]) / (run[-1][0] - run[0][0]) < MIN_SPEED_MS:
                    continue
                out.append(Stream([r[0] for r in run], ss, [r[3] for r in run], [r[4] for r in run]))
    return out


def synthetic_streams(profile: NetProfile = NORMAL, count: int = 60, duration_s: float = 30.0,
                      seed: int = 3, sync_delay_s: float = SYNC_DELAY_S,
                      wobble_ms2: float = 0.0) -> list[Stream]:
    """Trucks cruising, drawn through the display model.

    `wobble_ms2` is a human on a keyboard: throttle and lift in 1 to 4 s bursts.
    """
    from .sim import FRAME_STEPS

    rng = random.Random(seed)
    out: list[Stream] = []
    dt = 1.0 / PHYSICS_HZ
    for _ in range(count):
        v = rng.uniform(30.0, 100.0) / 3.6
        t0 = 1000.0
        path = TruePath(t0, 0.0, v)
        stream = TmpStream(path, profile, profile, random.Random(rng.getrandbits(64)), t0,
                           sync_delay_s)
        s, a, t, switch_at = 0.0, 0.0, t0, t0
        ts: list[float] = []
        xs: list[float] = []
        next_frame = rng.randrange(3)
        for k in range(int(duration_s * PHYSICS_HZ)):
            if t >= switch_at:
                if wobble_ms2 > 0.0:
                    a = -math.copysign(rng.uniform(0.5, 1.0) * wobble_ms2, a or 1.0)
                    switch_at = t + rng.uniform(1.0, 4.0)
                else:
                    a = rng.uniform(-0.15, 0.15)
                    switch_at = t + 4.0
            v = max(5.0, v + a * dt)
            s += v * dt
            t += dt
            path.append(s)
            stream.advance(t, dt)
            if k >= next_frame:
                ts.append(t)
                xs.append(stream.position(t))
                next_frame = k + rng.choice(FRAME_STEPS)
        speeds = [v] * len(ts)
        out.append(Stream(ts, xs, [SYNTH_EGO_GAP_M] * len(ts), speeds))
    return out


def model_streams() -> list[Stream]:
    """The model sample every calibration number in the README and the tests is read from."""
    return synthetic_streams(count=150, duration_s=60.0, seed=5)


def raw_stats(streams: list[Stream]) -> dict[str, float]:
    """Frame-level artefacts of the drawn position, as a flat dict."""
    ratios: list[float] = []
    rms: list[float] = []
    stalls: list[float] = []
    rewinds: list[float] = []
    moving = 0.0
    for st in streams:
        fits = [_fit(st.t, st.s, i, FIT_HALF_S) for i in range(len(st.t))]
        res = [st.s[i] - f[0] for i, f in enumerate(fits) if f is not None]
        if len(res) > 20:
            rms.append(math.sqrt(sum(r * r for r in res) / len(res)))
        stall_from = None
        for i in range(1, len(st.t)):
            f = fits[i]
            if f is None or f[1] < 5.0:
                stall_from = None
                continue
            dt = st.t[i] - st.t[i - 1]
            moving += dt
            ds = st.s[i] - st.s[i - 1]
            if ds == 0.0:
                stall_from = st.t[i - 1] if stall_from is None else stall_from
                continue
            if stall_from is not None:
                stalls.append(st.t[i] - stall_from)
                stall_from = None
            ratios.append(ds / (f[1] * dt))
            if ds < 0.0 and i >= 2 and st.s[i - 1] - st.s[i - 2] >= 0.0:
                rewinds.append(-ds)
    minutes = max(moving / 60.0, 1e-9)
    return {
        "moving_min": moving / 60.0,
        "ratio_p1": _pct(ratios, 1), "ratio_p10": _pct(ratios, 10), "ratio_p90": _pct(ratios, 90),
        "ratio_p99": _pct(ratios, 99),
        "rms_p50": _pct(rms, 50), "rms_p90": _pct(rms, 90),
        "stalls_per_min": len(stalls) / minutes,
        "stall_p50_s": _pct(stalls, 50), "stall_p90_s": _pct(stalls, 90),
        "rewinds_per_min": len(rewinds) / minutes,
    }


def _chain(st: Stream):
    """Yield (frame, previous Vehicle, Vehicle) through the shipped radar chain."""
    prev: Vehicle | None = None
    for i, t in enumerate(st.t):
        v = Vehicle(Position(0.0, 0.0, -st.s[i]), _FWD, _SIZE, 0.0, 0.0, 1, [], 1, True, False)
        if prev is None:
            v.time = t
        else:
            gap = st.ego_gap[i] if st.ego_gap else SYNTH_EGO_GAP_M
            v.update_from_last(prev, t, 0.0, 0.0, -st.s[i] + gap,
                               st.ego_speed[i] if st.ego_speed else 0.0)
        yield i, prev, v
        prev = v


def filter_stats(streams: list[Stream]) -> dict[str, float]:
    """What the shipped `Vehicle` chain makes of the streams on steady stretches."""
    speed_err: list[float] = []
    accel: list[float] = []
    freezes = holds = 0
    steady_s = total_s = 0.0
    for st in streams:
        total_s += st.t[-1] - st.t[0]
        for i, prev, v in _chain(st):
            t = st.t[i]
            if prev is not None:
                freezes += v._lag_since is not None and prev._lag_since is None
                holds += v._pos_mismatch_frames == 1
            if t - st.t[0] < 4.0:
                continue
            f = _fit(st.t, st.s, i, STEADY_HALF_S)
            if f is None or f[1] < MIN_SPEED_MS or abs(f[2]) > STEADY_ACCEL_MS2:
                continue
            steady_s += st.t[i] - st.t[i - 1]
            speed_err.append(abs(v.acc_speed - f[1]))
            accel.append(abs(v.acc_accel))
    minutes = max(total_s / 60.0, 1e-9)
    return {
        "steady_min": steady_s / 60.0,
        "acc_speed_err_p50": _pct(speed_err, 50), "acc_speed_err_p90": _pct(speed_err, 90),
        "acc_speed_err_p99": _pct(speed_err, 99),
        "acc_accel_p50": _pct(accel, 50), "acc_accel_p90": _pct(accel, 90),
        "acc_accel_p99": _pct(accel, 99),
        "freezes_per_min": freezes / minutes, "mismatch_holds_per_min": holds / minutes,
    }


def _mean_speed(st: Stream, t0: float, t1: float) -> float | None:
    idx = [j for j in range(len(st.t)) if t0 <= st.t[j] <= t1]
    if len(idx) < 8:
        return None
    return (st.s[idx[-1]] - st.s[idx[0]]) / (st.t[idx[-1]] - st.t[idx[0]])


def artefact_response(streams: list[Stream]) -> dict[str, float]:
    """Phantom decel `acc_accel` reads within 2 s of a pause, a rewind, or neither, on steady leads."""
    decel: dict[str, list[float]] = {"pause": [], "rewind": [], "quiet": []}
    for st in streams:
        if st.t[-1] - st.t[0] < MIN_STREAM_S:
            continue
        accel = [v.acc_accel for _, _, v in _chain(st)]
        n = len(st.t)
        last_quiet = -math.inf
        for i in range(2, n - 1):
            t = st.t[i]
            if t - st.t[0] < 4.0 or st.t[-1] - t < 3.0:
                continue
            ds, before_ds = st.s[i] - st.s[i - 1], st.s[i - 1] - st.s[i - 2]
            if ds < 0.0 <= before_ds:
                kind = "rewind"
            elif ds == 0.0 < before_ds:
                kind = "pause"
            elif t - last_quiet >= QUIET_SPACING_S:
                kind = "quiet"
            else:
                continue
            before = _mean_speed(st, t - 3.0, t - 0.3)
            after = _mean_speed(st, t + 1.5, t + 3.0)
            if (before is None or after is None or before < MIN_SPEED_MS
                    or abs(before - after) > STEADY_SPEED_TOL_MS):
                continue
            win = [k for k in range(i, n) if st.t[k] - t <= RESPONSE_WINDOW_S]
            if kind == "quiet":
                if any(st.s[k] - st.s[k - 1] <= 0.0 for k in win):
                    continue
                last_quiet = t
            decel[kind].append(-min(accel[k] for k in win))
    out: dict[str, float] = {}
    for kind, xs in decel.items():
        out[f"{kind}_n"] = float(len(xs))
        out[f"{kind}_decel_p50"] = _pct(xs, 50)
        out[f"{kind}_decel_p90"] = _pct(xs, 90)
    return out


def report(corpus: dict[str, float], synthetic: dict[str, float]) -> str:
    lines = [f"{'metric':26s} {'corpus':>9s} {'model':>9s}"]
    for key in corpus:
        lines.append(f"{key:26s} {corpus[key]:9.3f} {synthetic.get(key, float('nan')):9.3f}")
    return "\n".join(lines)
