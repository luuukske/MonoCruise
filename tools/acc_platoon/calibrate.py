"""TMP display model against real remote-truck streams from the clip corpus.

Both sides go through the same measurement: drawn-position artefacts binned by the
sender's speed and acceleration, per-session stall rates, the braking overshoot, then
the shipped `Vehicle` filter chain, so the comparison is on what ACC consumes. See
tools/acc_platoon/README.md, "Calibration". numpy is imported where it is used: CI
has none, and only the clip-store tests reach these measurements.
"""
from __future__ import annotations

import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from core.radar.traffic import Position, Quaternion, Size, Vehicle

from .netcode import NORMAL, PHYSICS_HZ, STALL_RATE_BY_SPEED, NetProfile, TmpStream, TruePath
from .plant import HEAVY, LIGHT, MEDIUM, TruckPlant

MIN_STREAM_S: float = 6.0
MAX_FRAME_GAP_S: float = 0.3
MIN_SPEED_MS: float = 8.0
FIT_HALF_S: float = 1.0
STEADY_HALF_S: float = 2.0
STEADY_ACCEL_MS2: float = 0.2
# Sender state for binning: displacement speeds over this window either side of a frame.
STATE_HALF_S: float = 1.0
# A frame is stalled below this share of its expected travel, as the radar's lag test reads it.
STALL_SHARE: float = 0.1
REWIND_MIN_M: float = 0.005
SPEED_BANDS: tuple[tuple[float, float], ...] = (
    (0.5, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 22.0), (22.0, 40.0))
ACCEL_BINS: tuple[tuple[float, float], ...] = (
    (-99.0, -6.0), (-6.0, -3.5), (-3.5, -2.0), (-2.0, -0.8), (-0.8, 0.4), (0.4, 1.0), (1.0, 99.0))
STEADY_BIN: int = 4
# Braking overshoot: corrections in this speed range, by deceleration class, against a far trend.
OVERSHOOT_SPEED: tuple[float, float] = (8.0, 22.0)
OVERSHOOT_DECEL: tuple[tuple[float, float], ...] = ((-3.5, -2.0), (-6.0, -3.5), (-12.0, -6.0))
TREND_FAR_S: tuple[float, float] = (2.0, 3.5)
# Corpus clips cover about this much time; model sessions are cut to match before per-session stats.
CLIP_S: float = 11.0
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
    session: int = 0


def _pct(xs, p: float) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))]


def clip_roots() -> list[Path]:
    """The local store and the contributed one, where present."""
    from core.aeb.clip_store import contributed_clip_root, default_clip_root

    return [r for r in (default_clip_root(), contributed_clip_root()) if r.is_dir()]


def corpus_streams(roots: list[Path] | Path | None = None, max_clips: int | None = None,
                   seed: int = 7, workers: int = 1) -> list[Stream]:
    """Every TMP vehicle in every clip of the stores, any label, one stream per clean run.

    Clips whose replay could not rebuild the physics-step clock are left out: on wall time
    a repeated game frame reads as a stall of every truck, which the live radar never sees.
    """
    if roots is None:
        roots = clip_roots()
    elif isinstance(roots, Path):
        roots = [roots]
    files = sorted(p for r in roots for p in r.glob("*.json.gz"))
    if max_clips is not None:
        files = random.Random(seed).sample(files, min(max_clips, len(files)))
    jobs = list(enumerate(files))
    if workers <= 1:
        parts = [_streams_of(jobs)]
    else:
        chunks = [jobs[k::workers] for k in range(workers)]
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_streams_of, chunks))
    seen: set[str] = set()
    out: list[Stream] = []
    for part in parts:
        for clip_id, streams in part:
            if clip_id not in seen:
                seen.add(clip_id)
                out.extend(streams)
    return out


def step_clock(t_live: list[float]) -> bool:
    """True when frame spacing sits on whole physics steps, the rebuilt replay clock."""
    steps = [(b - a) * PHYSICS_HZ for a, b in zip(t_live, t_live[1:]) if 0.0 < b - a < MAX_FRAME_GAP_S]
    on = sum(1 for x in steps if abs(x - round(x)) < 0.01)
    return len(steps) > 20 and on >= 0.95 * len(steps)


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
        if b[0] - a[0] > MAX_FRAME_GAP_S:
            runs.append([b])
        else:
            runs[-1].append(b)
    return runs


def _streams_of(jobs: list[tuple[int, Path]]) -> list[tuple[str, list[Stream]]]:
    from core.aeb.clip_store import deserialize_clip
    from core.aeb.clip_timebase import _traffic_kinematics, replay_frames

    out: list[tuple[str, list[Stream]]] = []
    for index, path in jobs:
        try:
            clip = deserialize_clip(path.read_bytes())
            frames = replay_frames(clip)
        except Exception:
            continue
        live = [f for f in frames if f.traffic_buf is not None and not f.ego.paused]
        if not step_clock([f.t_wall for f in live]):
            continue
        rows_by_id: dict[int, list[tuple]] = {}
        last_t = None
        for f in live:
            # Zero steps: the radar read the same game frame twice and the reader skips it.
            if last_t is not None and f.t_wall <= last_t:
                continue
            last_t = f.t_wall
            for k in _traffic_kinematics(f.traffic_buf) or ():
                if k.is_tmp:
                    gap = math.hypot(k.position.x - f.ego.coordinateX, k.position.z - f.ego.coordinateZ)
                    rows_by_id.setdefault(k.id, []).append(
                        (f.t_wall, k.position.x, k.position.z, gap, abs(f.ego.speed)))
        streams: list[Stream] = []
        for rows in rows_by_id.values():
            for run in _clean_runs(rows):
                if len(run) < 2 or run[-1][0] - run[0][0] < MIN_STREAM_S:
                    continue
                streams.append(Stream([r[0] for r in run], _along(run), [r[3] for r in run],
                                      [r[4] for r in run], index))
        out.append((clip.metadata.clip_id, streams))
    return out


class TrafficDriver:
    """A sender in mixed traffic: cruises, brakes gently or hard, sometimes to a stop, pulls away."""

    def __init__(self, rng: random.Random, v0: float) -> None:
        self._rng = rng
        self._target = v0
        self._accel = 1.0
        self._decel = 1.0
        self._until = 0.0
        self._wobble = 0.0
        self._wobble_until = 0.0

    def _next_phase(self, t: float) -> None:
        rng = self._rng
        self._decel = min(8.0, max(0.3, math.exp(rng.gauss(math.log(1.5), 0.8))))
        self._accel = rng.uniform(0.4, 1.6)
        roll = rng.random()
        if roll < 0.2:
            self._target = 0.0
            self._until = t + rng.uniform(6.0, 14.0)
        elif roll < 0.5:
            self._target = rng.uniform(78.0, 95.0) / 3.6
            self._until = t + rng.uniform(8.0, 20.0)
        else:
            self._target = rng.uniform(15.0, 78.0) / 3.6
            self._until = t + rng.uniform(4.0, 12.0)

    def command(self, t: float, v: float) -> float:
        if t >= self._until:
            self._next_phase(t)
        if t >= self._wobble_until:
            self._wobble = self._rng.gauss(0.0, 0.15)
            self._wobble_until = t + self._rng.uniform(1.0, 3.0)
        if self._target <= 0.0:
            return -self._decel if v > 0.05 else 0.0
        return max(-self._decel, min(self._accel, 1.2 * (self._target - v))) + self._wobble


def model_streams(profile: NetProfile = NORMAL, sessions: int = 120, senders: int = 6,
                  duration_s: float = 44.0, seed: int = 5, workers: int = 1) -> list[Stream]:
    """Receiver sessions of trucks in mixed traffic, each drawn through the display model.

    Senders run the plant with its gearbox under `TrafficDriver`, so every speed and
    acceleration bin the corpus fills is filled here too.
    """
    jobs = [(profile, senders, duration_s, seed * 1_000_003 + k, k) for k in range(sessions)]
    if workers <= 1:
        return [st for job in jobs for st in _model_session(job)]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return [st for part in pool.map(_model_session, jobs, chunksize=4) for st in part]


def _model_session(job: tuple[NetProfile, int, float, int, int]) -> list[Stream]:
    from .sim import FRAME_STEP_SHARES

    profile, senders, duration_s, seed, index = job
    rng = random.Random(seed)
    dt = 1.0 / PHYSICS_HZ
    steps = [s for s, _ in FRAME_STEP_SHARES]
    weights = [w for _, w in FRAME_STEP_SHARES]
    session = profile.session(rng)
    t0 = 1000.0
    out: list[Stream] = []
    for _ in range(senders):
        spec = rng.choice((HEAVY, MEDIUM, LIGHT))
        v0 = rng.uniform(0.0, 80.0) / 3.6
        plant = TruckPlant(spec, 0.0, v0, dt, random.Random(rng.getrandbits(64)))
        driver = TrafficDriver(random.Random(rng.getrandbits(64)), v0)
        path = TruePath(t0, 0.0, v0)
        stream = TmpStream(path, profile, profile, random.Random(rng.getrandbits(64)), t0,
                           session=session)
        t, next_frame = t0, rng.randrange(3)
        ts: list[float] = []
        xs: list[float] = []
        vs: list[float] = []
        for n in range(int(duration_s * PHYSICS_HZ)):
            plant.step(driver.command(t, plant.v), dt)
            t += dt
            path.append(plant.s, plant.v)
            stream.advance(t, dt)
            if n >= next_frame:
                ts.append(t)
                xs.append(stream.position(t))
                vs.append(plant.v)
                next_frame = n + rng.choices(steps, weights)[0]
        out.append(Stream(ts, xs, [SYNTH_EGO_GAP_M] * len(ts), vs, index))
    return out


def _kin(t, s, half: float = STATE_HALF_S):
    """Sender speed and accel at each frame from displacement over `half` either side."""
    import numpy as np

    n = len(t)
    idx = np.arange(n)
    j0 = np.searchsorted(t, t - half)
    j1 = np.searchsorted(t, t + half, side="right") - 1
    okb = (j0 < idx) & (t - t[j0] >= 0.7 * half)
    oka = (j1 > idx) & (t[j1] - t >= 0.7 * half)
    vb = (s - s[j0]) / np.maximum(t - t[j0], 1e-6)
    va = (s[j1] - s) / np.maximum(t[j1] - t, 1e-6)
    v = np.where(okb & oka, 0.5 * (vb + va), np.nan)
    a = (va - vb) / np.maximum(0.5 * (t[j1] - t[j0]), 1e-6)
    return v, a


def _runs(mask):
    """(start, end) index pairs of True runs, end exclusive."""
    import numpy as np

    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _frames(st: Stream):
    """Per-frame arrays: t, s, ds, dt, speed, accel, stalled, backwards."""
    import numpy as np

    t = np.asarray(st.t, dtype=float)
    s = np.asarray(st.s, dtype=float)
    v, a = _kin(t, s)
    ds = np.diff(s, prepend=s[0])
    dt = np.diff(t, prepend=t[0])
    exp = np.abs(v) * dt
    moving = np.isfinite(v) & (v > SPEED_BANDS[0][0])
    stalled = moving & (np.abs(ds) < STALL_SHARE * exp)
    back = moving & (ds < -np.maximum(REWIND_MIN_M, STALL_SHARE * exp))
    stalled[0] = back[0] = False
    return t, s, ds, dt, v, a, stalled, back


def _bin(v: float, a: float) -> tuple[int, int] | None:
    if not v == v:
        return None
    b = next((k for k, (lo, hi) in enumerate(SPEED_BANDS) if lo <= v < hi), None)
    c = next((k for k, (lo, hi) in enumerate(ACCEL_BINS) if lo <= a < hi), None)
    return None if b is None or c is None else (b, c)


def state_table(streams: list[Stream]) -> dict[tuple[int, int], dict[str, float]]:
    """Stalls and rewinds per minute, binned by the sender's speed and acceleration."""
    acc: dict[tuple[int, int], dict] = {}
    for st in streams:
        if len(st.t) < 30:
            continue
        t, s, ds, dt, v, a, stalled, back = _frames(st)
        for i in range(1, len(t)):
            key = _bin(v[i], a[i])
            if key is not None:
                d = acc.setdefault(key, {"time": 0.0, "stall": [], "rewind": []})
                d["time"] += dt[i]
        for kind, mask in (("stall", stalled), ("rewind", back)):
            for i0, i1 in _runs(mask):
                key = _bin(v[i0], a[i0])
                if key is not None and i1 < len(t):
                    acc.setdefault(key, {"time": 0.0, "stall": [], "rewind": []})[kind].append(
                        (t[i1 - 1] - t[i0 - 1] + 0.5 * dt[i1], i1 - i0))
    out: dict[tuple[int, int], dict[str, float]] = {}
    for key, d in acc.items():
        minutes = d["time"] / 60.0
        if minutes <= 0.0:
            continue
        out[key] = {
            "minutes": minutes,
            "stall_pm": len(d["stall"]) / minutes,
            "rewind_pm": len(d["rewind"]) / minutes,
            "stall_p50_s": _pct([x[0] for x in d["stall"]], 50),
            "stall_p90_s": _pct([x[0] for x in d["stall"]], 90),
            "rewind_frames_p50": _pct([x[1] for x in d["rewind"]], 50),
        }
    return out


def _base_stall_pm(v: float) -> float:
    from .netcode import _STALL_SPEEDS, _interp

    return _interp(STALL_RATE_BY_SPEED, _STALL_SPEEDS, max(v, 0.1))


def session_rates(streams: list[Stream], window_s: float = CLIP_S) -> dict[str, float]:
    """Each session's steady stalls over what the mean session would show, per clip-long window."""
    import numpy as np

    obs: dict[tuple[int, int], float] = {}
    exp: dict[tuple[int, int], float] = {}
    for st in streams:
        if len(st.t) < 30:
            continue
        t, s, ds, dt, v, a, stalled, back = _frames(st)
        steady = np.isfinite(v) & (v > 3.0) & (a > ACCEL_BINS[STEADY_BIN][0]) & (a < ACCEL_BINS[STEADY_BIN][1])
        win = ((t - t[0]) // window_s).astype(int)
        for i in np.where(steady)[0]:
            key = (st.session, int(win[i]))
            exp[key] = exp.get(key, 0.0) + _base_stall_pm(v[i]) * dt[i] / 60.0
            obs.setdefault(key, 0.0)
        for i0, _ in _runs(stalled & steady):
            key = (st.session, int(win[i0]))
            obs[key] = obs.get(key, 0.0) + 1.0
    keys = [k for k in exp if exp[k] > 0.02]
    ratios = [obs[k] / exp[k] for k in keys]
    pooled = sum(obs[k] for k in keys) / max(sum(exp[k] for k in keys), 1e-9)
    return {"sessions": float(len(ratios)), "pooled": pooled,
            "zero_share": float(np.mean([r == 0.0 for r in ratios])),
            "p50": _pct(ratios, 50), "p75": _pct(ratios, 75), "p90": _pct(ratios, 90),
            "p99": _pct(ratios, 99), "mean": float(np.mean(ratios))}


def overshoot(streams: list[Stream]) -> dict[str, float]:
    """Drawn lead over a trend fitted 2 to 3.5 s either side of each braking stall or rewind.

    Read at the correction and 1 s later, per deceleration class. On model streams this
    reads within 0.25 m of the model's own lead, so it can be held against the corpus.
    """
    import numpy as np

    rows: dict[int, list[tuple[float, float]]] = {k: [] for k in range(len(OVERSHOOT_DECEL))}
    for st in streams:
        if len(st.t) < 90:
            continue
        t, s, ds, dt, v, a, stalled, back = _frames(st)
        for mask in (stalled, back):
            for i0, i1 in _runs(mask):
                if i1 >= len(t) - 2 or not (OVERSHOOT_SPEED[0] < v[i0] < OVERSHOOT_SPEED[1]):
                    continue
                k = next((j for j, (lo, hi) in enumerate(OVERSHOOT_DECEL) if lo <= a[i0] < hi), None)
                t0 = t[i0 - 1]
                if k is None or t[0] > t0 - TREND_FAR_S[1] + 0.5 or t[-1] < t0 + TREND_FAR_S[1] - 0.5:
                    continue
                m = (((t >= t0 - TREND_FAR_S[1]) & (t <= t0 - TREND_FAR_S[0]))
                     | ((t >= t0 + TREND_FAR_S[0]) & (t <= t0 + TREND_FAR_S[1])))
                if m.sum() < 20:
                    continue
                p = np.polyfit(t[m] - t0, s[m], 2)
                later = float(np.interp(t0 + 1.0, t, s)) - float(np.polyval(p, 1.0))
                rows[k].append((s[i0 - 1] - float(np.polyval(p, 0.0)), later))
    out: dict[str, float] = {}
    for k, r in rows.items():
        out[f"n_{k}"] = float(len(r))
        out[f"lead_at_{k}"] = _pct([x[0] for x in r], 50)
        out[f"lead_later_{k}"] = _pct([x[1] for x in r], 50)
    return out


def _lsq(t, s, half: float):
    """Centred least-squares quadratic over +-half s at every frame: position, speed, count."""
    import numpy as np

    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half, side="right")
    tt = t - t[0]
    P = [np.r_[0.0, np.cumsum(tt ** k)] for k in range(5)]
    Q = [np.r_[0.0, np.cumsum(tt ** k * s)] for k in range(3)]
    S = [P[k][hi] - P[k][lo] for k in range(5)]
    R = [Q[k][hi] - Q[k][lo] for k in range(3)]
    c = tt
    m = [S[0], S[1] - c * S[0], S[2] - 2 * c * S[1] + c * c * S[0],
         S[3] - 3 * c * S[2] + 3 * c * c * S[1] - c ** 3 * S[0],
         S[4] - 4 * c * S[3] + 6 * c * c * S[2] - 4 * c ** 3 * S[1] + c ** 4 * S[0]]
    r = [R[0], R[1] - c * R[0], R[2] - 2 * c * R[1] + c * c * R[0]]
    A = np.stack([np.stack([m[0], m[1], m[2]], -1), np.stack([m[1], m[2], m[3]], -1),
                  np.stack([m[2], m[3], m[4]], -1)], -2)
    b = np.stack(r, -1)
    ok = (hi - lo) >= 7
    ok[ok] = np.abs(np.linalg.det(A[ok])) > 1e-9
    out = np.full((len(t), 3), np.nan)
    if ok.any():
        out[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return out[:, 0], out[:, 1]


def raw_stats(streams: list[Stream]) -> dict[str, float]:
    """Frame-level artefacts of the drawn position on moving stretches, as a flat dict."""
    import numpy as np

    ratios, rms = [], []
    stalls = rewinds = 0
    moving = 0.0
    for st in streams:
        if len(st.t) < 30:
            continue
        t, s, ds, dt, v, a, stalled, back = _frames(st)
        pos, vel = _lsq(t, s, FIT_HALF_S)
        fast = np.isfinite(vel) & (vel > MIN_SPEED_MS)
        res = (s - pos)[fast]
        if len(res) > 20:
            rms.append(float(np.sqrt(np.mean(res * res))))
        moving += float(np.sum(dt[fast]))
        ok = fast & ~stalled & (dt > 0)
        ratios.extend((ds[ok] / (vel[ok] * dt[ok])).tolist())
        stalls += sum(1 for i0, _ in _runs(stalled) if fast[i0])
        rewinds += sum(1 for i0, _ in _runs(back) if fast[i0])
    minutes = max(moving / 60.0, 1e-9)
    return {
        "moving_min": moving / 60.0,
        "ratio_p10": _pct(ratios, 10), "ratio_p90": _pct(ratios, 90),
        "rms_p50": _pct(rms, 50), "rms_p90": _pct(rms, 90),
        "stalls_per_min": stalls / minutes, "rewinds_per_min": rewinds / minutes,
    }


def frame_steps(streams: list[Stream]) -> dict[int, float]:
    """Share of radar frames that are 1, 2, 3 or 4 physics steps apart."""
    counts: dict[int, int] = {}
    for st in streams:
        for a, b in zip(st.t, st.t[1:]):
            k = int(round((b - a) * PHYSICS_HZ))
            counts[k] = counts.get(k, 0) + 1
    total = max(sum(counts.values()), 1)
    return {k: counts[k] / total for k in sorted(counts) if 1 <= k <= 4}


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


def _chain_lists(streams: list[Stream]) -> dict[str, list[float]]:
    """Raw samples behind `chain_stats`, for one worker's share of the streams."""
    import logging

    logging.disable(logging.WARNING)
    out: dict[str, list[float]] = {"speed_err": [], "accel": [], "pause": [], "rewind": [],
                                   "quiet": [], "freezes": [0.0], "holds": [0.0], "minutes": [0.0]}
    for st in streams:
        if len(st.t) < 30 or (st.s[-1] - st.s[0]) / (st.t[-1] - st.t[0]) < MIN_SPEED_MS:
            continue
        out["minutes"][0] += (st.t[-1] - st.t[0]) / 60.0
        vehicles = []
        for i, prev, v in _chain(st):
            vehicles.append(v)
            if prev is not None:
                out["freezes"][0] += v._lag_since is not None and prev._lag_since is None
                out["holds"][0] += v._pos_mismatch_frames == 1
            if st.t[i] - st.t[0] < 4.0:
                continue
            f = _fit(st.t, st.s, i, STEADY_HALF_S)
            if f is not None and f[1] >= MIN_SPEED_MS and abs(f[2]) <= STEADY_ACCEL_MS2:
                out["speed_err"].append(abs(v.acc_speed - f[1]))
                out["accel"].append(abs(v.acc_accel))
        _responses(st, [v.acc_accel for v in vehicles], out)
    return out


def _responses(st: Stream, accel: list[float], out: dict[str, list[float]]) -> None:
    """Phantom decel `acc_accel` reads within 2 s of a stall, a rewind, or neither, on a steady lead."""
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
        out[kind].append(-min(accel[k] for k in win))


def chain_stats(streams: list[Stream], workers: int = 1) -> dict[str, float]:
    """What the shipped `Vehicle` chain makes of streams moving at `MIN_SPEED_MS` or more.

    Steady stretches: `acc_speed` error against a 4 s fit and |`acc_accel`|. Events: the
    phantom decel `acc_accel` reads within 2 s of a stall, a rewind, or neither.
    """
    if workers <= 1:
        parts = [_chain_lists(streams)]
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_chain_lists, [streams[k::workers] for k in range(workers)]))
    lists = {k: [x for p in parts for x in p[k]] for k in parts[0]}
    minutes = max(sum(lists["minutes"]), 1e-9)
    out = {"acc_speed_err_p50": _pct(lists["speed_err"], 50), "acc_speed_err_p90": _pct(lists["speed_err"], 90),
           "acc_accel_p50": _pct(lists["accel"], 50), "acc_accel_p90": _pct(lists["accel"], 90),
           "freezes_per_min": sum(lists["freezes"]) / minutes,
           "mismatch_holds_per_min": sum(lists["holds"]) / minutes}
    for kind in ("pause", "rewind", "quiet"):
        out[f"{kind}_n"] = float(len(lists[kind]))
        out[f"{kind}_decel_p50"] = _pct(lists[kind], 50)
        out[f"{kind}_decel_p90"] = _pct(lists[kind], 90)
    return out


def _mean_speed(st: Stream, t0: float, t1: float) -> float | None:
    idx = [j for j in range(len(st.t)) if t0 <= st.t[j] <= t1]
    if len(idx) < 8:
        return None
    return (st.s[idx[-1]] - st.s[idx[0]]) / (st.t[idx[-1]] - st.t[idx[0]])


def state_report(corpus: dict, model: dict) -> str:
    """Stalls and rewinds per minute, corpus over model, one row per speed band."""
    head = " " * 12 + " ".join("%17s" % ("a %+.1f..%+.1f" % (max(lo, -9.9), min(hi, 9.9)))
                               for lo, hi in ACCEL_BINS)
    lines = ["stalls / rewinds per minute: corpus | model", head]
    for b, band in enumerate(SPEED_BANDS):
        cells = []
        for c in range(len(ACCEL_BINS)):
            x, y = corpus.get((b, c)), model.get((b, c))
            if x is None or x["minutes"] < 1.0:
                cells.append("%17s" % "-")
                continue
            ys = "%.1f/%.1f" % (y["stall_pm"], y["rewind_pm"]) if y else "-"
            cells.append("%17s" % ("%.1f/%.1f|%s" % (x["stall_pm"], x["rewind_pm"], ys)))
        lines.append("v %4.1f-%4.1f " % band + " ".join(cells))
    return "\n".join(lines)


def report(corpus: dict[str, float], synthetic: dict[str, float]) -> str:
    lines = [f"{'metric':26s} {'corpus':>9s} {'model':>9s}"]
    for key in corpus:
        lines.append(f"{key:26s} {corpus[key]:9.3f} {synthetic.get(key, float('nan')):9.3f}")
    return "\n".join(lines)
