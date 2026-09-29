"""Read brake_debug.csv: what does the braking-intensity slider really buy?

Full-pedal stops per slider value show how much a higher setting adds at full brake;
lag-aligned partial braking shows how far the learned capacity sits from the truck.
Stdlib only. Usage: ``python tools/brake_intensity_probe.py [path/to/brake_debug.csv]``.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "brake_debug.csv"

# Same fitted curve as core/sending_thread/accel_to_pedals.py, kept local so this
# probe never imports core.settings (that can touch the live config).
_RATE, _POWER = 2.4277, 0.8518
# Measurement chain for lag alignment, from core/sending_thread/README.md.
_DEAD_S, _PLANT_TAU_S, _TD_TAU_S = 0.12, 0.25, 0.30
_FULL_PEDAL = 0.85
_MIN_SPEED_MS = 8.0
_MAX_SLOPE_RAD = 0.05
_EPISODE_GAP_S = 0.3
_BINS = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.85)


def frac(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return 0.0 if p <= 0.0 else 1.0 - math.exp(-_RATE * p ** _POWER)


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("t_s", "speed_ms", "accel_ms2", "road_load_ms2", "slope_rad", "sent_brake",
                  "tune_pedal", "brake_intensity", "est_brake_ms2", "baseline_brake_ms2",
                  "brake_scale", "learn_count", "aeb_max_brake_ms2", "gas"):
            r[k] = _f(r, k)
    return rows


def episodes(rows: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for r in rows:
        if out and r["t_s"] - out[-1][-1]["t_s"] <= _EPISODE_GAP_S:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def full_pedal_stops(eps: list[list[dict]]) -> list[dict]:
    stops = []
    for ep in eps:
        hard = [r for r in ep if r["sent_brake"] >= _FULL_PEDAL and r["speed_ms"] >= _MIN_SPEED_MS
                and abs(r["slope_rad"]) <= _MAX_SLOPE_RAD and r["gas"] <= 0.01]
        if len(hard) < 3:
            continue
        best = max(hard, key=lambda r: -r["accel_ms2"] - r["road_load_ms2"])
        dec = -best["accel_ms2"] - best["road_load_ms2"]
        stops.append({
            "t": best["t_s"], "v0_kmh": max(r["speed_ms"] for r in ep) * 3.6,
            "I": best["brake_intensity"], "sent": best["sent_brake"],
            "a_phys": dec / max(frac(best["sent_brake"]), 1e-3),
            "a_tune": dec / max(frac(best["tune_pedal"]), 1e-3),
            "base": best["baseline_brake_ms2"], "est": best["est_brake_ms2"],
        })
    return stops


def lagged(ep: list[dict], key: str) -> list[float]:
    """Push *key* through dead time, plant lag and the tracking differentiator."""
    out, s1, s2, j = [], 0.0, 0.0, -1
    for i, r in enumerate(ep):
        while j < i and ep[j + 1]["t_s"] <= r["t_s"] - _DEAD_S:
            j += 1
        x = ep[j][key] if j >= 0 else 0.0
        dt = 0.05 if i == 0 else max(r["t_s"] - ep[i - 1]["t_s"], 1e-3)
        s1 += (1.0 - math.exp(-dt / _PLANT_TAU_S)) * (x - s1)
        s2 += (1.0 - math.exp(-dt / _TD_TAU_S)) * (s1 - s2)
        out.append(s2)
    return out


def partial_response(eps: list[list[dict]]) -> list[tuple]:
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    for ep in eps:
        for r in ep:
            r["_f_lag"] = frac(r["tune_pedal"]) if r["tune_pedal"] == r["tune_pedal"] else 0.0
        f_lag = lagged(ep, "_f_lag")
        p_lag = lagged(ep, "tune_pedal")
        for r, fl, pl in zip(ep, f_lag, p_lag):
            if (r["speed_ms"] < _MIN_SPEED_MS or abs(r["slope_rad"]) > _MAX_SLOPE_RAD
                    or r["gas"] > 0.01 or r["t_s"] - ep[0]["t_s"] < 1.0 or fl <= 0.02):
                continue
            b = next((i for i, hi in enumerate(_BINS) if pl <= hi), None)
            if b is None:
                continue
            dec = -r["accel_ms2"] - r["road_load_ms2"]
            buckets.setdefault(b, []).append((pl, dec / fl, r["est_brake_ms2"]))
    rows = []
    for b in sorted(buckets):
        vals = buckets[b]
        a = statistics.median(v[1] for v in vals)
        est = statistics.median(v[2] for v in vals)
        rows.append((_BINS[b], len(vals), statistics.median(v[0] for v in vals), a, est, a / est))
    return rows


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CSV
    if not path.exists():
        print(f"no {path.name}; enable debug in config.json and brake in game first")
        return 1
    rows = load(path)
    eps = episodes(rows)
    print(f"{path.name}: {len(rows)} rows, {len(eps)} braking episodes")

    stops = full_pedal_stops(eps)
    print("\nFull-pedal stops (peak A: curve on sent pedal = phys, on tune pedal = tune)")
    print("   t_s    v0   I     sent  A_phys  A_tune   base    est")
    for s in stops:
        print(f"{s['t']:7.1f} {s['v0_kmh']:5.0f} {s['I']:5.3f} {s['sent']:5.2f} "
              f"{s['a_phys']:7.2f} {s['a_tune']:7.2f} {s['base']:6.2f} {s['est']:6.2f}")
    by_i: dict[float, list[float]] = {}
    for s in stops:
        by_i.setdefault(round(s["I"], 2), []).append(s["a_phys"] / s["base"])
    if len(by_i) >= 2:
        lo, hi = min(by_i), max(by_i)
        ratio = statistics.median(by_i[hi]) / statistics.median(by_i[lo])
        print(f"\nfull-pedal capacity I={hi} vs I={lo}: x{ratio:.2f} "
              f"(slider ratio {hi / lo:.2f}; AEB counts on x1.00)")

    print("\nPartial braking, lag-aligned: delivered capacity vs learned (tune units)")
    print(" pedal<=    n  pedal  A_real  A_est  real/est")
    for hi, n, pl, a, est, r in partial_response(eps):
        print(f"  {hi:5.2f} {n:5d}  {pl:5.3f}  {a:6.2f} {est:6.2f}   {r:5.2f}")

    gates = Counter(r.get("learn_gate", "") for r in rows)
    counts = [r["learn_count"] for r in rows if r["learn_count"] == r["learn_count"]]
    scales = [r["brake_scale"] for r in rows if r["brake_scale"] == r["brake_scale"]]
    if counts and scales:
        print(f"\nlearner: {int(counts[-1] - counts[0])} samples accepted, "
              f"brake_scale {scales[0]:.3f} -> {scales[-1]:.3f}")
    print("gates:", ", ".join(f"{k or '-'} {v}" for k, v in gates.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
