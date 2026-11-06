"""Numbers for one bench run: how the wheel answered a step, how closely it followed a moving target.

A run is the dict ``tools/la_servo_bench.py`` records: ``samples`` rows of ``[t, steer, rate, applied, servo,
game, integ, scale]``, ``sent`` rows of ``[t, target, rate]``, the stage and the gains held.
"""

from __future__ import annotations

import math
from bisect import bisect_right

EXTRAP_S = 0.15
LAG_SCAN_S = 0.4
LAG_STEP_S = 0.005


def interp(ts: list[float], vs: list[float], t: float) -> float:
    i = bisect_right(ts, t)
    if i <= 0:
        return vs[0]
    if i >= len(ts):
        return vs[-1]
    t0, t1 = ts[i - 1], ts[i]
    return vs[i - 1] if t1 <= t0 else vs[i - 1] + (vs[i] - vs[i - 1]) * (t - t0) / (t1 - t0)


def aim_at(sent: list[list[float]], times: list[float], t: float) -> float:
    """Where the plugin aims at ``t``: the last request, its target moved on at its rate (and acceleration, when
    recorded) for at most 0.15 s.
    """
    i = bisect_right(times, t) - 1
    if i < 0:
        return sent[0][1]
    t_k, target, rate = sent[i][:3]
    accel = sent[i][3] if len(sent[i]) > 3 else 0.0
    a = min(max(t - t_k, 0.0), EXTRAP_S)
    return target + rate * a + 0.5 * accel * a * a


def step_response(ts: list[float], ys: list[float], t_step: float, start: float, goal: float, hold_s: float,
                  band_abs: float = 0.003) -> dict:
    """Rise 10 to 90 %, overshoot (fraction of the step), settling time into a band, and the hold's error and ripple."""
    out = {"rise_s": None, "overshoot": 0.0, "settle_s": None, "final_err": None, "ripple": None}
    size = goal - start
    seg = [(t, (y - start) / size) for t, y in zip(ts, ys) if t_step <= t < t_step + hold_s] if size else []
    if len(seg) < 5:
        return out
    t10 = next((t for t, y in seg if y >= 0.1), None)
    t90 = next((t for t, y in seg if y >= 0.9), None)
    if t10 is not None and t90 is not None:
        out["rise_s"] = t90 - t10
    out["overshoot"] = max(0.0, max(y for _, y in seg) - 1.0)
    band = max(band_abs / abs(size), 0.05)
    outside = [t for t, y in seg if abs(y - 1.0) > band]
    if abs(seg[-1][1] - 1.0) <= band:
        out["settle_s"] = outside[-1] - t_step if outside else 0.0
    tail = [y * size for t, y in seg if t >= t_step + hold_s - 0.5]
    if len(tail) >= 3:
        mean = sum(tail) / len(tail)
        out["final_err"] = mean / size - 1.0
        out["ripple"] = math.sqrt(sum((v - mean) ** 2 for v in tail) / len(tail))
    return out


def tracking(ts: list[float], ys: list[float], sent: list[list[float]]) -> dict:
    """RMS and worst error against the plugin's aim, and the delay that makes the wheel match it best."""
    times = [s[0] for s in sent]
    pts = list(zip(ts, ys))[::2]
    if not pts:
        return {"rms": None, "max": None, "lag_s": None}
    errs = [y - aim_at(sent, times, t) for t, y in pts]
    best, best_sse = 0.0, math.inf
    for k in range(int(LAG_SCAN_S / LAG_STEP_S) + 1):
        lag = k * LAG_STEP_S
        sse = sum((y - aim_at(sent, times, t - lag)) ** 2 for t, y in pts)
        if sse < best_sse:
            best, best_sse = lag, sse
    return {"rms": math.sqrt(sum(e * e for e in errs) / len(errs)), "max": max(abs(e) for e in errs), "lag_s": best}


def summarize(run: dict) -> dict:
    ts = [s[0] for s in run["samples"]]
    ys = [s[1] for s in run["samples"]]
    sent = run["sent"]
    out = {
        "name": run["name"], "label": run["label"], "kind": run["stage"]["kind"], "n": len(ts),
        "peak_servo": max((abs(s[4]) for s in run["samples"]), default=0.0),
        "buzz_min": min((s[7] for s in run["samples"]), default=1.0), "aborted": run.get("aborted"),
    }
    if out["kind"] == "step" and len(sent) >= 4:
        hold = sent[2][0] - sent[1][0]
        out["up"] = step_response(ts, ys, sent[1][0], sent[0][1], sent[1][1], hold)
        out["swing"] = step_response(ts, ys, sent[2][0], sent[1][1], sent[2][1], sent[3][0] - sent[2][0])
    elif ts:
        out["track"] = tracking(ts, ys, sent)
    return out


def _num(v: float | None, fmt: str, scale: float = 1.0, unit: str = "") -> str:
    return "never" if v is None else format(v * scale, fmt) + unit


def describe(s: dict) -> str:
    head = f"{s['name']} [{s['label']}]" + (" ABORTED" if s.get("aborted") else "")
    tail = f"servo peak {s['peak_servo']:.0f}, buzz min {s['buzz_min']:.2f}"
    if "up" in s:
        u, w = s["up"], s["swing"]
        return (f"{head}: rise {_num(u['rise_s'], '.3f', 1.0, ' s')}, over {_num(u['overshoot'], '.0f', 100, ' %')}, "
                f"settle {_num(u['settle_s'], '.2f', 1.0, ' s')}, ripple {_num(u['ripple'], '.4f')}; swing over "
                f"{_num(w['overshoot'], '.0f', 100, ' %')}, settle {_num(w['settle_s'], '.2f', 1.0, ' s')}; {tail}")
    if "track" in s:
        k = s["track"]
        return (f"{head}: error rms {_num(k['rms'], '.4f')}, worst {_num(k['max'], '.4f')}, lag "
                f"{_num(k['lag_s'], '.3f', 1.0, ' s')}; {tail}")
    return f"{head}: no samples"
