"""Where MonoCruise lags: the GUI stall probe's lines in monocruise.log, ranked by place and by thread.

Needs a drive with ``debug`` on. Every 10 s the probe logs process CPU per thread, each
worker's loop times against its budget and ego's place, traffic and map; this groups those
windows into map cells and ranks them. Stdlib only, never imports ``core.settings``.

    python tools/lag_report.py                       # monocruise.log in the repo root
    python tools/lag_report.py monocruise.log --cell 500 --top 20
    python tools/lag_report.py --json                # for agents

See core/thread_management/README.md, "GUI stall probe".
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HEAD = r"^(\d\d:\d\d:\d\d) \[gui_stall\s*\] \w+\s+"
_REPORT = re.compile(_HEAD + r"GUI longest gap (\d+) ms in the last \d+ s\. (.*)$")
_STALL = re.compile(_HEAD + r"GUI stalled (\d+) ms(?: \(\+(\d+) [^)]*\))?\. GUI thread at: (.*?)\. ((?:From \d+ ms in: )?(?:thread )?CPU.*)$")
# Stall lines before this prefix divided CPU read from when the gap was seen by the whole stall.
_FROM = re.compile(r"From \d+ ms in: ")
_LOOPS = re.compile(_HEAD + r"Loops in the last \d+ s: (.*)$")
_CPU = re.compile(r"CPU (\d+) ms over (\d+) ms: (.*)")
_SHARE = re.compile(r"(\S+) (\d+) ms \(\d+%\)")
_LOOP_ROW = re.compile(
    r"(\S+) (\d+)/(\d+) Hz, p50 ([\d.]+) p99 ([\d.]+) max (\d+) ms of (\d+)(?:, cpu ([\d.]+) ms)?"
)
_POS = re.compile(r"x (-?\d+) z (-?\d+), (\d+) km/h, (SP|TMP)(, paused)?")
_COUNTS = re.compile(r"(\d+) vehicles \+ (\d+) trailers, map (\S+) (\d+) lanes (\d+) routed")
# A stall with less process CPU than this share of its span is quiet: the GUI waited on
# something other than busy Python threads (OS scheduling, disk, a blocking native call).
_BUSY_SHARE = 0.5
_CONTEXT_KEYS = ("vehicles", "trailers", "lanes", "routed")


@dataclass
class Context:
    x: float | None = None
    z: float | None = None
    speed_kmh: float = 0.0
    tmp: bool = False
    paused: bool = False
    vehicles: int = 0
    trailers: int = 0
    map_status: str = "unknown"
    lanes: int = 0
    routed: int = 0


@dataclass
class Loop:
    hz: float
    target_hz: float
    p50_ms: float
    p99_ms: float
    max_ms: float
    budget_ms: float
    cpu_ms: float | None


@dataclass
class Window:
    t: str
    gap_ms: float
    cpu_share: float | None
    threads: dict[str, float]
    ctx: Context
    loops: dict[str, Loop] = field(default_factory=dict)


@dataclass
class Stall:
    t: str
    ms: float
    cpu_share: float | None
    gui_at: str
    ctx: Context

    @property
    def kind(self) -> str:
        if self.cpu_share is None:
            return "unknown"
        return "busy" if self.cpu_share >= _BUSY_SHARE else "quiet"


def _cpu(text: str) -> tuple[float | None, dict[str, float]]:
    """Process share of one core and per-thread shares from a probe CPU field."""
    m = _CPU.match(text)
    if not m:
        return None, {}
    span = max(float(m.group(2)), 1.0)
    shares = {name: float(ms) / span for name, ms in _SHARE.findall(m.group(3))}
    return float(m.group(1)) / span, shares


def _context(text: str) -> Context:
    ctx = Context()
    m = _POS.search(text)
    if m:
        ctx.x, ctx.z, ctx.speed_kmh = float(m.group(1)), float(m.group(2)), float(m.group(3))
        ctx.tmp, ctx.paused = m.group(4) == "TMP", bool(m.group(5))
    m = _COUNTS.search(text)
    if m:
        ctx.vehicles, ctx.trailers = int(m.group(1)), int(m.group(2))
        ctx.map_status, ctx.lanes, ctx.routed = m.group(3), int(m.group(4)), int(m.group(5))
    return ctx


def _split_at(text: str) -> tuple[str, Context]:
    cpu, _, ctx = text.partition(". At ")
    return cpu, _context(ctx)


def parse(lines) -> tuple[list[Window], list[Stall]]:
    """Probe windows and stall lines; older logs without loop or place fields still parse."""
    windows: list[Window] = []
    stalls: list[Stall] = []
    for line in lines:
        line = line.rstrip("\n")
        m = _REPORT.match(line)
        if m:
            cpu_text, ctx = _split_at(m.group(3))
            share, threads = _cpu(cpu_text)
            windows.append(Window(m.group(1), float(m.group(2)), share, threads, ctx))
            continue
        m = _LOOPS.match(line)
        if m and windows and not windows[-1].loops:
            for row in _LOOP_ROW.finditer(m.group(2)):
                name, hz, target, p50, p99, worst, budget, cpu = row.groups()
                windows[-1].loops[name] = Loop(
                    float(hz), float(target), float(p50), float(p99), float(worst), float(budget),
                    float(cpu) if cpu is not None else None,
                )
            continue
        m = _STALL.match(line)
        if m:
            cpu_text, ctx = _split_at(m.group(5))
            seen = _FROM.match(cpu_text)
            share = _cpu(cpu_text[seen.end():])[0] if seen else None
            stalls.append(Stall(m.group(1), float(m.group(2)), share, m.group(4), ctx))
    return windows, stalls


def _driving(w: Window) -> bool:
    return w.ctx.x is not None and not w.ctx.paused and w.cpu_share is not None


def _cell(ctx: Context, size: float) -> tuple[int, int]:
    return math.floor(ctx.x / size), math.floor(ctx.z / size)


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def analyse(windows: list[Window], stalls: list[Stall], cell_m: float = 1000.0, top: int = 15) -> dict:
    drive = [w for w in windows if _driving(w)]
    shares = [w.cpu_share for w in windows if w.cpu_share is not None]
    out: dict = {
        "windows": len(windows),
        "driving_windows": len(drive),
        "process_cpu_pct": {
            "p50": round(100 * _pct(shares, 0.5)), "p90": round(100 * _pct(shares, 0.9)),
            "max": round(100 * max(shares, default=0.0)),
        },
        "windows_over_80_pct": sum(1 for s in shares if s >= 0.8),
        "stalls": {k: sum(1 for s in stalls if s.kind == k) for k in ("busy", "quiet", "unknown")},
        "has_place": any(w.ctx.x is not None for w in windows),
    }
    cells: dict[tuple[int, int], list[Window]] = defaultdict(list)
    for w in drive:
        cells[_cell(w.ctx, cell_m)].append(w)
    cell_stalls: dict[tuple[int, int], list[Stall]] = defaultdict(list)
    for s in stalls:
        if s.ctx.x is not None:
            cell_stalls[_cell(s.ctx, cell_m)].append(s)
    rows = []
    for key, ws in cells.items():
        busiest: dict[str, float] = defaultdict(float)
        for w in ws:
            for name, share in w.threads.items():
                busiest[name] += share / len(ws)
        name, share = max(busiest.items(), key=lambda kv: kv[1], default=("none", 0.0))
        st = cell_stalls.get(key, [])
        rows.append({
            "x": round((key[0] + 0.5) * cell_m), "z": round((key[1] + 0.5) * cell_m),
            "windows": len(ws), "tmp": any(w.ctx.tmp for w in ws),
            "cpu_mean_pct": round(100 * sum(w.cpu_share for w in ws) / len(ws)),
            "cpu_max_pct": round(100 * max(w.cpu_share for w in ws)),
            "worst_gap_ms": round(max(max((w.gap_ms for w in ws), default=0.0), max((s.ms for s in st), default=0.0))),
            "stalls_busy": sum(1 for s in st if s.kind == "busy"), "stalls_quiet": sum(1 for s in st if s.kind == "quiet"),
            "vehicles_max": max(w.ctx.vehicles for w in ws), "trailers_max": max(w.ctx.trailers for w in ws),
            "lanes_max": max(w.ctx.lanes for w in ws), "routed_max": max(w.ctx.routed for w in ws),
            "busiest": name, "busiest_pct": round(100 * share),
        })
    rows.sort(key=lambda r: (r["cpu_mean_pct"], r["worst_gap_ms"]), reverse=True)
    out["cells"] = rows[:top]
    out["threads"] = _threads(drive if out["has_place"] else [w for w in windows if w.cpu_share is not None])
    out["worst_stalls"] = [
        {"t": s.t, "ms": round(s.ms), "cpu_pct": None if s.cpu_share is None else round(100 * s.cpu_share),
         "kind": s.kind, "gui_at": " | ".join(s.gui_at.split(" | ")[-2:]), "ctx": asdict(s.ctx)}
        for s in sorted(stalls, key=lambda s: s.ms, reverse=True)[:10]
    ]
    return out


def _threads(drive: list[Window]) -> list[dict]:
    names = sorted({n for w in drive for n in w.threads} | {n for w in drive for n in w.loops})
    rows = []
    for name in names:
        cpu = [w.threads.get(name, 0.0) for w in drive]
        loops = [w.loops[name] for w in drive if name in w.loops]
        corr = {}
        for key in _CONTEXT_KEYS:
            r = _pearson([float(getattr(w.ctx, key)) for w in drive], cpu)
            corr[key] = None if r is None else round(r, 2)
        rows.append({
            "thread": name,
            "cpu_mean_pct": round(100 * sum(cpu) / len(cpu), 1) if cpu else 0.0,
            "cpu_max_pct": round(100 * max(cpu, default=0.0)),
            "loop_windows": len(loops),
            "p99_over_budget": sum(1 for lp in loops if lp.p99_ms > lp.budget_ms),
            "rate_under_90_pct": sum(1 for lp in loops if lp.target_hz > 0 and lp.hz < 0.9 * lp.target_hz),
            "worst_p99_ms": max((lp.p99_ms for lp in loops), default=None),
            "worst_max_ms": max((lp.max_ms for lp in loops), default=None),
            "budget_ms": loops[0].budget_ms if loops else None,
            "corr": corr,
        })
    rows.sort(key=lambda r: r["cpu_mean_pct"], reverse=True)
    return rows


def _fmt(v) -> str:
    return "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)


def render(report: dict, cell_m: float) -> str:
    cpu = report["process_cpu_pct"]
    out = [
        f"{report['windows']} probe windows of 10 s, {report['driving_windows']} driving.",
        f"Process CPU, % of one core (the GIL ceiling is 100): p50 {cpu['p50']}, p90 {cpu['p90']}, max {cpu['max']}; "
        f"{report['windows_over_80_pct']} windows at 80 or more.",
        f"GUI stalls: {report['stalls']['busy']} busy (threads used the CPU), "
        f"{report['stalls']['quiet']} quiet (little CPU: OS scheduling, disk or a blocking native call), "
        f"{report['stalls']['unknown']} unknown (no CPU reading, or a probe line from before 2026-11-23).",
        "",
    ]
    if not report["has_place"]:
        out += ["No place in these lines: the log predates the probe's place fields. Drive again with debug on.", ""]
    else:
        out.append(f"Hotspots, {cell_m:.0f} m cells, busiest first:")
        out.append(f"{'x':>7} {'z':>7} {'mode':4} {'win':>4} {'cpu%':>5} {'max%':>5} {'gap ms':>7} {'busy/q':>7} "
                   f"{'veh':>4} {'trl':>4} {'lanes':>6} {'routed':>6}  busiest")
        for r in report["cells"]:
            out.append(
                f"{r['x']:>7} {r['z']:>7} {'TMP' if r['tmp'] else 'SP':4} {r['windows']:>4} {r['cpu_mean_pct']:>5} "
                f"{r['cpu_max_pct']:>5} {r['worst_gap_ms']:>7} {r['stalls_busy']:>3}/{r['stalls_quiet']:<3} "
                f"{r['vehicles_max']:>4} {r['trailers_max']:>4} {r['lanes_max']:>6} {r['routed_max']:>6}  "
                f"{r['busiest']} {r['busiest_pct']}%"
            )
        out.append("")
    scope = "while driving" if report["has_place"] else "in every window"
    out.append(f"Threads {scope} (corr: CPU against vehicles / trailers / lanes / routed per window):")
    out.append(f"{'thread':<24} {'cpu%':>5} {'max%':>5} {'p99>budget':>10} {'rate<90%':>8} "
               f"{'worst p99':>9} {'worst max':>9} {'budget':>6}  corr")
    for r in report["threads"]:
        if r["cpu_mean_pct"] < 0.5 and not r["p99_over_budget"]:
            continue
        corr = " ".join(_fmt(r["corr"][k]) for k in _CONTEXT_KEYS)
        late = f"{r['p99_over_budget']}/{r['loop_windows']}" if r["loop_windows"] else "-"
        slow = f"{r['rate_under_90_pct']}/{r['loop_windows']}" if r["loop_windows"] else "-"
        out.append(
            f"{r['thread']:<24} {r['cpu_mean_pct']:>5} {r['cpu_max_pct']:>5} {late:>10} {slow:>8} "
            f"{_fmt(r['worst_p99_ms']):>9} {_fmt(r['worst_max_ms']):>9} {_fmt(r['budget_ms']):>6}  {corr}"
        )
    if report["worst_stalls"]:
        out += ["", "Worst GUI stalls:"]
        for s in report["worst_stalls"]:
            c = s["ctx"]
            where = "place unknown" if c["x"] is None else f"x {c['x']:.0f} z {c['z']:.0f} {'TMP' if c['tmp'] else 'SP'}"
            out.append(f"  {s['t']} {s['ms']} ms {s['kind']} (cpu {_fmt(s['cpu_pct'])}%) {where}: GUI at {s['gui_at']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="?", default=str(_ROOT / "monocruise.log"))
    ap.add_argument("--cell", type=float, default=1000.0, help="cell size in metres (default 1000)")
    ap.add_argument("--top", type=int, default=15, help="hotspot rows (default 15)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    path = Path(args.log)
    if not path.is_file():
        print(f"no log at {path.name}", file=sys.stderr)
        return 1
    with path.open(encoding="utf-8", errors="replace") as fh:
        windows, stalls = parse(fh)
    report = analyse(windows, stalls, args.cell, args.top)
    print(json.dumps(report, indent=1) if args.json else render(report, args.cell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
