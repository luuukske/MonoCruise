"""Which Python code held the GIL in a py-spy drive recording (tools/profile_drive.py), per thread and chunk.

MonoCruise has one core for all its Python threads, so a function's share of a chunk's GIL
samples is its share of that core. The default compares the hottest chunks with the calm half of
the drive: what grew is what to look at. Places come from the GUI stall probe in monocruise.log.

    python tools/profile_report.py                    # latest recording in tools/profile_out
    python tools/profile_report.py --at 01:16:05      # that chunk in detail, by log time
    python tools/profile_report.py --hot 5 --json     # for agents

See tools/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import lag_report  # noqa: E402

OUT = ROOT / "tools" / "profile_out"
_THREAD = re.compile(r"^thread \(([^)]*)\)(?::\s*(.*))?$")
_FRAME = re.compile(r"^(.*) \((.+):(\d+)\)$")
# A frame in this share of its thread's samples is a wrapper (thread bootstrap, loop runner).
_WRAPPER_SHARE = 0.9
_WRAPPER_FILES = ("threading.py", "core/thread_management/base_thread.py")
_TOP = 8


@dataclass
class ThreadProfile:
    samples: int = 0
    own: Counter = field(default_factory=Counter)
    incl: Counter = field(default_factory=Counter)


@dataclass
class Chunk:
    file: str
    start: datetime
    end: datetime
    expected: float
    threads: dict[str, ThreadProfile]
    place: str | None = None

    @property
    def samples(self) -> int:
        return sum(t.samples for t in self.threads.values())

    @property
    def share(self) -> float:
        return self.samples / max(self.expected, 1.0)


class _Shortener:
    """Repo-relative file names, the last two parts outside the repo: no absolute path is printed."""

    def __init__(self, root: Path) -> None:
        self._root = os.path.normcase(os.path.abspath(root)) + os.sep
        self._cache: dict[str, str] = {}

    def __call__(self, path: str) -> str:
        short = self._cache.get(path)
        if short is None:
            full = os.path.abspath(path)
            if os.path.normcase(full).startswith(self._root):
                short = full[len(self._root):].replace(os.sep, "/")
            else:
                short = "/".join(Path(path).parts[-2:])
            self._cache[path] = short
        return short


def _thread_name(token: str) -> str:
    m = _THREAD.match(token)
    if not m:
        return token
    name = (m.group(2) or "").strip() or f"thread {m.group(1)}"
    return "gui" if name == "MainThread" else name


def parse_raw(lines, short: _Shortener) -> dict[str, ThreadProfile]:
    """py-spy raw lines (``thread;frame;frame count``) into per-thread own and inclusive samples."""
    threads: dict[str, ThreadProfile] = {}
    for line in lines:
        stack, _, count = line.rstrip("\n").rpartition(" ")
        if not stack or not count.isdigit():
            continue
        parts = stack.split(";")
        if parts[0].startswith("process ") and len(parts) > 1:
            parts = parts[1:]
        prof = threads.setdefault(_thread_name(parts[0]), ThreadProfile())
        keys = []
        for frame in parts[1:]:
            m = _FRAME.match(frame)
            keys.append(f"{short(m.group(2))}:{m.group(1)}" if m else frame)
        n = int(count)
        prof.samples += n
        if keys:
            prof.own[keys[-1]] += n
            for key in set(keys):
                prof.incl[key] += n
    return threads


def load(rec_dir: Path, root: Path = ROOT) -> tuple[dict, list[Chunk]]:
    manifest = json.loads((rec_dir / "manifest.json").read_text(encoding="utf-8"))
    rate = float(manifest.get("rate_hz", 100))
    chunk_s = float(manifest.get("chunk_s", 10))
    short = _Shortener(root)
    rows = []
    for c in manifest.get("chunks", []):
        path = rec_dir / c["file"]
        if path.is_file():
            start, end = datetime.fromisoformat(c["start"]), datetime.fromisoformat(c["end"])
            rows.append((c["file"], path, start, end, (end - start).total_seconds()))
    # py-spy takes rate x chunk ticks per chunk, a little faster than the wall clock on Windows;
    # only a chunk cut short (MonoCruise closed) has fewer, in proportion to its wall time.
    full = sorted(wall for *_, wall in rows if wall >= 0.8 * chunk_s)
    ref = full[len(full) // 2] if full else chunk_s
    chunks = []
    for name, path, start, end, wall in rows:
        with path.open(encoding="utf-8", errors="replace") as fh:
            threads = parse_raw(fh, short)
        expected = rate * chunk_s * min(1.0, max(wall, 0.1) / ref)
        chunks.append(Chunk(name, start, end, expected, threads))
    return manifest, chunks


def latest_recording(out: Path = OUT) -> Path | None:
    """The newest recording folder, with chunks or not, so an empty one can say why."""
    runs = sorted(p for p in out.glob("*") if p.is_dir()) if out.is_dir() else []
    return runs[-1] if runs else None


def why_empty(rec: Path) -> str:
    error = rec / "error.txt"
    reason = error.read_text(encoding="utf-8", errors="replace").strip() if error.is_file() else "no py-spy chunk finished"
    return f"recording {rec.name} has no chunks: {reason}"


def _tod(t: datetime) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def attach_places(chunks: list[Chunk], log_lines) -> None:
    """Each chunk gets the stall probe window that overlaps it most (a window is the 10 s before its line)."""
    windows, _ = lag_report.parse(log_lines)
    spans = []
    for w in windows:
        h, m, s = (int(v) for v in w.t.split(":"))
        end = h * 3600 + m * 60 + s
        spans.append((end - 10.0, end, w))
    for c in chunks:
        lo, hi = _tod(c.start), _tod(c.end)
        best, best_overlap = None, 3.0
        for a, b, w in spans:
            overlap = min(hi, b) - max(lo, a)
            if overlap > best_overlap:
                best, best_overlap = w, overlap
        if best is not None:
            c.place = _describe(best)


def _describe(w) -> str:
    ctx = w.ctx
    cpu = "" if w.cpu_share is None else f"process {100 * w.cpu_share:.0f} %"
    if ctx.x is None:
        return cpu or "place unknown"
    return (f"{cpu}, x {ctx.x:.0f} z {ctx.z:.0f}, {ctx.speed_kmh:.0f} km/h, {'TMP' if ctx.tmp else 'SP'}, "
            f"{ctx.vehicles} vehicles, {ctx.routed} routed, {ctx.lanes} lanes")


def _wrappers(chunks: list[Chunk]) -> set[tuple[str, str]]:
    total: dict[str, int] = Counter()
    incl: Counter = Counter()
    for c in chunks:
        for name, t in c.threads.items():
            total[name] += t.samples
            for key, n in t.incl.items():
                incl[(name, key)] += n
    return {
        k for k, n in incl.items()
        if (total[k[0]] and n >= _WRAPPER_SHARE * total[k[0]]) or k[1].split(":")[0].endswith(_WRAPPER_FILES)
    }


def _mean_shares(chunks: list[Chunk], attr: str) -> Counter:
    out: Counter = Counter()
    for c in chunks:
        for name, t in c.threads.items():
            for key, n in getattr(t, attr).items():
                out[(name, key)] += n / c.expected / len(chunks)
    return out


def _thread_shares(chunks: list[Chunk]) -> Counter:
    out: Counter = Counter()
    for c in chunks:
        for name, t in c.threads.items():
            out[name] += t.samples / c.expected / len(chunks)
    return out


def grew(hot: list[Chunk], calm: list[Chunk], top: int = 15) -> dict:
    """Thread and function shares of one core, hot chunks against calm ones, largest growth first."""
    wrap = _wrappers(hot + calm)
    out = {}
    th_hot, th_calm = _thread_shares(hot), _thread_shares(calm)
    out["threads"] = sorted(
        ({"thread": n, "hot_pct": round(100 * th_hot[n], 1), "calm_pct": round(100 * th_calm[n], 1),
          "delta_pct": round(100 * (th_hot[n] - th_calm[n]), 1)} for n in set(th_hot) | set(th_calm)),
        key=lambda r: r["delta_pct"], reverse=True,
    )[:top]
    for attr, label in (("own", "own"), ("incl", "inclusive")):
        h, c = _mean_shares(hot, attr), _mean_shares(calm, attr)
        rows = [
            {"thread": k[0], "function": k[1], "hot_pct": round(100 * h[k], 1), "calm_pct": round(100 * c[k], 1),
             "delta_pct": round(100 * (h[k] - c[k]), 1)}
            for k in set(h) | set(c) if not (attr == "incl" and k in wrap)
        ]
        rows.sort(key=lambda r: r["delta_pct"], reverse=True)
        out[label] = rows[:top]
    return out


def detail(chunk: Chunk, wrap: set[tuple[str, str]], top: int = _TOP) -> dict:
    threads = []
    for name, t in sorted(chunk.threads.items(), key=lambda kv: kv[1].samples, reverse=True):
        if t.samples < 0.005 * chunk.expected:
            continue

        def pct(n: int) -> float:
            return round(100 * n / chunk.expected, 1)

        incl = [(k, n) for k, n in t.incl.most_common() if (name, k) not in wrap][:top]
        threads.append({
            "thread": name, "pct": pct(t.samples),
            "own": [{"function": k, "pct": pct(n)} for k, n in t.own.most_common(top)],
            "inclusive": [{"function": k, "pct": pct(n)} for k, n in incl],
        })
    return {"start": chunk.start.strftime("%H:%M:%S"), "end": chunk.end.strftime("%H:%M:%S"),
            "gil_pct": round(100 * chunk.share), "place": chunk.place, "threads": threads}


def analyse(chunks: list[Chunk], hot_n: int = 3, at: str | None = None) -> dict:
    ranked = sorted(chunks, key=lambda c: c.share, reverse=True)
    hot = ranked[:hot_n]
    median = sorted(c.share for c in chunks)[len(chunks) // 2] if chunks else 0.0
    calm = [c for c in chunks if c.share <= median and c not in hot] or ranked[hot_n:]
    shares = sorted(c.share for c in chunks)
    report = {
        "chunks": len(chunks),
        "gil_pct": {"p50": round(100 * median), "p90": round(100 * shares[int(0.9 * (len(shares) - 1))]) if shares else 0,
                    "max": round(100 * shares[-1]) if shares else 0},
        "hot": [{"start": c.start.strftime("%H:%M:%S"), "gil_pct": round(100 * c.share), "place": c.place} for c in hot],
        "calm_chunks": len(calm),
        "grew": grew(hot, calm) if hot and calm else {},
    }
    focus = hot[0] if hot else None
    if at:
        h, m, s = (int(v) for v in at.split(":"))
        t = h * 3600 + m * 60 + s
        focus = next((c for c in chunks if _tod(c.start) <= t <= _tod(c.end)), None)
        if focus is None:
            report["error"] = f"no chunk holds {at}"
    report["detail"] = detail(focus, _wrappers(chunks)) if focus else None
    return report


def render(report: dict, name: str) -> str:
    g = report["gil_pct"]
    out = [f"Recording {name}: {report['chunks']} chunks. GIL held, % of each chunk: p50 {g['p50']}, p90 {g['p90']}, max {g['max']}.", ""]
    if report.get("error"):
        out += [report["error"], ""]
    if report["grew"]:
        hot = ", ".join(f"{h['start']} ({h['gil_pct']} %)" for h in report["hot"])
        out.append(f"Grew in the hottest chunks [{hot}] against the {report['calm_chunks']} calm ones, % of one core:")
        for label in ("threads", "own", "inclusive"):
            out.append(f"  {label}:")
            for r in report["grew"][label]:
                if r["delta_pct"] <= 0.0:
                    continue
                what = r["thread"] if label == "threads" else f"{r['thread']:<22} {r['function']}"
                out.append(f"    {r['delta_pct']:+6.1f}  {what}  (hot {r['hot_pct']}, calm {r['calm_pct']})")
        out.append("")
    d = report["detail"]
    if d:
        out.append(f"Chunk {d['start']}-{d['end']}, GIL held {d['gil_pct']} %" + (f" ({d['place']})" if d["place"] else "") + ":")
        for t in d["threads"]:
            out.append(f"  {t['thread']} {t['pct']} %")
            for label in ("own", "inclusive"):
                if t[label]:
                    out.append(f"    {label + ':':<11}" + "; ".join(f"{f['function']} {f['pct']}" for f in t[label]))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", nargs="?", help="a folder in tools/profile_out (default: the latest)")
    ap.add_argument("--log", default=str(ROOT / "monocruise.log"), help="log with the GUI stall probe lines")
    ap.add_argument("--hot", type=int, default=3, help="hottest chunks to compare (default 3)")
    ap.add_argument("--at", help="show the chunk holding this log time (HH:MM:SS)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    rec = Path(args.recording) if args.recording else latest_recording()
    if rec is None:
        print("no recording in tools/profile_out: run python tools/profile_drive.py first", file=sys.stderr)
        return 1
    if not (rec / "manifest.json").is_file():
        print(why_empty(rec), file=sys.stderr)
        return 1
    _, chunks = load(rec)
    log = Path(args.log)
    if log.is_file():
        with log.open(encoding="utf-8", errors="replace") as fh:
            attach_places(chunks, fh)
    report = analyse(chunks, args.hot, args.at)
    print(json.dumps(report, indent=1) if args.json else render(report, rec.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
