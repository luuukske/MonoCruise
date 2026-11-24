"""tools/profile_drive.py records py-spy chunks on the log's clock; tools/profile_report.py says what grew.

py-spy itself is never run here: the recorder takes a stand-in, the report reads synthetic chunks.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from tools import profile_drive as pd
from tools import profile_report as pr

_ROOT = Path(__file__).resolve().parents[1]
_T0 = datetime(2026, 11, 23, 1, 15, 50)


def _frame(func: str, rel: str, line: int = 1) -> str:
    return f"{func} ({_ROOT / rel}:{line})"


def _stack(thread: str, *frames: str, n: int) -> str:
    return ";".join([f"thread (0x{abs(hash(thread)) % 65536:x}): {thread}", *frames]) + f" {n}"


_LOOP = _frame("BaseThread._run_lifecycle", "core/thread_management/base_thread.py", 270)
_AEB = _frame("AEBThread.loop", "core/aeb/thread.py", 900)
_CLEAR = _frame("occupancy_profile", "core/aeb/clearance.py", 120)
_MAP = _frame("MapThread._target_routes", "core/map_thread/thread.py", 300)


def _chunk(aeb_clearance: int, aeb_other: int, map_n: int) -> list[str]:
    return [
        _stack("aeb_thread", _LOOP, _AEB, _CLEAR, n=aeb_clearance),
        _stack("aeb_thread", _LOOP, _AEB, n=aeb_other),
        _stack("map_thread", _LOOP, _MAP, n=map_n),
        _stack("MainThread", _frame("main", "monocruise.py", 627), n=40),
    ]


def _recording(tmp_path: Path, chunks: list[list[str]], last_wall_s: float | None = None) -> Path:
    rec = tmp_path / "20261123_011550"
    rec.mkdir()
    manifest = {"rate_hz": 100, "chunk_s": 10, "chunks": []}
    for i, lines in enumerate(chunks):
        start = _T0 + timedelta(seconds=10 * i)
        wall = 9.4 if last_wall_s is None or i < len(chunks) - 1 else last_wall_s
        name = f"{start:%H%M%S}.txt"
        (rec / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["chunks"].append({
            "file": name, "start": start.isoformat(timespec="milliseconds"),
            "end": (start + timedelta(seconds=wall)).isoformat(timespec="milliseconds"),
        })
    (rec / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rec


def _drive(tmp_path: Path) -> Path:
    calm = [_chunk(20, 150, 50) for _ in range(4)]
    return _recording(tmp_path, [*calm, _chunk(450, 200, 250), _chunk(30, 150, 60)])


def test_a_chunk_samples_only_the_gil_holder_without_pausing_monocruise():
    args = pd.pyspy_args("py-spy", 4242, Path("out.txt"), 100, 10)
    assert args[:2] == ["py-spy", "record"]
    for flag in ("--gil", "--nonblocking", "--threads", "--full-filenames"):
        assert flag in args
    assert args[args.index("--pid") + 1] == "4242" and args[args.index("--format") + 1] == "raw"
    assert args[args.index("--duration") + 1] == "10"


def test_the_recorder_writes_a_manifest_on_the_wall_clock(tmp_path):
    left = [2]

    def alive() -> bool:
        left[0] -= 1
        return left[0] >= 0

    def fake_run(args, **_):
        Path(args[args.index("--output") + 1]).write_text(_stack("aeb_thread", _AEB, n=5), encoding="utf-8")
        return type("Done", (), {"stdout": "", "stderr": ""})()

    out = tmp_path / "rec"
    pd.record(4242, alive, out, "py-spy", 100, 1, run=fake_run)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rate_hz"] == 100 and len(manifest["chunks"]) >= 1
    first = manifest["chunks"][0]
    assert (out / first["file"]).is_file()
    assert datetime.fromisoformat(first["start"]) <= datetime.fromisoformat(first["end"])


def test_the_recorder_stops_after_repeated_py_spy_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(pd.time, "sleep", lambda _s: None)
    calls = []

    def failing(args, **_):
        calls.append(args)
        return type("Done", (), {"stdout": "", "stderr": "Error: permission denied"})()

    manifest = pd.record(4242, lambda: True, tmp_path / "rec", "py-spy", 100, 1, run=failing)
    assert manifest["chunks"] == [] and len(calls) == 3


def test_a_virtualenv_launcher_hands_over_its_python_child():
    kids = {100: [(150, "conhost.exe"), (200, "python.exe")]}
    assert pd.interpreter_pid(100, children=lambda pid: kids.get(pid, []), wait_s=0.0) == 200
    assert pd.interpreter_pid(300, children=lambda pid: kids.get(pid, []), wait_s=0.0) == 300


def test_a_failed_recording_says_why_instead_of_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pd.time, "sleep", lambda _s: None)

    def failing(args, **_):
        return type("Done", (), {"stdout": "", "stderr": "Error: Failed to find python version from target process"})()

    rec = tmp_path / "20261123_014933"
    pd.record(4242, lambda: True, rec, "py-spy", 100, 1, run=failing)
    capsys.readouterr()
    assert pr.main([str(rec)]) == 1
    err = capsys.readouterr().err
    assert "has no chunks" in err and "Failed to find python version" in err
    assert pr.latest_recording(tmp_path) == rec


def test_subprocess_prefixed_stacks_still_parse():
    line = 'process 55248:"python.exe target.py";' + _stack("aeb_thread", _AEB, n=7)
    threads = pr.parse_raw([line], pr._Shortener(_ROOT))
    assert threads["aeb_thread"].samples == 7


def test_monocruise_runs_on_the_repo_virtualenv_when_there_is_one(tmp_path):
    assert pd.monocruise_python(tmp_path) == pd.sys.executable
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_text("", encoding="utf-8")
    assert pd.monocruise_python(tmp_path).endswith("python.exe")


def test_what_grew_in_the_hot_chunk_ranks_first(tmp_path):
    _, chunks = pr.load(_drive(tmp_path))
    report = pr.analyse(chunks, hot_n=1)
    assert report["hot"][0]["start"] == "01:16:30"
    own = report["grew"]["own"][0]
    assert (own["thread"], own["function"]) == ("aeb_thread", "core/aeb/clearance.py:occupancy_profile")
    assert own["hot_pct"] == 45.0 and own["calm_pct"] == 2.0
    assert report["grew"]["threads"][0]["thread"] == "aeb_thread"
    inclusive = {r["function"] for r in report["grew"]["inclusive"]}
    assert not any("base_thread.py" in f for f in inclusive), "the loop runner is a wrapper"


def test_the_report_prints_no_absolute_path_and_names_the_gui(tmp_path):
    _, chunks = pr.load(_drive(tmp_path))
    text = pr.render(pr.analyse(chunks, hot_n=1), "rec")
    assert not re.search(r"[A-Za-z]:[\\/]", text)
    assert "gui" in text and "MainThread" not in text
    assert "core/aeb/clearance.py:occupancy_profile" in text


def test_a_chunk_cut_short_counts_against_its_own_length(tmp_path):
    rec = _recording(tmp_path, [_chunk(100, 100, 100), _chunk(100, 100, 100), _chunk(10, 10, 10)], last_wall_s=0.94)
    _, chunks = pr.load(rec)
    assert chunks[0].expected == 1000.0
    assert abs(chunks[-1].expected - 100.0) < 1.0


def test_at_picks_the_chunk_holding_a_log_time(tmp_path):
    _, chunks = pr.load(_drive(tmp_path))
    assert pr.analyse(chunks, at="01:16:05")["detail"]["start"] == "01:16:00"
    assert "error" in pr.analyse(chunks, at="03:00:00")


def test_chunks_get_their_place_from_the_stall_probe(tmp_path):
    _, chunks = pr.load(_drive(tmp_path))
    log = [
        "01:16:39 [gui_stall   ] DEBUG    GUI longest gap 458 ms in the last 10 s. CPU 9938 ms over 10000 ms: "
        "aeb_thread 3094 ms (31%). At x -600 z 2400, 0 km/h, TMP, 24 vehicles + 2 trailers, map ready 208 lanes 12 routed",
    ]
    pr.attach_places(chunks, log)
    hot = next(c for c in chunks if c.start.strftime("%H:%M:%S") == "01:16:30")
    assert hot.place.startswith("process 99 %, x -600 z 2400") and "12 routed" in hot.place
    assert chunks[0].place is None


def test_the_cli_reads_the_latest_recording(tmp_path, capsys):
    rec = _drive(tmp_path)
    assert pr.main([str(rec), "--log", str(tmp_path / "missing.log"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] == 6
    assert pr.latest_recording(tmp_path) == rec
