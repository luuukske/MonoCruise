"""Record a drive with py-spy in 10 s chunks on the log's clock, for tools/profile_report.py.

Starts MonoCruise from source (or attaches to one with ``--pid``) and samples only the thread
holding the GIL, so each chunk says which Python code used the one core the process has. py-spy
reads the process from outside and never pauses it (``--nonblocking``). Needs ``pip install py-spy``.

    python tools/profile_drive.py              # start MonoCruise, record until it closes
    python tools/profile_drive.py --pid 1234   # attach to a MonoCruise already running

Close MonoCruise normally to end the recording; Ctrl+C in this console reaches MonoCruise too.
Chunks and ``manifest.json`` go to ``tools/profile_out/<start>/`` (gitignored). See tools/README.md.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "profile_out"
CHUNK_S = 10
RATE_HZ = 100
# Consecutive chunks py-spy may fail before the recording stops.
_MAX_FAILURES = 3


def pyspy_args(exe: str, pid: int, out: Path, rate: int, seconds: int) -> list[str]:
    """One chunk: GIL holder only, raw collapsed stacks with full file names, never pausing MonoCruise."""
    return [
        exe, "record", "--pid", str(pid), "--format", "raw", "--rate", str(rate), "--threads", "--gil",
        "--nonblocking", "--full-filenames", "--duration", str(seconds), "--output", str(out),
    ]


def monocruise_python(root: Path = ROOT) -> str:
    """The repo's own virtualenv when there is one (MonoCruise's packages live there), else this Python."""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        exe = root / rel
        if exe.is_file():
            return str(exe)
    return sys.executable


def child_processes(pid: int) -> list[tuple[int, str]]:
    """(pid, exe name) of the processes ``pid`` started; empty off Windows or when unreadable."""
    if sys.platform != "win32":
        return []

    class _Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong), ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long), ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    if snap in (None, ctypes.c_void_p(-1).value):
        return []
    out = []
    try:
        entry = _Entry()
        entry.dwSize = ctypes.sizeof(_Entry)
        ok = k32.Process32FirstW(ctypes.c_void_p(snap), ctypes.byref(entry))
        while ok:
            if entry.th32ParentProcessID == pid:
                out.append((int(entry.th32ProcessID), entry.szExeFile))
            ok = k32.Process32NextW(ctypes.c_void_p(snap), ctypes.byref(entry))
    finally:
        k32.CloseHandle(ctypes.c_void_p(snap))
    return out


def interpreter_pid(pid: int, children=child_processes, wait_s: float = 5.0) -> int:
    """The Python interpreter behind ``pid``. A Windows virtualenv's python.exe is a launcher that
    runs the real interpreter as its child, and py-spy finds no Python in the launcher itself."""
    deadline = time.monotonic() + wait_s
    while True:
        pythons = [cpid for cpid, name in children(pid) if name.lower().startswith("python")]
        if pythons:
            return pythons[0]
        if time.monotonic() >= deadline:
            return pid
        time.sleep(0.2)


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(0x1000, 0, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def record(pid: int, alive, out_dir: Path, exe: str, rate: int = RATE_HZ, chunk_s: int = CHUNK_S, run=subprocess.run) -> dict:
    """Chunks back to back while ``alive()``; the manifest is rewritten after each so a crash keeps what was recorded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"rate_hz": rate, "chunk_s": chunk_s, "chunks": []}
    failures = 0
    while alive() and failures < _MAX_FAILURES:
        start = datetime.now()
        path = out_dir / f"{len(manifest['chunks']) + failures:04d}_{start:%H%M%S}.txt"
        done = run(pyspy_args(exe, pid, path, rate, chunk_s), capture_output=True, text=True)
        end = datetime.now()
        if path.is_file() and path.stat().st_size > 0:
            failures = 0
            manifest["chunks"].append({
                "file": path.name,
                "start": start.isoformat(timespec="milliseconds"),
                "end": end.isoformat(timespec="milliseconds"),
            })
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        elif alive():
            failures += 1
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or ["no output"]
            print(f"py-spy chunk failed ({failures}/{_MAX_FAILURES}): {tail[0]}", file=sys.stderr)
            (out_dir / "error.txt").write_text(f"py-spy failed on pid {pid}: {tail[0]}\n", encoding="utf-8")
            time.sleep(1.0)
    if failures >= _MAX_FAILURES:
        print("RECORDING STOPPED: py-spy keeps failing, MonoCruise runs on unrecorded", file=sys.stderr)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pid", type=int, help="attach to a running MonoCruise instead of starting one")
    ap.add_argument("--rate", type=int, default=RATE_HZ, help=f"samples per second (default {RATE_HZ})")
    ap.add_argument("--chunk", type=int, default=CHUNK_S, help=f"seconds per chunk (default {CHUNK_S})")
    ap.add_argument("--python", help="interpreter for MonoCruise (default: the repo's .venv, else this one)")
    args, extra = ap.parse_known_args(argv)
    exe = shutil.which("py-spy")
    if exe is None:
        print("py-spy not found: pip install py-spy", file=sys.stderr)
        return 1
    out_dir = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
    child = None
    if args.pid is None:
        python = args.python or monocruise_python()
        child = subprocess.Popen([python, str(ROOT / "monocruise.py"), *extra], cwd=str(ROOT))
        time.sleep(1.0)
        pid = interpreter_pid(child.pid)
    else:
        pid = interpreter_pid(args.pid, wait_s=0.0)

    def alive() -> bool:
        return child.poll() is None if child is not None else pid_alive(pid)

    print(f"recording MonoCruise (interpreter pid {pid}) into tools/profile_out/{out_dir.name}, {args.chunk} s chunks")
    try:
        record(pid, alive, out_dir, exe, args.rate, args.chunk)
    except KeyboardInterrupt:
        print("recording stopped", file=sys.stderr)
    if child is not None:
        child.wait()
    manifest = out_dir / "manifest.json"
    chunks = len(json.loads(manifest.read_text(encoding="utf-8"))["chunks"]) if manifest.is_file() else 0
    print(f"{chunks} chunks. Next: python tools/profile_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
