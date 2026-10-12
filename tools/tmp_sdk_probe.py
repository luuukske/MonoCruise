"""TruckersMP SDK vehicle motion next to ETS2LA traffic: live view and recorder.

Experiment for TMP lag detection (tmp_plugin/README.md, "Experimental vehicle
block"). Needs ``monocruise_tmp.dll`` 1.0.3 or newer installed, the game running
on TruckersMP, and ETS2LA's traffic plugin. Nothing here ships in the app.

    python tools/tmp_sdk_probe.py live
    python tools/tmp_sdk_probe.py record            # Ctrl+C to stop
    python tools/tmp_sdk_analyze.py <recording>

Recordings go to ``tools/tmp_sdk_runs/`` (gitignored) and hold positions and
motion only: no player identity, no latency. Never imports ``core.settings``.
"""

from __future__ import annotations

import argparse
import logging
import math
import struct
import sys
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.radar.reader import TrafficReader  # noqa: E402
from core.radar.scs_pose import ScsPose, ScsPoseReader  # noqa: E402
from core.radar.tmp_sdk import (  # noqa: E402
    HEADER, MAX_SLOTS, SLOT, SdkBody, TmpSdkReader, decode_frame, qpc_s,
)

RUNS_DIR = Path(__file__).resolve().parent / "tmp_sdk_runs"
FILE_MAGIC = b"MCTSDK01"
RECORD = struct.Struct("<dQBBHdddffff")

_PAIRED_READ_ATTEMPTS = 3


def trimmed(raw: bytes) -> bytes:
    """Header plus used slots only, for recording."""
    count = min(HEADER.unpack_from(raw, 0)[4], MAX_SLOTS)
    return raw[:HEADER.size + count * SLOT.size]


@dataclass(frozen=True)
class Record:
    """One recorded poll: the probe's QPC time, the paired ego pose and both buffers."""

    t_s: float
    sim_time_us: int
    paused: bool
    pose_ok: bool
    ego_x: float
    ego_y: float
    ego_z: float
    ego_speed: float
    ego_yaw_norm: float
    ego_pitch: float
    ego_steer: float
    traffic: bytes
    parked: bytes | None
    sdk: bytes


def _write_blob(f, blob: bytes | None) -> None:
    data = b"" if blob is None else zlib.compress(blob, 6)
    f.write(struct.pack("<I", len(data)))
    f.write(data)


def write_record(f, rec: Record) -> None:
    f.write(RECORD.pack(
        rec.t_s, rec.sim_time_us, int(rec.paused), int(rec.pose_ok), 0,
        rec.ego_x, rec.ego_y, rec.ego_z,
        rec.ego_speed, rec.ego_yaw_norm, rec.ego_pitch, rec.ego_steer,
    ))
    _write_blob(f, rec.traffic)
    _write_blob(f, rec.parked)
    _write_blob(f, rec.sdk)


def _read_blob(f) -> bytes | None:
    head = f.read(4)
    if len(head) < 4:
        raise EOFError
    n = struct.unpack("<I", head)[0]
    data = f.read(n)
    if len(data) < n:
        raise EOFError
    return zlib.decompress(data) if n else None


def read_records(path: Path) -> Iterator[Record]:
    """Every complete record in a recording; a truncated tail (killed recorder) is dropped."""
    with open(path, "rb") as f:
        if f.read(len(FILE_MAGIC)) != FILE_MAGIC:
            raise ValueError(f"{path.name} is not a tmp_sdk_probe recording")
        while True:
            fixed = f.read(RECORD.size)
            if len(fixed) < RECORD.size:
                return
            try:
                traffic, parked, sdk = _read_blob(f), _read_blob(f), _read_blob(f)
            except (EOFError, zlib.error):
                return
            (t_s, sim_us, paused, pose_ok, _pad, ex, ey, ez,
             speed, yaw, pitch, steer) = RECORD.unpack(fixed)
            yield Record(
                t_s, sim_us, bool(paused), bool(pose_ok), ex, ey, ez,
                speed, yaw, pitch, steer, traffic or b"", parked, sdk or b"",
            )


class Sampler:
    """Traffic plus the ego pose of the same game frame, the way RadarThread pairs them."""

    def __init__(self) -> None:
        self.traffic = TrafficReader()
        self.pose = ScsPoseReader()
        self.vehicles = TmpSdkReader()
        self._pose_retry_at = 0.0
        logging.getLogger("core.radar.scs_pose").setLevel(logging.ERROR)

    def _revive_pose(self) -> None:
        """ScsPoseReader gives up for good on a block the game has not filled yet; retry every 5 s."""
        now = time.monotonic()
        if self.pose._shm is None and self.pose._retry_at == math.inf and now >= self._pose_retry_at:
            self.pose = ScsPoseReader()
            self._pose_retry_at = now + 5.0

    def sample(self) -> tuple[tuple[bytes, bytes | None] | None, ScsPose | None, bytes | None]:
        self._revive_pose()
        raw, pose = None, None
        for _ in range(_PAIRED_READ_ATTEMPTS):
            sim_before = self.pose.simulated_time_us()
            raw = self.traffic.copy_raw()
            pose = self.pose.read()
            if pose is None or sim_before is None or pose.simulated_time_us == sim_before:
                break
        return raw, pose, self.vehicles.read_raw()

    def close(self) -> None:
        self.traffic.close()
        self.pose.close()
        self.vehicles.close()


def _record(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else RUNS_DIR / time.strftime("tmp_sdk_%Y%m%d_%H%M%S.bin")
    out.parent.mkdir(parents=True, exist_ok=True)
    sampler = Sampler()
    min_dt = 1.0 / args.hz
    last_t, last_hb, last_sim = -1.0, None, None
    frames = no_sdk = 0
    warned_pose = False
    t0 = time.monotonic()
    print(f"recording to {out.name} at up to {args.hz:.0f} Hz; Ctrl+C to stop")
    try:
        with open(out, "wb") as f:
            f.write(FILE_MAGIC)
            while True:
                time.sleep(0.002)
                now = qpc_s()
                if now - last_t < min_dt:
                    continue
                raw, pose, sdk = sampler.sample()
                if raw is None:
                    continue
                if pose is None and not warned_pose:
                    print("no SCS pose yet (game not running or still loading); recording without ego pose")
                    warned_pose = True
                hb = struct.unpack_from("<I", sdk, 4)[0] if sdk else None
                sim = pose.simulated_time_us if pose else None
                if hb == last_hb and sim == last_sim:
                    continue
                last_t, last_hb, last_sim = now, hb, sim
                no_sdk += sdk is None
                write_record(f, Record(
                    now, sim or 0, bool(pose.paused) if pose else False, pose is not None,
                    pose.x if pose else 0.0, pose.y if pose else 0.0, pose.z if pose else 0.0,
                    pose.speed if pose else 0.0, pose.yaw_norm if pose else 0.0,
                    pose.pitch_raw if pose else 0.0, pose.steer if pose else 0.0,
                    raw[0], raw[1], trimmed(sdk) if sdk else b"",
                ))
                frames += 1
                if frames % (int(args.hz) * 10) == 0:
                    f.flush()
                    mb = out.stat().st_size / 1e6
                    print(f"{time.monotonic() - t0:7.0f} s  {frames} frames  {mb:.0f} MB  "
                          f"frames without SDK data: {no_sdk}")
    except KeyboardInterrupt:
        pass
    finally:
        sampler.close()
    print(f"stopped: {frames} frames in {out}")
    return 0


def _match_prev(s: SdkBody, prev: list[tuple[int, float, float, float, int]]):
    """Previous sample of the same body: same kind, nearest within 5 m."""
    best, best_d = None, 5.0
    for p in prev:
        d = math.hypot(s.x - p[1], s.z - p[2])
        if p[0] == s.kind and d < best_d:
            best, best_d = p, d
    return best


def _live(args: argparse.Namespace) -> int:
    reader = TmpSdkReader()
    prev: list[tuple[int, float, float, float, int]] = []
    try:
        while True:
            raw = reader.read_raw()
            frame = decode_frame(raw, qpc_s()) if raw else None
            if frame is None:
                print("no SDK vehicle data (plugin 1.0.3 not loaded, or not on TruckersMP)")
                time.sleep(1.0)
                continue
            t = frame.read_s
            print()
            print(f"connected={int(frame.connected)}  bodies={len(frame.bodies)}")
            print("  rig kind   dist   sdk km/h  long km/h  pos km/h  age ms  upd/s")
            for s in frame.bodies[:args.rows]:
                pos_kmh = upd_hz = float("nan")
                p = _match_prev(s, prev)
                if p is not None and t - p[3] > 1e-3:
                    pos_kmh = math.hypot(s.x - p[1], s.z - p[2]) / (t - p[3]) * 3.6
                    upd_hz = (s.updates - p[4]) / (t - p[3])
                kind = ("truck", "trail", "EGO", "EGO-t")[s.kind]
                age = f"{s.age_ms:6.0f}" if s.sampled else "     -"
                print(f"  {s.rig:3d} {kind:5s} {s.distance_m:6.1f} {s.speed_h * 3.6:9.1f} "
                      f"{s.speed_long() * 3.6:10.1f} {pos_kmh:9.1f} {age} {upd_hz:6.1f}")
            prev = [(s.kind, s.x, s.z, t, s.updates) for s in frame.bodies]
            time.sleep(args.period)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record SDK + traffic + ego pose to tools/tmp_sdk_runs/")
    rec.add_argument("--hz", type=float, default=60.0, help="max record rate (default 60)")
    rec.add_argument("--out", help="output file (default: timestamped in tools/tmp_sdk_runs/)")
    live = sub.add_parser("live", help="print the nearest SDK bodies")
    live.add_argument("--rows", type=int, default=10)
    live.add_argument("--period", type=float, default=0.5)
    args = ap.parse_args()
    return _record(args) if args.cmd == "record" else _live(args)


if __name__ == "__main__":
    raise SystemExit(main())
