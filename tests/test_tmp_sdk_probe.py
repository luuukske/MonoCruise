"""The TMP SDK lag experiment tools agree with the plugin and find what a synthetic drive contains.

See tmp_plugin/README.md, "Experimental vehicle block". A synthetic TruckersMP truck
brakes and stalls once; the analyzer must see the stall, the packet gap and an SDK
speed that leads the radar's.
"""
from __future__ import annotations

import math
import re
import struct
from pathlib import Path

from core.radar import tmp_sdk
from core.radar.reader import _TOTAL_FORMAT
from tools import tmp_sdk_analyze as an
from tools import tmp_sdk_probe as probe

_REPO = Path(__file__).resolve().parents[1]
_FPS = 60.0
_SAMPLE_HZ = 10.0
_STALL = (12.0, 12.45)
_EGO_SPEED = 5.0


def test_the_probe_reads_the_layout_the_plugin_writes():
    src = (_REPO / "tmp_plugin" / "src" / "main.cpp").read_text(encoding="utf-8")
    name = re.search(r'kVehiclesName\[\]\s*=\s*L"([^"]+)"', src).group(1).replace("\\\\", "\\")
    assert name == tmp_sdk.VEHICLES_TAG
    assert re.search(r"kVehiclesVersion\s*=\s*(\d+)", src).group(1) == str(tmp_sdk.VEHICLES_VERSION)
    assert re.search(r"kMaxSlots\s*=\s*(\d+)", src).group(1) == str(tmp_sdk.MAX_SLOTS)
    assert f"sizeof( VehiclesHeader ) == {tmp_sdk.HEADER.size}" in src
    assert f"sizeof( VehicleSlot ) == {tmp_sdk.SLOT.size}" in src


def _truck_speed(t: float) -> float:
    """20 m/s, braking at 4 m/s2 from 5 s to 8 s, then 8 m/s."""
    if t < 5.0:
        return 20.0
    return max(8.0, 20.0 - 4.0 * (t - 5.0))


def _truck_z(t: float) -> float:
    """Integral of ``_truck_speed``, driving toward -z (yaw 0)."""
    if t < 5.0:
        return -20.0 * t
    if t < 8.0:
        return -(100.0 + 20.0 * (t - 5.0) - 2.0 * (t - 5.0) ** 2)
    return -(142.0 + 8.0 * (t - 8.0))


def _traffic(x: float, z: float) -> bytes:
    slot = [x, 1.0, z, 1.0, 0.0, 0.0, 0.0, 2.5, 3.5, 6.0, 0.0, 0.0, 0, 7, 1, 0] + [0.0] * 30
    return struct.pack(_TOTAL_FORMAT, *slot, *([0] * 46 * 39))


def _sdk(hb: int, t: float, bodies: list[tuple]) -> bytes:
    head = tmp_sdk.HEADER.pack(tmp_sdk.VEHICLES_VERSION, hb, int(t * 1e6), 42, len(bodies), 1)
    return head + b"".join(tmp_sdk.SLOT.pack(*b) for b in bodies)


def _drive() -> list[probe.Record]:
    """Ego crawls behind; the truck's packets carry its true speed at 10 Hz and stop during the stall."""
    records = []
    held_z = None
    last_sample_t, updates, sample_v = 0.0, 0, 20.0
    for n in range(int(16.0 * _FPS)):
        t = n / _FPS
        stalled = _STALL[0] <= t < _STALL[1]
        if not stalled and t - last_sample_t >= 1.0 / _SAMPLE_HZ - 1e-9:
            last_sample_t, updates, sample_v = t, updates + 1, _truck_speed(t)
        z = _truck_z(t)
        held_z = held_z if stalled and held_z is not None else z
        shown_z = held_z if stalled else z
        age_ms = (t - last_sample_t) * 1000.0
        ego_z = 150.0 - _EGO_SPEED * t
        truck = (0.0, 1.0, shown_z, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -sample_v, 0.0, 0.0, 0.0,
                 age_ms, updates, ego_z - shown_z, tmp_sdk.KIND_TRACTOR, 1 | 2 | 4 | 8, 1)
        ego = (0.0, 1.0, ego_z, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -_EGO_SPEED, 0.0, 0.0, 0.0,
               -1.0, 0, 0.0, tmp_sdk.KIND_LOCAL_TRACTOR, 1 | 2 | 4, 0)
        records.append(probe.Record(
            t_s=100.0 + t, sim_time_us=int((100.0 + t) * 1e6), paused=False, pose_ok=True,
            ego_x=0.0, ego_y=1.0, ego_z=ego_z, ego_speed=_EGO_SPEED, ego_yaw_norm=0.0,
            ego_pitch=0.0, ego_steer=0.0, traffic=_traffic(0.0, shown_z), parked=None,
            sdk=_sdk(n + 1, 100.0 + t, [ego, truck]),
        ))
    return records


def test_a_recording_survives_the_file_round_trip(tmp_path):
    records = _drive()[:50]
    path = tmp_path / "run.bin"
    with open(path, "wb") as f:
        f.write(probe.FILE_MAGIC)
        for rec in records:
            probe.write_record(f, rec)
        f.write(b"\x01\x02")
    assert list(probe.read_records(path)) == records


def test_the_analyzer_finds_the_synthetic_drive():
    run, lines = an.analyze(_drive())
    text = "\n".join(lines)

    assert list(run.tracks) == [7]
    assert all(s.match_m < 1e-3 for s in run.tracks[7]), "float32 traffic positions"
    ego_err = [sdk - tel for tel, sdk, _ in run.ego]
    assert max(abs(e) for e in ego_err) < 1e-4, "SDK forward axis or sign is wrong"

    rates = [s.updates for s in run.tracks[7]]
    assert 9.0 < (rates[-1] - rates[0]) / 16.0 < 10.5
    assert "changed between samples (no new packet): 0/" in text

    stalls = [ln for ln in lines if ln.startswith("ETS2LA position stalls")]
    assert stalls and stalls[0].endswith(": 1")
    assert "with an SDK gap >250 ms: 1" in text
    assert "SDK sample gaps (age > 250 ms): 1" in text


def test_the_sdk_speed_leads_the_radar_on_the_synthetic_brake():
    run, _ = an.analyze(_drive())
    seg = run.tracks[7]
    level = 14.0
    t_ref = an._crossing(seg, "ref_speed", level, True, 100.0, 110.0)
    t_sdk = an._crossing(seg, "v_h", level, True, 100.0, 110.0)
    t_radar = an._crossing(seg, "radar_speed", level, True, 100.0, 110.0)
    assert not any(math.isnan(t) for t in (t_ref, t_sdk, t_radar))
    assert abs(t_sdk - t_ref) < 0.15
    assert t_radar > t_sdk
