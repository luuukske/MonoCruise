"""Ego pose sampled in the traffic buffer's game frame. See core/radar/README.md section 16."""
from __future__ import annotations

import struct

from core.radar import scs_pose
from core.radar.scs_pose import ScsPose, layout_supported, library_agrees, pose_from_buffer
from core.radar.thread import RadarThread

_BLOCK = 21600


def _block(revision: int = 12, active: bool = True, **fields) -> bytearray:
    buf = bytearray(_BLOCK)
    struct.pack_into("<I", buf, 40, revision)
    values = {
        "sdkActive": active, "paused": False, "simulatedTime": 123_450_000,
        "speed": 21.5, "coordinateX": -64875.889, "coordinateY": 12.25,
        "coordinateZ": 3019.5, "rotationX": 0.375, "rotationY": 0.0625,
    }
    values.update(fields)
    for name, value in values.items():
        offset, packer = scs_pose._FIELDS[name]
        packer.pack_into(buf, offset, value)
    return buf


def test_pose_is_decoded_from_the_documented_offsets():
    pose = pose_from_buffer(_block())
    assert pose == ScsPose(
        simulated_time_us=123_450_000, paused=False, x=-64875.889, y=12.25, z=3019.5,
        yaw_norm=0.375, pitch_raw=0.0625, speed=21.5,
    )


def test_inactive_sdk_yields_no_pose():
    assert pose_from_buffer(_block(active=False)) is None


def test_only_known_plugin_revisions_are_trusted():
    assert layout_supported(_block(revision=12))
    assert layout_supported(_block(revision=10))
    assert not layout_supported(_block(revision=11))
    assert not layout_supported(bytearray(64))


def test_offsets_agree_with_truck_telemetry_when_it_is_installed():
    """CI has no truck_telemetry and gets None; any machine that has it must agree."""
    assert library_agrees(_block()) in (True, None)
    assert library_agrees(_block(revision=11)) in (False, None)


class _FakePose:
    def __init__(self, sims: list[int], pose_fields: dict | None = None) -> None:
        self._sims = list(sims)
        self._fields = pose_fields or {}
        self.reads = 0

    def simulated_time_us(self) -> int | None:
        return self._sims.pop(0)

    def read(self) -> ScsPose | None:
        self.reads += 1
        sim = self._sims.pop(0)
        base = dict(paused=False, x=1.0, y=2.0, z=3.0, yaw_norm=0.25, pitch_raw=0.0, speed=20.0)
        base.update(self._fields)
        return ScsPose(simulated_time_us=sim, **base)

    def close(self) -> None:
        pass


class _FakeTraffic:
    def __init__(self) -> None:
        self.copies = 0

    def copy_raw(self):
        self.copies += 1
        return (b"t%d" % self.copies, None)


def test_a_frame_boundary_between_the_reads_is_retried_once():
    rt = RadarThread()
    rt._pose = _FakePose([100, 116_667, 116_667, 116_667])
    rt._traffic = _FakeTraffic()
    raw, pose = rt._sample_traffic_and_pose()
    assert rt._traffic.copies == 2
    assert raw == (b"t2", None) and pose.simulated_time_us == 116_667


def test_a_stable_frame_reads_once():
    rt = RadarThread()
    rt._pose = _FakePose([100, 100])
    rt._traffic = _FakeTraffic()
    raw, pose = rt._sample_traffic_and_pose()
    assert rt._traffic.copies == 1 and pose.simulated_time_us == 100


def test_loop_publishes_the_paired_pose_not_the_telemetry_thread_copy(monkeypatch):
    rt = RadarThread()
    stale = (9.0, 9.0, 9.0, 0.5, 5.0, 0.1, False, False, 0.0, 50, 20000.0, 6, 1)
    monkeypatch.setattr(rt, "_read_ego", lambda: stale)
    pose = ScsPose(simulated_time_us=66_667, paused=False, x=1.0, y=2.0, z=3.0,
                   yaw_norm=0.25, pitch_raw=0.0, speed=20.0)
    monkeypatch.setattr(rt, "_sample_traffic_and_pose", lambda: (None, pose))
    rt.running = True
    rt.loop()
    with rt.data._lock:
        assert (rt.data.ego_x, rt.data.ego_y, rt.data.ego_z) == (1.0, 2.0, 3.0)
        assert rt.data.ego_speed == 20.0 and rt.data.ego_yaw_norm == 0.25
        assert rt.data.ego_steer == 0.1
    assert rt._ego_position_history[-1] == (0.066667, 1.0, 3.0)


def test_loop_falls_back_to_the_telemetry_thread_pose_without_the_block(monkeypatch):
    rt = RadarThread()
    stale = (9.0, 8.0, 7.0, 0.5, 5.0, 0.1, False, False, 0.0, 50, 20000.0, 6, 1)
    monkeypatch.setattr(rt, "_read_ego", lambda: stale)
    monkeypatch.setattr(rt, "_sample_traffic_and_pose", lambda: (None, None))
    rt.running = True
    rt.loop()
    with rt.data._lock:
        assert (rt.data.ego_x, rt.data.ego_y, rt.data.ego_z) == (9.0, 8.0, 7.0)
        assert rt.data.ego_speed == 5.0
