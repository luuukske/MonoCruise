"""Debug gap dump must survive a threat body and never kill the AEB loop.

Regression: the threat-body block bound a local named `clearance` (a float body
separation), shadowing the `ClearanceResult` parameter of the same name. Any
frame with both a threat body and a clearance result raised AttributeError
inside the dump, which crashed `aeb_thread` out of its restart quota.
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clearance import ClearanceResult
from core.aeb.thread import AEBThread
from core.radar.traffic import build_arc, capsule_extents

_EGO_OFFSET = (CAL.arc_start_pctg - 0.5) * (2.0 * CAL.ego_half_length)
_CAP_FWD, _CAP_BACK = capsule_extents(
    CAL.ego_half_length, CAL.ego_half_length, _EGO_OFFSET,
)


class _Size:
    def __init__(self, width: float, length: float) -> None:
        self.width = width
        self.height = 1.5
        self.length = length


class _Threat:
    """Only what the dump reads off a vehicle."""

    id = 237
    is_trailer = False
    is_parked = False
    size = _Size(2.2, 6.0)

    def get_corners(self):
        return ((-1.1, -12.0), (1.1, -12.0), (1.1, -18.0), (-1.1, -18.0))


def _thread() -> AEBThread:
    t = AEBThread.__new__(AEBThread)
    t._engaged = False
    t._gap_debug_was_engaged = False
    t._gap_debug_last_mono = 0.0
    return t


def _ego_arc():
    return build_arc(
        0.0, 0.0, 0.0, 8.7, 0.0, CAL.ego_half_width, 3.0,
        fwd_len=_CAP_FWD, back_len=_CAP_BACK,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )


def _target_arc():
    return build_arc(
        0.0, -15.0, 0.0, 0.0, 0.0, 1.1, 3.0,
        fwd_len=3.0, back_len=3.0,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )


def _dump(thread: AEBThread, clearance, *, now_mono: float = 10.0) -> None:
    threat = _Threat()
    target = _target_arc()
    thread._log_gap_debug(
        now_mono=now_mono,
        cal=CAL,
        ego_x=0.0,
        ego_z=0.0,
        ego_yaw_rad=0.0,
        ego_speed=8.7,
        ego_hw=CAL.ego_half_width,
        ego_half_l=CAL.ego_half_length,
        ego_cap_fwd=_CAP_FWD,
        ego_cap_back=_CAP_BACK,
        ego_front_to_surface=_CAP_FWD + CAL.ego_half_width,
        ego_arc=_ego_arc(),
        vehicles_eff=[threat],
        vehicle_collision_data={threat.id: ([target],)},
        best_threat_vid=threat.id,
        best_ttb=2.4,
        best_unbraked_ttc=2.4,
        best_closing_distance=12.0,
        best_v_closing=8.7,
        best_ego_travel=9.0,
        best_hit_dist=12.0,
        best_codir_cap=4.0,
        required_decel=3.5,
        required_decel_engage=3.5,
        effective_required=3.5,
        effective_required_engage=3.5,
        engage_threshold=8.46,
        effective_max_decel=8.46,
        brake_ttb_active=False,
        brake_ttb_engage_active=False,
        certain_engage=True,
        geom_threat_latched=False,
        latched_distance_threat=False,
        target_raw=0.0,
        target_published=0.0,
        colliding_ids={threat.id},
        response_s=0.3,
        clearance=clearance,
        clearance_vid=threat.id,
    )


def test_threat_body_dump_keeps_clearance_result(caplog):
    """A frame with both a threat body and a clearance result must log both."""
    clearance = ClearanceResult(
        required_ms2=3.5, t_bind_s=1.2, s_bind_m=9.4, v_pass_ms=0.0,
        pad_rate_ms=8.7, clears=False, n_samples=13,
    )
    with caplog.at_level(logging.INFO, logger="core.aeb.thread"):
        _dump(_thread(), clearance)

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    text = caplog.text
    assert "clearance vid=237 req=3.50 s_bind=9.40 t_bind=1.20" in text
    assert "n=13" in text
    # Body separation still reported, under its own name in the same dump.
    assert "threat id=237" in text
    assert "clearance=" in text.split("threat id=237", 1)[1]


def test_dump_failure_does_not_propagate(caplog, monkeypatch):
    """A broken dump is logged, not raised: it must never stop the AEB loop."""
    thread = _thread()

    def _boom(**kwargs):
        raise RuntimeError("dump exploded")

    monkeypatch.setattr(thread, "_log_gap_debug_impl", _boom)
    with caplog.at_level(logging.ERROR, logger="core.aeb.thread"):
        thread._log_gap_debug(now_mono=1.0)

    assert "gap debug dump failed" in caplog.text
