"""ACC debug log: every column filled while following, the law terms are the law's own, no ids.
See core/cruise_control_thread/README.md, "ACC debug log"."""
from __future__ import annotations

import csv
import math
from types import SimpleNamespace

from core.cruise_control_thread import acc_controller, idm_cah
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.acc_debug import ACC_LOG_HEADER, AccDebugLog, acc_debug_row
from core.cruise_control_thread.blinker_arbitration import BlinkerState
from core.longitudinal import acc as acc_child
from core.sending_thread.debug_csv import drain_debug_csv
from core.settings import Settings

DT = 1.0 / 60.0


def _follow(monkeypatch, ctrl, leads, ticks=120, v=22.0):
    clock = [100.0]
    monkeypatch.setattr(acc_controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ctrl, "_read_acc_snapshot", lambda: (leads, None, BlinkerState()))
    cap = 0.0
    for _ in range(ticks):
        clock[0] += DT
        cap = ctrl.accel_cap_ms2(v, 1.5)
    return cap


def test_a_following_row_fills_every_column_from_the_law(monkeypatch):
    ctrl = AdaptiveCruiseController()
    leads = [_LeadSnapshot(vid=7, dist_m=60.0, v_lead_ms=21.0, a_lead_ms2=-0.4, score=6.0),
             _LeadSnapshot(vid=8, dist_m=85.0, v_lead_ms=20.0, a_lead_ms2=-0.6, score=6.0),
             _LeadSnapshot(vid=9, dist_m=110.0, v_lead_ms=18.0, a_lead_ms2=-1.2, score=6.0)]
    cap = _follow(monkeypatch, ctrl, leads)
    row = acc_debug_row(ctrl, cap, 1.5, 22.0, prev_vid=7)
    assert row.pop("_vid") == 7
    missing = [k for k in ACC_LOG_HEADER if k not in ("t_s", "utc") and k not in row]
    assert not missing, missing
    assert row["chain_len"] == 3 and row["lead_changed"] is False and row["lead_grace"] is False
    assert row["ant_member"] in (1, 2) and 0.0 <= row["ant_confirm"] <= 1.0
    t_h = row["headway_s"]
    law = idm_cah.lead_law(ctrl.config, row["dist_m"], 22.0, row["v_lead_ms"], row["a_lead_ms2"], t_h,
                           row["a_lead_ff_ms2"])
    assert row["law_ms2"] == law
    assert all(math.isfinite(row[k]) for k in ("iidm_ms2", "cah_ms2", "followed_ms2", "law_out_ms2", "need_ms2"))


def test_a_new_lead_is_a_flag_and_never_an_id(monkeypatch):
    ctrl = AdaptiveCruiseController()
    cap = _follow(monkeypatch, ctrl, [_LeadSnapshot(vid=9, dist_m=40.0, v_lead_ms=22.0, a_lead_ms2=0.0, score=6.0)])
    row = acc_debug_row(ctrl, cap, 1.5, 22.0, prev_vid=7)
    assert row["lead_changed"] is True
    assert not any("id" in k.split("_") or k in ("vid", "name", "player") for k in ACC_LOG_HEADER)


def test_no_lead_writes_a_short_row(monkeypatch):
    ctrl = AdaptiveCruiseController()
    cap = _follow(monkeypatch, ctrl, [])
    row = acc_debug_row(ctrl, cap, 1.5, 22.0, prev_vid=None)
    assert row["chain_len"] == 0 and "law_ms2" not in row


def test_the_log_writes_only_with_debug_on(monkeypatch, tmp_path):
    monkeypatch.setattr(acc_child, "Path", lambda *_: tmp_path / "core" / "longitudinal" / "acc.py")
    monkeypatch.setattr(Settings.instance(), "acc_enabled", True)
    monkeypatch.setattr(Settings.instance(), "cc_mode", "Cruise control")
    child = acc_child.AdaptiveCruiseController()
    lead = _LeadSnapshot(vid=3, dist_m=30.0, v_lead_ms=20.0, a_lead_ms2=-0.5, score=6.0)
    monkeypatch.setattr(child._inner, "_read_acc_snapshot", lambda: ([lead], None, BlinkerState()))
    clock = [100.0]
    monkeypatch.setattr(acc_controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(acc_child.time, "monotonic", lambda: clock[0])
    ctx = SimpleNamespace(speed_ms=22.0)
    for debug in (False, True):
        monkeypatch.setattr(Settings.instance(), "debug", debug)
        for _ in range(60):
            clock[0] += DT
            child.step(ctx, cruise_bid_ms2=1.0)
    assert drain_debug_csv()
    path = tmp_path / "acc_debug.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ACC_LOG_HEADER
    assert 15 <= len(rows) - 1 <= 25, "20 Hz for the second with debug on, nothing before"
    AccDebugLog(tmp_path).close()
