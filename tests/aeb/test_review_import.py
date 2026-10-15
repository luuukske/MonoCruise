"""Reviewer label import: never over a label, always with the reviewer's handle (AGENTS.md)."""
from __future__ import annotations

from core.aeb.clip_schema import Label
from core.aeb.clip_store import ClipStore
from tests.aeb.test_clip_capture import _make_clip
from tools.aeb_review_import import (
    apply_plan, apply_removal, plan_import, removal_plan, scan_local,
)


def _rec(cid: str, by: str, cls: str = "fp", at: str = "2026-10-15T10:00:00Z", **label) -> dict:
    body = {"class": cls, "severity": 2, "should_trigger": None, "target_vid": 7,
            "desired_peak_decel_ms2": None, "notes": "ghost"}
    body.update(label)
    return {"clip_id": cid, "by": by, "received_at": at, "label": body}


def _store(tmp_path, *ids: str):
    store = ClipStore(root=tmp_path / "clips")
    return store, {cid: store.write(_make_clip(clip_id=cid)) for cid in ids}


def test_a_reviewer_label_is_written_with_its_handle_and_a_backup(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    plan = plan_import([_rec("aaaa0001", "r_aa")], scan_local([store]))
    assert apply_plan(plan, store, tmp_path / "bk") == 1
    label = store.peek_metadata(paths["aaaa0001"]).label
    assert (label.class_, label.target_vid, label.by) == ("fp", 7, "r_aa")
    assert (tmp_path / "bk" / paths["aaaa0001"].name).exists()


def test_lukas_label_is_never_overwritten(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    store.write_label(paths["aaaa0001"], Label(class_="tp", severity=4))
    plan = plan_import([_rec("aaaa0001", "r_aa")], scan_local([store]))
    assert plan.kept == 1 and plan.write == []
    assert apply_plan(plan, store, tmp_path / "bk") == 0
    label = store.peek_metadata(paths["aaaa0001"]).label
    assert (label.class_, label.by) == ("tp", "")


def test_a_label_set_after_planning_is_kept(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    plan = plan_import([_rec("aaaa0001", "r_aa")], scan_local([store]))
    store.write_label(paths["aaaa0001"], Label(class_="tp", severity=4))
    assert apply_plan(plan, store, tmp_path / "bk") == 0
    assert store.peek_metadata(paths["aaaa0001"]).label.class_ == "tp"


def test_records_without_a_reviewer_handle_are_rejected(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    bad = [_rec("aaaa0001", by) for by in ("", "lukas", "R_AA", "r_a", "r_../x")]
    bad += [_rec("aaaa0001", "r_aa", cls="maybe"), _rec("aaaa0001", "r_aa", severity=9)]
    plan = plan_import(bad, scan_local([store]))
    assert plan.rejected == len(bad) and plan.write == []


def test_a_handle_inside_the_label_cannot_pose_as_lukas(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    rec = _rec("aaaa0001", "r_aa")
    rec["label"]["by"] = ""
    plan = plan_import([rec], scan_local([store]))
    apply_plan(plan, store, tmp_path / "bk")
    assert store.peek_metadata(paths["aaaa0001"]).label.by == "r_aa"


def test_disagreeing_reviewers_write_nothing(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    plan = plan_import([_rec("aaaa0001", "r_aa", "fp"), _rec("aaaa0001", "r_bb", "tp")],
                       scan_local([store]))
    assert plan.write == [] and list(plan.disputed) == ["aaaa0001"]


def test_agreeing_reviewers_write_the_first_label(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    recs = [_rec("aaaa0001", "r_bb", at="2026-10-15T11:00:00Z", notes="second"),
            _rec("aaaa0001", "r_aa", at="2026-10-15T10:00:00Z", notes="first")]
    plan = plan_import(recs, scan_local([store]))
    assert len(plan.write) == 1 and plan.write[0][1]["by"] == "r_aa"


def test_a_reviewers_later_save_replaces_their_earlier_one(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001")
    recs = [_rec("aaaa0001", "r_aa", "tp", at="2026-10-15T12:00:00Z"),
            _rec("aaaa0001", "r_aa", "fp", at="2026-10-15T10:00:00Z")]
    plan = plan_import(recs, scan_local([store]))
    assert plan.disputed == {} and plan.write[0][1]["class"] == "tp"


def test_clips_with_no_local_copy_are_reported(tmp_path):
    store, _paths = _store(tmp_path, "aaaa0001")
    plan = plan_import([_rec("bbbb0002", "r_aa")], scan_local([store]))
    assert plan.missing == ["bbbb0002"] and plan.write == []


def test_removal_clears_only_that_reviewers_labels(tmp_path):
    store, paths = _store(tmp_path, "aaaa0001", "bbbb0002", "cccc0003")
    store.write_label(paths["aaaa0001"], Label(class_="fp", severity=2, by="r_aa"))
    store.write_label(paths["bbbb0002"], Label(class_="fp", severity=2, by="r_bb"))
    store.write_label(paths["cccc0003"], Label(class_="tp", severity=2))
    doomed = removal_plan(scan_local([store]), "r_aa")
    assert doomed == [paths["aaaa0001"]]
    assert apply_removal(doomed, "r_aa", store, tmp_path / "bk") == 1
    assert store.peek_metadata(paths["aaaa0001"]).label is None
    assert store.peek_metadata(paths["bbbb0002"]).label.by == "r_bb"
    assert store.peek_metadata(paths["cccc0003"]).label.class_ == "tp"


def test_removal_never_touches_lukas_labels(tmp_path):
    store, paths = _store(tmp_path, "cccc0003")
    store.write_label(paths["cccc0003"], Label(class_="tp", severity=2))
    assert removal_plan(scan_local([store]), "") == []
    assert apply_removal([paths["cccc0003"]], "", store, tmp_path / "bk") == 0
    assert store.peek_metadata(paths["cccc0003"]).label.class_ == "tp"
