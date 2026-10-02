"""Brake debug CSV: activity window, rate, header rotation. See core/sending_thread/README.md."""
from __future__ import annotations

import csv

from core.sending_thread.debug_csv import (
    BRAKE_LOG_HEADER,
    BRAKE_LOG_NAME,
    BrakeDebugLog,
    open_debug_csv,
)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_logs_while_braking_and_for_the_tail_only(tmp_path):
    log = BrakeDebugLog(tmp_path)
    t = 100.0
    log.tick(t, False, {"sent_brake": 0.0})           # idle: nothing
    for _ in range(20):                                # 1 s braking at 100 Hz
        t += 0.01
        log.tick(t, True, {"sent_brake": 0.3, "controller": "cc", "aeb_active": False})
    for _ in range(300):                               # 3 s released: 2 s tail
        t += 0.01
        log.tick(t, False, {"sent_brake": 0.0})
    log.close()

    rows = _rows(tmp_path / BRAKE_LOG_NAME)
    assert rows[0] == BRAKE_LOG_HEADER
    body = rows[1:]
    # 20 Hz over 0.2 s braking + 2.0 s tail, give or take a sample at each edge.
    assert 40 <= len(body) <= 48
    first = dict(zip(BRAKE_LOG_HEADER, body[0]))
    assert first["sent_brake"] == "0.3000"
    assert first["controller"] == "cc"
    assert first["aeb_active"] == "0"


def test_a_changed_header_rotates_the_old_file(tmp_path):
    path = tmp_path / BRAKE_LOG_NAME
    path.write_text("t_s,old_column\n1,2\n", encoding="utf-8")
    fh, writer = open_debug_csv(path, BRAKE_LOG_HEADER)
    fh.close()
    assert _rows(path)[0] == BRAKE_LOG_HEADER
    rotated = [p for p in tmp_path.iterdir() if p.name != BRAKE_LOG_NAME]
    assert len(rotated) == 1 and _rows(rotated[0])[0] == ["t_s", "old_column"]


def test_an_unwritable_root_never_raises(tmp_path):
    log = BrakeDebugLog(tmp_path / "missing" / "dir")
    log.tick(1.0, True, {"sent_brake": 0.5})
    log.tick(2.0, True, {"sent_brake": 0.5})
    log.close()
