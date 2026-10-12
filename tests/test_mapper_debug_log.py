"""Mapper debug CSV: debug-only, header-only open, size rotation. See core/sending_thread/README.md."""
from __future__ import annotations

import csv
import inspect

import pytest

import core.sending_thread.accel_to_pedals as atp
from core.sending_thread.accel_to_pedals import AccelToPedals
from core.settings import Settings

HEADER = atp._DEBUG_LOG_HEADER_ROW
HEADER_LINE = ",".join(HEADER) + "\n"


@pytest.fixture
def debug_on(monkeypatch):
    monkeypatch.setattr(Settings.instance(), "debug", True)


def _mapper(root) -> AccelToPedals:
    mapper = AccelToPedals()
    mapper._project_root = root
    return mapper


def _log(mapper: AccelToPedals, now: float) -> None:
    """One debug step with every numeric field at 0, robust to new columns."""
    params = inspect.signature(mapper._log_debug_step).parameters
    kwargs = {name: 0.0 for name in params}
    kwargs.update(now=now, gear=0, pedal_state=0)
    mapper._log_debug_step(**kwargs)


def _rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_debug_off_never_creates_the_log(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings.instance(), "debug", False)
    mapper = _mapper(tmp_path)
    for i in range(5):
        _log(mapper, 10.0 + 0.2 * i)
    assert list(tmp_path.iterdir()) == []


def test_debug_on_logs_rows_under_the_header(tmp_path, debug_on):
    mapper = _mapper(tmp_path)
    _log(mapper, 10.0)
    _log(mapper, 10.2)
    mapper.close()
    rows = _rows(tmp_path / atp._DEBUG_LOG_NAME)
    assert rows[0] == HEADER
    assert len(rows) == 3


def test_open_reads_only_the_header_line(tmp_path, debug_on):
    """The old open read the whole file (382 MB) on the sending thread and tripped the watchdog."""
    path = tmp_path / atp._DEBUG_LOG_NAME
    body = ("1.000,x" + "," * (len(HEADER) - 2) + "\n") * 40_000
    # Bytes no UTF-8 decoder accepts, well past the first read buffer.
    original = HEADER_LINE.encode() + body.encode() + b"\xff\xfe\xfd\n"
    path.write_bytes(original)

    mapper = _mapper(tmp_path)
    _log(mapper, 10.0)
    mapper.close()

    after = path.read_bytes()
    assert after.startswith(original)           # appended, never rewritten
    assert after.count(b"\n") == original.count(b"\n") + 1
    assert [p.name for p in tmp_path.iterdir()] == [atp._DEBUG_LOG_NAME]


def test_oversized_log_is_rotated_aside_at_open(tmp_path, debug_on, monkeypatch):
    monkeypatch.setattr(atp, "_DEBUG_LOG_MAX_BYTES", 1000)
    path = tmp_path / atp._DEBUG_LOG_NAME
    old = HEADER_LINE + "1.000\n" * 400
    path.write_text(old, encoding="utf-8")

    mapper = _mapper(tmp_path)
    _log(mapper, 10.0)
    mapper.close()

    rotated = [p for p in tmp_path.iterdir() if p.name != atp._DEBUG_LOG_NAME]
    assert len(rotated) == 1 and rotated[0].read_text(encoding="utf-8") == old
    rows = _rows(path)
    assert rows[0] == HEADER and len(rows) == 2


def test_log_under_the_cap_keeps_growing_in_place(tmp_path, debug_on, monkeypatch):
    monkeypatch.setattr(atp, "_DEBUG_LOG_MAX_BYTES", 1_000_000)
    path = tmp_path / atp._DEBUG_LOG_NAME
    path.write_text(HEADER_LINE + "1.000\n" * 10, encoding="utf-8")

    mapper = _mapper(tmp_path)
    _log(mapper, 10.0)
    mapper.close()

    assert [p.name for p in tmp_path.iterdir()] == [atp._DEBUG_LOG_NAME]
    assert len(_rows(path)) == 12


def test_headerless_log_is_rotated_not_rewritten(tmp_path, debug_on):
    path = tmp_path / atp._DEBUG_LOG_NAME
    path.write_text("1.000,2.000\n3.000,4.000\n", encoding="utf-8")

    mapper = _mapper(tmp_path)
    _log(mapper, 10.0)
    mapper.close()

    rotated = [p for p in tmp_path.iterdir() if p.name != atp._DEBUG_LOG_NAME]
    assert len(rotated) == 1 and _rows(rotated[0]) == [["1.000", "2.000"], ["3.000", "4.000"]]
    assert _rows(path)[0] == HEADER


def test_turning_debug_off_closes_the_log(tmp_path, monkeypatch):
    inst = Settings.instance()
    monkeypatch.setattr(inst, "debug", True)
    mapper = _mapper(tmp_path)
    _log(mapper, 10.0)
    assert mapper._debug_log_file is not None
    monkeypatch.setattr(inst, "debug", False)
    _log(mapper, 10.2)
    assert mapper._debug_log_file is None
    assert len(_rows(tmp_path / atp._DEBUG_LOG_NAME)) == 2


def test_unwritable_root_never_raises(tmp_path, debug_on):
    mapper = _mapper(tmp_path / "missing" / "dir")
    _log(mapper, 10.0)
    _log(mapper, 10.2)
    assert mapper._debug_log_file is None
