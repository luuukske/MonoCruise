"""The Linux per-thread CPU read parses procfs and never asks pthread_getcpuclockid about a thread."""
from __future__ import annotations

from pathlib import Path

import pytest

import core.thread_management.gui_stall_probe as gsp

# Fields after the closing bracket: state, ppid, pgrp, session, tty, tpgid, flags,
# minflt, cminflt, majflt, cmajflt, utime, stime.
_PLAIN = b"1234 (python) S 1 1234 1234 0 -1 4194560 100 0 0 0 150 50 0 0 20 0 1 0 5"
_ODD_NAME = b"7 (a b) c) S 1 7 7 0 -1 0 0 0 0 0 30 10 0 0"


def test_user_and_system_ticks_add_up_to_seconds():
    assert gsp.task_stat_cpu_s(_PLAIN, 100) == pytest.approx(2.0)


def test_a_name_with_spaces_and_brackets_does_not_shift_the_fields():
    assert gsp.task_stat_cpu_s(_ODD_NAME, 100) == pytest.approx(0.4)


@pytest.mark.parametrize("bad", [b"", b"garbage", b"1 (x) S 1", b"1 (x) S 1 1 1 0 -1 0 0 0 0 0 a b"])
def test_a_malformed_line_reads_as_unknown(bad):
    assert gsp.task_stat_cpu_s(bad, 100) is None


def test_no_ticks_per_second_reads_as_unknown():
    assert gsp.task_stat_cpu_s(_PLAIN, 0) is None


def test_the_probe_never_calls_pthread_getcpuclockid():
    # It segfaulted CI on a thread that had already exited (2026-11-26).
    source = Path(gsp.__file__).read_text(encoding="utf-8")
    assert "pthread_getcpuclockid(" not in source
