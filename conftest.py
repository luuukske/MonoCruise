"""Root conftest: sys.path, config sandbox, and skip discipline for the whole suite."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import core.settings as _settings  # noqa: E402  (needs the sys.path insert above)

# Redirect every settings read/write into a throwaway directory before any test
# can call Settings.load/save. Tests drive real settings-writing code paths, and
# without this they overwrite the developer's live config.json with defaults.
_CONFIG_SANDBOX = Path(tempfile.mkdtemp(prefix="monocruise-tests-"))
_REAL_CONFIG = _settings.CONFIG_PATH
_settings.CONFIG_PATH = _CONFIG_SANDBOX / "config.json"
_settings.BACKUP_PATH = _settings.CONFIG_PATH.with_suffix(
    _settings.CONFIG_PATH.suffix + ".bak"
)
if _REAL_CONFIG.is_file():
    # Seed from the real file so tests see the same values they always did.
    shutil.copy2(str(_REAL_CONFIG), str(_settings.CONFIG_PATH))

# A skip carrying one of these markers is a known, accepted gap. Anything else
# that skips is a test silently asserting nothing, which is what the gate below
# is for.
_ALLOWED_SKIP_MARKERS = ("needs_clips",)
_STRICT_SKIPS = os.environ.get("MONOCRUISE_STRICT_SKIPS", "") not in ("", "0")
_unexpected_skips: dict[str, str] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_clips: needs an AEB clip fixture that lives outside the repo",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped:
        return
    if any(item.get_closest_marker(m) for m in _ALLOWED_SKIP_MARKERS):
        return
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    _unexpected_skips.setdefault(report.nodeid, reason)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _unexpected_skips:
        return
    label = "ERROR" if _STRICT_SKIPS else "WARNING"
    terminalreporter.write_sep("=", f"{label}: unmarked skips", red=_STRICT_SKIPS)
    for nodeid, reason in _unexpected_skips.items():
        terminalreporter.write_line(f"  {nodeid}: {reason}")
    terminalreporter.write_line(
        "These tests reported success while asserting nothing. Install the missing "
        "dependency, or mark the test with an allowed skip marker."
    )


def pytest_sessionfinish(session, exitstatus):
    if _STRICT_SKIPS and _unexpected_skips and session.exitstatus == 0:
        session.exitstatus = 1
    shutil.rmtree(str(_CONFIG_SANDBOX), ignore_errors=True)
