"""AGENTS.md rules that a machine can check, so they stop costing context budget.

Two shapes of check here. Absolute rules fail on the first violation. Ratchets
carry a recorded baseline: the tree is not clean today, and the point is that it
must not get worse. Lower a baseline whenever you clean something up; never
raise one to make a build pass.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("core", "ui", "shared", "updater", "checker", "tools", "tests")
LOOSE_FILES = ("monocruise.py", "conftest.py")

LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}

# Baselines. See the module docstring before touching one.
MAX_COMMENT_RUNS_OVER_BUDGET = 20
MAX_COMMENT_RUN_LENGTH = 4
MAX_POPUP_ON_FAILURE_LOGS = 7
MAX_FILES_OVER_800_LINES = 10
MAX_DUPLICATED_WINDOW_PCT = 3.0

# Files already past the ceiling when it was introduced. Do not add to this.
OVERSIZED_FILES = {
    "core/aeb/thread.py",
    "ui/cc_panel/main.py",
    "core/radar/traffic.py",
    "core/sending_thread/thread.py",
    "ui/main_window/settings_panel.py",
    "updater/updater.py",
}
FILE_LINE_CEILING = 1200


def _py_files():
    for name in SOURCE_ROOTS:
        root = REPO / name
        if root.is_dir():
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" not in path.parts:
                    yield path
    for name in LOOSE_FILES:
        path = REPO / name
        if path.is_file():
            yield path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _comments(path: Path):
    """(lineno, text) for every comment token in the file."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(_read(path)).readline)
        return [(t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def _docstrings(tree: ast.Module):
    nodes = [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in nodes:
        text = ast.get_docstring(node)
        if text:
            yield getattr(node, "lineno", 1), text


def _log_calls(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in LOG_LEVELS:
                yield node


def test_no_em_dash_in_comments_docstrings_or_log_messages():
    """House style. UI placeholder glyphs are exempt: this is about prose."""
    offenders = []
    for path in _py_files():
        for lineno, text in _comments(path):
            if "—" in text:
                offenders.append(f"{_rel(path)}:{lineno} (comment)")
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue
        for lineno, text in _docstrings(tree):
            if "—" in text:
                offenders.append(f"{_rel(path)}:{lineno} (docstring)")
        for call in _log_calls(tree):
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and "—" in arg.value:
                    offenders.append(f"{_rel(path)}:{call.lineno} (log message)")
    assert not offenders, f"em dash in code-adjacent text; use a comma or colon: {offenders}"


def test_no_absolute_or_user_specific_paths():
    """A committed home directory leaks a username and breaks on every other machine."""
    # Built from fragments so this checker does not match its own source.
    pattern = re.compile("|".join([
        "/" + "home/",
        "/" + "Users/",
        r"[A-Za-z]:\\{1,2}" + "Users",
        r"\\{2}" + "Users",
    ]))
    offenders = []
    for path in _py_files():
        for i, line in enumerate(_read(path).splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{_rel(path)}:{i}")
    assert not offenders, f"absolute or user-specific path in a committed file: {offenders}"


def test_popup_is_never_attached_to_a_trace_or_debug_log():
    """Exception traces and debug state are for the log file, never the screen."""
    offenders = []
    for path in _py_files():
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue
        for call in _log_calls(tree):
            if call.func.attr not in {"exception", "debug"}:
                continue
            if _has_popup_true(call):
                offenders.append(f"{_rel(path)}:{call.lineno} ({call.func.attr})")
    assert not offenders, f"popup=True on a trace/debug log: {offenders}"


def test_popup_on_failure_logs_does_not_grow():
    """error/critical popups are legitimate but rare: each one is a design decision.

    Current holders are the thread-management failure notices and the pedal
    reconnect failure. Adding one means arguing it belongs on screen.
    """
    sites = []
    for path in _py_files():
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue
        for call in _log_calls(tree):
            if call.func.attr in {"error", "critical"} and _has_popup_true(call):
                sites.append(f"{_rel(path)}:{call.lineno}")
    assert len(sites) <= MAX_POPUP_ON_FAILURE_LOGS, (
        f"popup=True on error/critical grew past {MAX_POPUP_ON_FAILURE_LOGS}: {sites}"
    )


def _has_popup_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg != "extra" or not isinstance(kw.value, ast.Dict):
            continue
        for key, value in zip(kw.value.keys, kw.value.values):
            if (isinstance(key, ast.Constant) and key.value == "popup"
                    and isinstance(value, ast.Constant) and value.value is True):
                return True
    return False


def _comment_runs():
    """(length, location) for every run of consecutive whole-line comments."""
    runs = []
    for path in _py_files():
        length = 0
        start = 0
        lines = _read(path).splitlines()
        for i, line in enumerate(lines + [""], 1):
            if line.strip().startswith("#"):
                if length == 0:
                    start = i
                length += 1
                continue
            if length > 2:
                runs.append((length, f"{_rel(path)}:{start}"))
            length = 0
    return runs


def test_comment_blocks_stay_within_the_budget():
    """The 2-line cap came out of a 5,899-line comment purge; this stops the regrowth."""
    runs = _comment_runs()
    assert len(runs) <= MAX_COMMENT_RUNS_OVER_BUDGET, (
        f"comment runs over 2 lines grew past {MAX_COMMENT_RUNS_OVER_BUDGET}: "
        f"{sorted(runs, reverse=True)}"
    )
    worst = max((n for n, _ in runs), default=0)
    assert worst <= MAX_COMMENT_RUN_LENGTH, (
        f"a comment block grew to {worst} lines; put durable explanation in a README: "
        f"{[loc for n, loc in runs if n > MAX_COMMENT_RUN_LENGTH]}"
    )


def test_no_new_oversized_file():
    """Large files are where agents lose context and re-derive instead of reusing."""
    over = {
        _rel(p): n
        for p in _py_files()
        if (n := len(_read(p).splitlines())) > FILE_LINE_CEILING
    }
    new = set(over) - OVERSIZED_FILES
    assert not new, f"new file over {FILE_LINE_CEILING} lines: {[f'{k}:{over[k]}' for k in new]}"

    shrunk = OVERSIZED_FILES - set(over)
    assert not shrunk, (
        f"these dropped under the ceiling: remove them from OVERSIZED_FILES: {sorted(shrunk)}"
    )


def test_large_file_count_does_not_grow():
    big = [_rel(p) for p in _py_files() if len(_read(p).splitlines()) > 800]
    assert len(big) <= MAX_FILES_OVER_800_LINES, (
        f"files over 800 lines grew past {MAX_FILES_OVER_800_LINES}: {sorted(big)}"
    )


def test_agent_docs_only_point_at_files_that_exist():
    """AGENTS.md referenced `main.py` for months; the entry point is monocruise.py.

    A pointer to a file that is not there is worse than no pointer: an agent
    burns a search on it and then guesses.
    """
    docs = [REPO / "AGENTS.md"] + sorted(
        p for p in REPO.rglob("README.md")
        if "__pycache__" not in p.parts
        and "build" not in p.parts
        and "dist" not in p.parts
        and "archive" not in p.parts
        and ".venv" not in p.parts
    )
    # Backticked paths only, so prose like "the loop method" is never mistaken for one.
    ref = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|json|toml|yml|yaml|iss|spec))`")
    offenders = []
    for doc in docs:
        for lineno, line in enumerate(_read(doc).splitlines(), 1):
            for match in ref.findall(line):
                if "/" not in match and match not in {"AGENTS.md", "README.md"}:
                    continue  # bare filename: could be any of several, skip
                candidates = [REPO / match, doc.parent / match]
                if not any(c.exists() for c in candidates):
                    offenders.append(f"{_rel(doc)}:{lineno} -> {match}")
    assert not offenders, f"agent-facing doc points at a file that does not exist: {offenders}"


def test_duplication_stays_under_budget():
    """Clones are where a fix lands in one copy and not the other."""
    windows: Counter[str] = Counter()
    for path in _py_files():
        if "tests" in path.parts:
            continue
        lines = [ln.strip() for ln in _read(path).splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        for i in range(len(lines) - 5):
            windows["\n".join(lines[i:i + 6])] += 1

    total = sum(windows.values())
    duplicated = sum(count for count in windows.values() if count > 1)
    pct = 100.0 * duplicated / max(total, 1)
    assert pct <= MAX_DUPLICATED_WINDOW_PCT, (
        f"duplicated 6-line windows at {pct:.2f}%, budget {MAX_DUPLICATED_WINDOW_PCT}%. "
        "Extract the shared logic instead of copying it."
    )
