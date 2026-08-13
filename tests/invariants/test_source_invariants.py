"""Structural invariants from AGENTS.md, checked over the source tree.

These rules are about *where* code may appear, not what a function returns, so
they are checked with the AST rather than by running a controller. Each one
guards a regression that already shipped at least once.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _py_files(*roots: Path):
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    cur = node
    while cur in parents:
        cur = parents[cur]
        yield cur


def test_lv_accelerationx_never_leaves_telemetry_ingest():
    """`lv_accelerationX` is the truck-local LATERAL axis despite its name.

    Using it as accel/decel feedback causes phantom braking on right turns.
    Only the telemetry ingest may name it; every consumer must differentiate
    `speed` instead.
    """
    allowed = {"core/telemetry_thread/thread.py"}
    offenders = [
        f"{_rel(p)}:{i}"
        for p in _py_files(REPO / "core", REPO / "ui", REPO / "shared")
        if _rel(p) not in allowed
        for i, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1)
        if "lv_accelerationX" in line
    ]
    assert not offenders, (
        "lv_accelerationX is lateral, not longitudinal; use a tracking "
        f"differentiator on speed instead. Found at: {offenders}"
    )


def test_exactly_one_accel_to_pedals_instance():
    """Two mappers diverge in per-instance PID/EMA state and broke handover."""
    sites = []
    for path in _py_files(REPO / "core", REPO / "ui", REPO / "shared"):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "AccelToPedals":
                    sites.append(f"{_rel(path)}:{node.lineno}")
    assert len(sites) == 1, f"expected one AccelToPedals instantiation, found {sites}"
    assert sites[0].startswith("core/sending_thread/thread.py"), (
        f"the single mapper must live in SendingThread, found at {sites[0]}"
    )


def _base_thread_subclasses(tree: ast.Module):
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if name == "BaseThread":
                yield node


def _self_call_names(fn: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if isinstance(target, ast.Name) and target.id == "self":
                names.add(node.func.attr)
    return names


def _sleep_linenos(fn: ast.AST) -> list[int]:
    hits = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "sleep":
            hits.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id == "sleep":
            hits.append(node.lineno)
    return hits


def test_no_sleep_reachable_from_a_thread_loop():
    """BaseThread paces the loop; a sleep inside it desynchronises the watchdog.

    Scope: `loop` plus every same-class method it calls, transitively. `setup`
    and `teardown` are allowed to sleep and are not reachable from here.
    """
    offenders = []
    for path in _py_files(REPO / "core"):
        for cls in _base_thread_subclasses(_parse(path)):
            methods = {
                m.name: m for m in cls.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "loop" not in methods:
                continue
            reached, stack = set(), ["loop"]
            while stack:
                name = stack.pop()
                if name in reached or name not in methods:
                    continue
                reached.add(name)
                stack.extend(_self_call_names(methods[name]))
            for name in sorted(reached):
                for lineno in _sleep_linenos(methods[name]):
                    offenders.append(f"{_rel(path)}:{lineno} ({cls.name}.{name})")
    assert not offenders, (
        f"sleep reachable from a thread loop; use BaseThread pacing: {offenders}"
    )


def test_registry_reads_are_guarded():
    """A thread must never crash because a sibling is missing or down."""
    offenders = []
    for path in _py_files(REPO / "core", REPO / "ui"):
        tree = _parse(path)
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "get_thread":
                continue
            if not any(isinstance(a, ast.Try) for a in _enclosing(node, parents)):
                offenders.append(f"{_rel(path)}:{node.lineno}")
    assert not offenders, (
        f"registry.get_thread outside a try block (missing sibling crashes the thread): {offenders}"
    )


def test_user_gas_above_mapper_is_branch_independent():
    """The flag must also be set while the sole-bidder limiter cap binds.

    Nesting it under the CC/ACC max-merge branch round-trips the winner label
    limiter -> cc -> limiter on the enable tick, which lets one tick of uncapped
    user pedal reach the game and then surges the gas past the limit.
    """
    path = REPO / "core" / "sending_thread" / "thread.py"
    tree = _parse(path)
    parents = _parent_map(tree)

    computed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Attribute)]
        if not any(t.attr == "user_gas_above_mapper" for t in targets):
            continue
        # Resets to a literal False are fine; only the real computation matters.
        if isinstance(node.value, ast.Constant):
            continue
        computed.append(node)

    assert computed, "no computed assignment to user_gas_above_mapper found"

    for node in computed:
        for ancestor in _enclosing(node, parents):
            if not isinstance(ancestor, ast.If):
                continue
            names = {n.id for n in ast.walk(ancestor.test) if isinstance(n, ast.Name)}
            assert "cruise_active_controller" not in names, (
                f"{_rel(path)}:{node.lineno} user_gas_above_mapper is nested under a "
                "branch on cruise_active_controller; it must be branch-independent"
            )


_SLOPE_TOKENS = ("rotationY", "road_pitch", "grade_rad", "slope_accel")
_CRUISE_PID_ROOTS = (
    REPO / "core" / "longitudinal",
    REPO / "core" / "cruise_control_thread",
)


def test_cruise_pid_does_not_read_road_pitch():
    """Grade feed-forward belongs in AccelToPedals, not the speed PID."""
    offenders = [
        f"{_rel(path)}:{i}"
        for root in _CRUISE_PID_ROOTS
        for path in _py_files(root)
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if any(tok in line for tok in _SLOPE_TOKENS)
    ]
    assert not offenders, (
        "slope/pitch terms in the cruise PID; keep grade FF in AccelToPedals. "
        f"Found at: {offenders}"
    )
