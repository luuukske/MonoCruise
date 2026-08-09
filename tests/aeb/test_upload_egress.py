"""Only two modules in core/aeb may talk to the network, and this proves it.

Without this the single-choke-point design in the contribution plan is a
convention rather than a property, and a later change could add a second path
that skips the consent check without anything failing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "core" / "aeb"

# upload.py sends clips and gates every send on consent; intake_policy.py fetches
# the kill switch and refuses to run at all unless the user opted in.
_ALLOWED = {"upload.py", "intake_policy.py"}

_NETWORK_MODULES = {
    "requests", "urllib", "urllib3", "http", "socket", "httpx", "aiohttp", "ftplib", "smtplib",
}


def _imported_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize(
    "path", sorted(p for p in _PACKAGE.glob("*.py") if p.name not in _ALLOWED),
    ids=lambda p: p.name,
)
def test_no_other_aeb_module_reaches_the_network(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _imported_modules(tree) & _NETWORK_MODULES
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}; outbound requests belong in "
        f"one of {sorted(_ALLOWED)} so the consent check cannot be bypassed"
    )


def test_the_allowlist_still_describes_real_files():
    """A renamed module must not silently widen the rule to nothing."""
    for name in _ALLOWED:
        assert (_PACKAGE / name).is_file()


def test_every_send_goes_through_the_consent_check():
    source = (_PACKAGE / "upload.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    senders = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_handle"
    ]
    assert senders, "upload.py no longer has the _handle entry point"
    called = {
        node.func.id for node in ast.walk(senders[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "contribution_enabled" in called
    assert "upload_blocked_reason" in called
