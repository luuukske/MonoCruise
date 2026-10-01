"""Store discovery and the agent workspace directory. Dev only, never shipped."""

from __future__ import annotations

import os
from pathlib import Path

from core.aeb.clip_store import ClipStore, contributed_clip_root, default_clip_root

_REPO = Path(__file__).resolve().parents[2]
_WORKSPACE_ENV = "MONOCRUISE_AEB_AGENT_DIR"
# Default lives under the gitignored corpus-run folder so proposals, journals and
# the index cache never reach a commit.
_DEFAULT_WORKSPACE = "tools/aeb_corpus_run/agent"

STORE_NAMES = ("local", "remote")


def repo_root() -> Path:
    return _REPO


def workspace() -> Path:
    """Directory holding the index cache, proposals and the label journal."""
    override = os.environ.get(_WORKSPACE_ENV, "").strip()
    path = Path(override) if override else (_REPO / _DEFAULT_WORKSPACE)
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_roots() -> dict[str, Path]:
    """The two clip roots by short name: ``local`` captures, ``remote`` pulls."""
    return {"local": default_clip_root(), "remote": contributed_clip_root()}


def stores(which: str = "both") -> list[tuple[str, ClipStore]]:
    """(origin, store) pairs for ``local``, ``remote`` or ``both``."""
    roots = store_roots()
    if which in roots:
        return [(which, ClipStore(roots[which]))]
    if which not in ("both", "all"):
        raise ValueError(f"unknown store {which!r}; use local, remote or both")
    return [(name, ClipStore(root)) for name, root in roots.items() if root.is_dir()]


def rel_to_repo(path: Path) -> str:
    """Repo-relative POSIX path when the file is inside the tree, else its name."""
    try:
        return Path(path).resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return Path(path).name
