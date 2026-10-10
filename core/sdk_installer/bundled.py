"""Upstream plugin files shipped with the build: the fallback when GitHub cannot be asked.

Filled at build time by tools/fetch_sdk_bundle.py. See the "Bundled fallback" section of
README.md in this package.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .remote import RemoteFile, git_blob_sha_of

log = logging.getLogger("sdk")

BUNDLE_DIR = Path(__file__).resolve().parent / "bundled"
MANIFEST_FILE = "manifest.json"


def read_manifest(version_dir: Path) -> dict[str, str]:
    """name -> git-blob SHA the build verified, or {} when absent or unreadable."""
    try:
        data = json.loads((version_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        return {}
    return {k: v for k, v in files.items() if isinstance(k, str) and isinstance(v, str)}


def bundled_listing(version: str, bundle_dir: Path | None = None) -> dict[str, RemoteFile]:
    """Shipped files for one game version that still hash to the manifest."""
    version_dir = (bundle_dir or BUNDLE_DIR) / version
    listing: dict[str, RemoteFile] = {}
    for name, sha in read_manifest(version_dir).items():
        path = version_dir / name
        if git_blob_sha_of(path) != sha:
            log.warning("the bundled %s for game version %s is missing or damaged", name, version)
            continue
        listing[name] = RemoteFile(name, sha, path.stat().st_size, "", local_path=path)
    return listing
