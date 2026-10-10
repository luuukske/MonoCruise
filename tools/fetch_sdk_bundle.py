"""Fill core/sdk_installer/bundled/ with the live upstream game plugin set.

Run by .github/workflows/release.yml before PyInstaller, so every build ships the plugins
that were current when it was made. See "Bundled fallback" in core/sdk_installer/README.md.
GITHUB_TOKEN, when set, goes to api.github.com only and is never written anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.sdk_installer.bundled import BUNDLE_DIR, MANIFEST_FILE  # noqa: E402
from core.sdk_installer.manager import COURTESY_FILES, SdkManager  # noqa: E402
from core.sdk_installer.remote import (  # noqa: E402
    SdkSource,
    SdkSourceError,
    SdkVersionUnsupported,
)


def fetch(out_dir: Path, token: str | None) -> dict[str, list[str]]:
    """Download and SHA-check every published game version's set into ``out_dir``."""
    versions = SdkSource("", token=token).list_versions()
    if not versions:
        raise SdkSourceError("could not list the published game versions")
    tracked_for = SdkManager(data_dir=out_dir).tracked_files

    # Only version folders are generated; LICENSES.txt next to them is tracked in git.
    for stale in (p for p in out_dir.iterdir() if p.is_dir()) if out_dir.is_dir() else ():
        shutil.rmtree(stale)

    fetched: dict[str, list[str]] = {}
    for version in versions:
        source = SdkSource(version, token=token)
        try:
            listing = source.list_files()
        except SdkVersionUnsupported:
            continue  # a version folder without a Windows build
        missing = [n for n in tracked_for(version) if n not in listing]
        if missing:
            raise SdkSourceError(f"game version {version} upstream lacks {', '.join(missing)}")
        names = [*tracked_for(version), *(n for n in COURTESY_FILES if n in listing)]
        version_dir = out_dir / version
        for name in names:
            source.download(listing[name], version_dir / name)
        # Manifest last: a set without one is never offered (bundled.read_manifest).
        manifest = {"files": {name: listing[name].sha for name in names}}
        (version_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        fetched[version] = names
    if not fetched:
        raise SdkSourceError("no published game version has a Windows plugin set")
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the bundled game plugin fallback.")
    parser.add_argument("--out", type=Path, default=BUNDLE_DIR, help="bundle directory")
    args = parser.parse_args(argv)
    try:
        fetched = fetch(args.out, os.environ.get("GITHUB_TOKEN") or None)
    except SdkSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for version, names in fetched.items():
        print(f"{version}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
