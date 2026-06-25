"""Release helper for MonoCruise.

Two subcommands:

    python tools/release.py bump {major|minor|patch}
        Bumps core/version.py, rewrites the [Unreleased] heading in CHANGELOG.md
        to "[X.Y.Z] - YYYY-MM-DD", inserts a fresh empty [Unreleased] block above
        it, commits both files, tags vX.Y.Z, and pushes commit + tag.

    python tools/release.py bump --set X.Y.Z
        Same as above but with an explicit version.

    python tools/release.py notes X.Y.Z
        Prints the CHANGELOG section for X.Y.Z to stdout (used by CI to build
        the GitHub release body).

The bump command refuses to run if the [Unreleased] block is empty: i.e. you
must have at least one line of changelog entries below the [Unreleased] heading
before publishing a release.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "core" / "version.py"
CHANGELOG = ROOT / "CHANGELOG.md"

_VERSION_RE = re.compile(r'^__version__\s*=\s*"(?P<v>\d+\.\d+\.\d+)"\s*$', re.M)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _read_version() -> str:
    m = _VERSION_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"could not find __version__ in {VERSION_FILE}")
    return m.group("v")


def _write_version(new_version: str) -> None:
    text = VERSION_FILE.read_text(encoding="utf-8")
    new_text = _VERSION_RE.sub(f'__version__ = "{new_version}"', text)
    VERSION_FILE.write_text(new_text, encoding="utf-8")


def _bump(current: str, level: str) -> str:
    major, minor, patch = (int(p) for p in current.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"unknown bump level: {level}")


def _split_changelog(text: str) -> list[tuple[str, str]]:
    """Return list of (heading_line, body) for each ## section, in order."""
    parts: list[tuple[str, str]] = []
    current_head: str | None = None
    current_body: list[str] = []
    head_pos: int | None = None
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            if current_head is not None:
                parts.append((current_head, "".join(current_body)))
            current_head = line
            current_body = []
            head_pos = len(parts)
        else:
            if current_head is None:
                # preamble: hold onto it as a sentinel entry
                parts.append(("", line)) if head_pos is None else None
            else:
                current_body.append(line)
    if current_head is not None:
        parts.append((current_head, "".join(current_body)))
    return parts


def _extract_section(version: str) -> str | None:
    """Return the body of the `## [version]` section, or None if missing."""
    text = CHANGELOG.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^##\s+\[" + re.escape(version) + r"\][^\n]*\n(?P<body>.*?)(?=^##\s+\[|\Z)",
        re.M | re.S,
    )
    m = pattern.search(text)
    if not m:
        return None
    return m.group("body").strip("\n")


def _unreleased_has_content() -> bool:
    body = _extract_section("Unreleased") or ""
    return bool(body.strip())


def _promote_unreleased(new_version: str) -> None:
    today = _dt.date.today().isoformat()
    text = CHANGELOG.read_text(encoding="utf-8")
    if not re.search(r"^##\s+\[Unreleased\]\s*$", text, re.M):
        raise SystemExit("CHANGELOG.md has no [Unreleased] heading")
    new_text = re.sub(
        r"^##\s+\[Unreleased\]\s*$",
        f"## [Unreleased]\n\n## [{new_version}] - {today}",
        text,
        count=1,
        flags=re.M,
    )
    CHANGELOG.write_text(new_text, encoding="utf-8")


def _run(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def cmd_bump(args: argparse.Namespace) -> None:
    current = _read_version()
    if args.set:
        if not _SEMVER_RE.match(args.set):
            raise SystemExit(f"--set must be MAJOR.MINOR.PATCH, got {args.set!r}")
        new_version = args.set
    else:
        new_version = _bump(current, args.level)
    if new_version == current:
        raise SystemExit(f"new version equals current ({current}): nothing to bump")

    if not _unreleased_has_content():
        raise SystemExit(
            "[Unreleased] section in CHANGELOG.md is empty. "
            "Add changelog entries before bumping."
        )

    print(f"bumping {current} -> {new_version}")
    _write_version(new_version)
    _promote_unreleased(new_version)

    if args.dry_run:
        print("dry-run: skipping git commit/tag/push")
        return

    tag = f"v{new_version}"
    _run("git", "add", str(VERSION_FILE.relative_to(ROOT)), str(CHANGELOG.relative_to(ROOT)))
    _run("git", "commit", "-m", f"Release {tag}")
    _run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    _run("git", "push")
    _run("git", "push", "origin", tag)
    print(f"\nPushed {tag}. CI will build and publish the release.")


def cmd_notes(args: argparse.Namespace) -> None:
    body = _extract_section(args.version)
    if body is None:
        raise SystemExit(f"no section for [{args.version}] in CHANGELOG.md")
    sys.stdout.write(body.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MonoCruise release helper")
    sub = parser.add_subparsers(dest="command", required=True)

    bump = sub.add_parser("bump", help="bump version, commit, tag, push")
    bump.add_argument("level", nargs="?", choices=["major", "minor", "patch"])
    bump.add_argument("--set", dest="set", help="explicit MAJOR.MINOR.PATCH")
    bump.add_argument("--dry-run", action="store_true", help="skip git operations")
    bump.set_defaults(func=cmd_bump)

    notes = sub.add_parser("notes", help="print CHANGELOG section for a version")
    notes.add_argument("version", help="MAJOR.MINOR.PATCH (no leading v)")
    notes.set_defaults(func=cmd_notes)

    args = parser.parse_args()
    if args.command == "bump" and not args.set and not args.level:
        parser.error("bump requires either a level (major|minor|patch) or --set")
    args.func(args)


if __name__ == "__main__":
    main()

