"""MonoCruise release helper (bump | notes). Needs gh CLI and nonempty [Unreleased] changelog."""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "core" / "version.py"
CHANGELOG = ROOT / "CHANGELOG.md"

# Semver MAJOR.MINOR.PATCH plus optional -LABEL.N pre-release (e.g. 1.1.0-preview.1).
_VERSION_RE = re.compile(
    r'^__version__\s*=\s*"(?P<v>\d+\.\d+\.\d+(?:-[A-Za-z]+\.\d+)?)"\s*$', re.M
)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[A-Za-z]+\.\d+)?$")
_PRE_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)(?:-(?P<label>[A-Za-z]+)\.(?P<num>\d+))?$"
)

# gh install paths: detect PATH stale after install (never invoke gh by these paths).
_GH_INSTALL_PATHS = (
    r"C:\Program Files\GitHub CLI\gh.exe",
    r"C:\Program Files (x86)\GitHub CLI\gh.exe",
    "/usr/bin/gh",
    "/usr/local/bin/gh",
    "/opt/homebrew/bin/gh",
)

_EPILOG = """\
examples:
  python tools/release.py bump patch                  # 1.1.0 -> 1.1.1
  python tools/release.py bump minor --pre preview    # 1.1.0 -> 1.2.0-preview.1
  python tools/release.py bump --pre preview          # 1.1.0-preview.1 -> 1.1.0-preview.2
  python tools/release.py bump patch                  # 1.1.0-preview.3 -> 1.1.0  (finalize)
  python tools/release.py bump --set 1.2.0-rc.1
  python tools/release.py bump minor --dry-run        # print plan, change nothing
  python tools/release.py bump patch --yes            # skip confirmation prompt
  python tools/release.py notes 1.1.0
"""


def _read_version() -> str:
    m = _VERSION_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"could not find __version__ in {VERSION_FILE}")
    return m.group("v")


def _write_version(new_version: str) -> None:
    text = VERSION_FILE.read_text(encoding="utf-8")
    new_text = _VERSION_RE.sub(f'__version__ = "{new_version}"', text)
    VERSION_FILE.write_text(new_text, encoding="utf-8")


def _parse_version(v: str) -> tuple[str, str | None, int | None]:
    """Split a version into (base, pre_label, pre_num). "1.1.0" -> ("1.1.0", None, None)
    "1.1.0-preview.2" -> ("1.1.0", "preview", 2)"""
    m = _PRE_RE.match(v)
    if not m:
        raise SystemExit(f"unparseable version: {v!r}")
    num = m.group("num")
    return m.group("base"), m.group("label"), (int(num) if num is not None else None)


def _bump_base(base: str, level: str) -> str:
    major, minor, patch = (int(p) for p in base.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"unknown bump level: {level}")


def _bump(current: str, level: str | None = None, pre: str | None = None) -> str:
    """Next semver from current version, optional bump level, and --pre label. See bump --help."""
    base, label, num = _parse_version(current)

    if pre is not None:
        if level is not None:
            base = _bump_base(base, level)
            return f"{base}-{pre}.1"
        if label == pre:
            return f"{base}-{pre}.{(num or 0) + 1}"
        return f"{base}-{pre}.1"

    if label is not None:
        # Finalizing a pre-release: the suffix drops, the base stays.
        return base

    if level is None:
        raise SystemExit("nothing to bump: pass a level (major|minor|patch) or --pre")
    return _bump_base(base, level)


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


def _check_gh() -> None:
    """Fail before anything is written or pushed if gh is unusable. Ordering matters: once the tag
    is pushed, CI starts a build that expects a draft to exist. Discovering a missing gh at that
    point would leave a tag with no release behind it, so the check runs first."""
    if shutil.which("gh") is None:
        hint = (
            "Install from https://cli.github.com, then run: gh auth login"
        )
        # gh on disk but not PATH: shell started before install updated PATH.
        if any(Path(p).exists() for p in _GH_INSTALL_PATHS):
            hint = (
                "It is installed, but not on this shell's PATH. The installer "
                "updates PATH for new processes only, so restart your terminal "
                "(or IDE) and run this again."
            )
        raise SystemExit(
            "gh (GitHub CLI) not found, but bump needs it to create the "
            f"release draft under your own account.\n{hint}"
        )
    result = subprocess.run(
        ["gh", "auth", "status"], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit("gh is installed but not logged in. Run: gh auth login")


def _create_draft(tag: str, version: str) -> None:
    """Draft GitHub release under caller gh account (CI attaches artifacts and publishes)."""
    notes = _extract_section(version) or ""
    notes_file = ROOT / ".release-notes.md"
    notes_file.write_text(notes.rstrip() + "\n", encoding="utf-8")
    cmd = [
        "gh", "release", "create", tag,
        "--draft",
        "--verify-tag",
        "--title", f"MonoCruise {tag}",
        "--notes-file", str(notes_file),
    ]
    # Same rule CI used to apply: any version carrying a -LABEL.N suffix is a
    # prerelease, which is what puts it on the preview channel.
    if "-" in version:
        cmd.append("--prerelease")
    try:
        _run(*cmd)
    finally:
        notes_file.unlink(missing_ok=True)


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _print_plan(
    *,
    current: str,
    new_version: str,
    dry_run: bool,
) -> None:
    today = _dt.date.today().isoformat()
    tag = f"v{new_version}"
    is_prerelease = "-" in new_version
    notes = (_extract_section("Unreleased") or "").strip()
    version_rel = VERSION_FILE.relative_to(ROOT).as_posix()
    changelog_rel = CHANGELOG.relative_to(ROOT).as_posix()

    print()
    print("Release plan")
    print("============")
    print(f"  Version       : {current}  ->  {new_version}")
    print(f"  Tag           : {tag}")
    print(f"  Channel       : {'prerelease (preview)' if is_prerelease else 'stable'}")
    print(f"  Date          : {today}")
    print()
    print("Files that will be updated:")
    print(f"  - {version_rel}   (__version__ = \"{new_version}\")")
    print(f"  - {changelog_rel}  ([Unreleased] -> [{new_version}] - {today})")
    print()
    print("Changelog notes (become the GitHub release body):")
    if notes:
        for line in notes.splitlines():
            print(f"  {line}")
    else:
        print("  (empty)")
    print()
    if dry_run:
        print("Git / GitHub (skipped in --dry-run):")
    else:
        print("Git / GitHub actions:")
    print(f"  - git add {version_rel} {changelog_rel}")
    print(f"  - git commit -m \"Release {tag}\"")
    print(f"  - git tag -a {tag} -m \"Release {tag}\"")
    print("  - git push")
    print(f"  - git push origin {tag}")
    prerelease_flag = " --prerelease" if is_prerelease else ""
    print(
        f"  - gh release create {tag} --draft --verify-tag"
        f" --title \"MonoCruise {tag}\"{prerelease_flag}"
    )
    print()


def cmd_bump(args: argparse.Namespace) -> None:
    current = _read_version()
    if args.set:
        if not _SEMVER_RE.match(args.set):
            raise SystemExit(
                f"--set must be MAJOR.MINOR.PATCH[-LABEL.N], got {args.set!r}"
            )
        new_version = args.set
    else:
        new_version = _bump(current, args.level, args.pre)
    if new_version == current:
        raise SystemExit(f"new version equals current ({current}): nothing to bump")

    if not _unreleased_has_content():
        raise SystemExit(
            "[Unreleased] section in CHANGELOG.md is empty. "
            "Add changelog entries before bumping."
        )

    if not args.dry_run:
        _check_gh()

    _print_plan(current=current, new_version=new_version, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry-run only: nothing was changed or created.")
        return

    if not args.yes and not _confirm("Proceed with this release?"):
        raise SystemExit("Aborted. Nothing was changed or created.")

    print()
    print(f"Bumping {current} -> {new_version}")
    _write_version(new_version)
    _promote_unreleased(new_version)

    tag = f"v{new_version}"
    _run(
        "git", "add",
        str(VERSION_FILE.relative_to(ROOT)),
        str(CHANGELOG.relative_to(ROOT)),
    )
    _run("git", "commit", "-m", f"Release {tag}")
    _run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    _run("git", "push")
    _run("git", "push", "origin", tag)
    _create_draft(tag, new_version)
    print(f"\nPushed {tag} and drafted its release.")
    print("CI will attach the build artifacts and publish it under your name.")


def cmd_notes(args: argparse.Namespace) -> None:
    body = _extract_section(args.version)
    if body is None:
        raise SystemExit(f"no section for [{args.version}] in CHANGELOG.md")
    sys.stdout.write(body.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MonoCruise release helper: bump, tag, push, and draft a GitHub release.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bump = sub.add_parser(
        "bump",
        help="bump version, update changelog, commit, tag, push, draft release",
        description=(
            "Compute the next version, show a full plan, then (after confirmation) "
            "update version + changelog, commit, tag, push, and create a draft "
            "GitHub release. Nothing is written until you confirm."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bump.add_argument(
        "level",
        nargs="?",
        choices=["major", "minor", "patch"],
        help="semver level to bump (omit with --pre to advance a pre-release, "
             "or with --set for an explicit version)",
    )
    bump.add_argument(
        "--set",
        dest="set",
        metavar="VERSION",
        help="set an explicit version MAJOR.MINOR.PATCH[-LABEL.N]",
    )
    bump.add_argument(
        "--pre",
        dest="pre",
        metavar="LABEL",
        help="pre-release label, e.g. 'preview' or 'rc' (produces -LABEL.N)",
    )
    bump.add_argument(
        "--dry-run",
        action="store_true",
        help="print the release plan and exit without changing or creating anything",
    )
    bump.add_argument(
        "-y", "--yes",
        action="store_true",
        help="skip the confirmation prompt (still refuses an empty [Unreleased])",
    )
    bump.set_defaults(func=cmd_bump)

    notes = sub.add_parser(
        "notes",
        help="print the CHANGELOG section for a version",
        description="Print the CHANGELOG.md body for a given version to stdout.",
    )
    notes.add_argument(
        "version",
        help="version to look up, MAJOR.MINOR.PATCH[-LABEL.N] (no leading v)",
    )
    notes.set_defaults(func=cmd_notes)

    args = parser.parse_args()
    if args.command == "bump" and not args.set and not args.level and not args.pre:
        bump.error("bump requires a level (major|minor|patch), --pre, or --set")
    args.func(args)


if __name__ == "__main__":
    main()
