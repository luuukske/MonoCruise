# Security Advisories

## Vulnerability Scan Log

---

### Scan: 2026-04-25 — Direct dependencies, PyPI/OSV database

| Package | Version | Severity | Advisory | Summary | Fix Available |
|---------|---------|----------|----------|---------|---------------|
| pillow | ~~11.3.0~~ → **12.2.0** | **HIGH** | [GHSA-cfh3-3jmp-rvhc](https://github.com/advisories/GHSA-cfh3-3jmp-rvhc) | Out-of-bounds write when loading crafted PSD images (RCE risk) | ✅ Fixed 2026-04-25 |
| pillow | ~~11.3.0~~ → **12.2.0** | **HIGH** | [GHSA-whj4-6x5x-4v2j](https://github.com/advisories/GHSA-whj4-6x5x-4v2j) | FITS GZIP decompression bomb — unbounded memory use | ✅ Fixed 2026-04-25 |
| mistune | ~~2.0.0rc1~~ → **3.2.0** | **HIGH** | [GHSA-fw3v-x4f2-v673](https://github.com/advisories/GHSA-fw3v-x4f2-v673) | ReDoS via catastrophic regex backtracking in inline markup | ✅ Fixed 2026-04-25 |
| requests | ~~2.32.5~~ → **2.33.1** | MODERATE | [GHSA-gc5v-m9x4-r6x2](https://github.com/advisories/GHSA-gc5v-m9x4-r6x2) | Predictable temp filename in `extract_zipped_paths()` — symlink attack | ✅ Fixed 2026-04-25 |

**Scope:** direct .venv packages (no pyproject.toml/requirements.txt found — scanned installed packages)
**Clean packages checked:** flask, werkzeug, jinja2, urllib3, numpy, pandas, markupsafe, bottle, gevent, pyinstaller, beautifulsoup4, moviepy, pynput

#### Notes
- `mistune==2.0.0rc1` is a **release candidate** — upgrading to the stable `2.0.3+` fixes the ReDoS and moves off a pre-release.
- Both `pillow` HIGH issues require input from untrusted image files to trigger. Risk is low if images are only loaded from the local ETS2 telemetry pipeline, but upgrade is still recommended.
- `requests` MODERATE requires local attacker access to exploit; low real-world risk for a desktop app.

---

### Scan: 2026-04-25 (run 2) — .venv installed packages, OSV database

| Package | Version | Severity | Advisory | Summary | Fix Available |
|---------|---------|----------|----------|---------|---------------|
| python-dotenv | 1.2.1 | MODERATE | [GHSA-mf9w-mj56-hr94](https://github.com/advisories/GHSA-mf9w-mj56-hr94) / CVE-2026-28684 | `set_key()`/`unset_key()` follow symlinks via cross-device rename fallback — arbitrary file overwrite | ✅ Fix: upgrade to 1.2.2 |

**Scope:** .venv installed packages, OSV batch query
**Clean packages checked:** flask 3.1.3, werkzeug 3.1.6, jinja2 3.1.6, requests 2.33.1, pillow 12.2.0, numpy 2.4.2, urllib3 2.6.3, setuptools 80.10.2, gevent 25.9.1, bottle 0.13.4, pyside6 6.10.2, pynput 1.8.1, pyinstaller 6.18.0, moviepy 2.2.1, beautifulsoup4 4.14.3, psutil 7.2.2, pandas 3.0.2, matplotlib 3.10.8, cffi 2.0.0

#### Notes
- `python-dotenv` MODERATE: project code does **not** call `set_key()` or `unset_key()` — confirmed by grep. Vulnerability is unexploitable in current usage. Upgrade to 1.2.2 when convenient.
- All previously flagged HIGH issues (pillow, mistune) confirmed fixed at current installed versions.
