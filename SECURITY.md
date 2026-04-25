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
