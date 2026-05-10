# Security Advisories

## Vulnerability Scan Log

---

### Scan: 2026-04-24 — Direct dependencies, PyPI/OSV database

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

### Scan: 2026-04-25 — .venv installed packages, OSV database

| Package | Version | Severity | Advisory | Summary | Fix Available |
|---------|---------|----------|----------|---------|---------------|
| python-dotenv | 1.2.1 | MODERATE | [GHSA-mf9w-mj56-hr94](https://github.com/advisories/GHSA-mf9w-mj56-hr94) / CVE-2026-28684 | `set_key()`/`unset_key()` follow symlinks via cross-device rename fallback — arbitrary file overwrite | ✅ Fix: upgrade to 1.2.2 |

**Scope:** .venv installed packages, OSV batch query
**Clean packages checked:** flask 3.1.3, werkzeug 3.1.6, jinja2 3.1.6, requests 2.33.1, pillow 12.2.0, numpy 2.4.2, urllib3 2.6.3, setuptools 80.10.2, gevent 25.9.1, bottle 0.13.4, pyside6 6.10.2, pynput 1.8.1, pyinstaller 6.18.0, moviepy 2.2.1, beautifulsoup4 4.14.3, psutil 7.2.2, pandas 3.0.2, matplotlib 3.10.8, cffi 2.0.0

#### Notes
- `python-dotenv` MODERATE: project code does **not** call `set_key()` or `unset_key()` — confirmed by grep. Vulnerability is unexploitable in current usage. Upgrade to 1.2.2 when convenient.
- All previously flagged HIGH issues (pillow, mistune) confirmed fixed at current installed versions.

---

### Scan: 2026-04-26 — .venv installed packages, safety-cli 3.7.0

| Package | Version | Severity | Advisory | Summary | Fix Available |
|---------|---------|----------|----------|---------|---------------|
| pygments | 2.19.2 | MEDIUM | CVE-2026-4539 / SFTY-20260322-35073 | Security flaw in `AdlLexer` — details not fully public yet | ✅ Upgrade to 2.20.0 (available on PyPI) |

**Scope:** 122 .venv installed packages scanned (direct dependencies only noted)
**Clean direct deps:** dash 4.1.0, flask 3.1.3, flask-socketio 5.6.1, matplotlib 3.10.8, numpy 2.4.2, pandas 3.0.2, plotly 6.7.0, psutil 7.2.2, pygame 2.6.1, pyside6 6.10.2, requests 2.33.1

#### Notes
- `pygments` MEDIUM (CVE-2026-4539): affects the `AdlLexer` code path. Pygments is a transitive/dev dependency (syntax highlighting), not directly imported by project code. Low exploitability in a desktop app context. Upgrade to 2.20.0 when convenient.
- All previously flagged vulnerabilities (pillow, mistune, requests, python-dotenv) remain at fixed versions.

---

### Scan: 2026-04-27 — .venv installed packages, safety-cli 3.7.0

Vulnerability check 2026-04-27 — 122 packages scanned. No new vulnerabilities. 2 previously documented issues remain unpatched (pygments MEDIUM, python-dotenv MODERATE) — no fix applied since last scan.

---

### Scan: 2026-04-29 — direct project imports, OSV database

Vulnerability check 2026-04-29 — All direct dependencies clean. 1 previously documented issue persists (pygments LOW, see 2026-04-26 entry). No new vulnerabilities found.

**Scope:** direct imports from project source (PySide6 6.10.1, Flask 3.1.3, flask-cors 6.0.2, pygame 2.6.1, psutil 7.1.3, numpy 2.2.6, pandas 3.0.1, plotly 6.6.0, matplotlib 3.10.8, websockets 13.1) — OSV batch query

#### Notes
- `python-dotenv` no longer installed in current .venv — previously flagged MODERATE advisory no longer applicable.
- `pygments` 2.19.2: GHSA-5239-wwwm-4pmq (ReDoS via GUID regex, LOW) — fix available at 2.20.0, unchanged since 2026-04-26 scan.
- Environment regression detected: current .venv has older versions than 2026-04-27 scan (numpy 2.2.6 vs 2.4.2, psutil 7.1.3 vs 7.2.2, PySide6 6.10.1 vs 6.10.2, pandas 3.0.1 vs 3.0.2, plotly 6.6.0 vs 6.7.0). No security impact — no vulnerabilities in these older versions per OSV.

---

### Scan: 2026-05-03 — .venv installed packages, safety-cli 3.7.0

Vulnerability check 2026-05-03 — 125 packages scanned. No new vulnerabilities. 1 previously documented issue persists (pygments 2.19.2 MEDIUM, CVE-2026-4539 — see 2026-04-26 entry).

**Scope:** 125 .venv installed packages (safety-cli open-source database)

#### Notes
- `python-dotenv 1.2.1` has been reinstalled since the 2026-04-29 scan (which noted it was absent). Previously flagged MODERATE advisory GHSA-mf9w-mj56-hr94 / CVE-2026-28684 applies again (symlink attack via `set_key()`/`unset_key()`). Project code confirmed not calling those functions; fix available at 1.2.2 when convenient.
- `pygments 2.19.2` MEDIUM (CVE-2026-4539): unchanged. Fix at 2.20.0.
- 3 net-new packages vs last scan (122→125): `Authlib 1.7.0`, `joserfc 1.6.4`, `dparse 0.6.4` — none flagged by safety-cli.

---

### Scan: 2026-05-10 — .venv installed packages, safety-cli 3.7.0

Vulnerability check 2026-05-10 — 128 packages scanned. No new vulnerabilities. 2 previously documented issues persist (pygments 2.19.2 MEDIUM, python-dotenv 1.2.1 MODERATE).

**Scope:** 128 .venv installed packages (safety-cli open-source database)

#### Notes
- `pygments 2.19.2` MEDIUM (CVE-2026-4539 / SFTY-20260322-35073): unchanged since 2026-04-26. Fix available at 2.20.0. Transitive/dev dependency; low exploitability in desktop app context.
- `python-dotenv 1.2.1` MODERATE (CVE-2026-28684 / GHSA-mf9w-mj56-hr94): unchanged since 2026-04-25. Project code does not call `set_key()`/`unset_key()` — unexploitable in current usage. Fix at 1.2.2.
- 3 net-new packages vs last scan (125→128): `annotated-doc 0.0.4`, `truck-telemetry 0.0.3`, `safety-schemas 0.0.16` — none flagged.
