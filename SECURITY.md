# Security Advisories

## Last Scan: 2026-07-09

**Scope:** PySide6 6.11.1, psutil 7.2.2, pygame 2.6.1, requests 2.34.2, urllib3 2.7.0, packaging 26.2, pygments 2.20.0, pytest 9.1.1, pyinstaller 6.21.0, truck-telemetry 0.0.3, pillow 12.3.0 (+ transitive: idna 3.11, certifi 2026.1.4, charset-normalizer 3.4.4)

**Tools:** pip-audit 2.10.1 (OSV / PyPI advisories), safety 3.7.0 (Safety DB)

Declared dependencies in `requirements.txt` and `requirements-dev.txt` resolve with no known vulnerabilities. One open issue in the installed environment (stale transitive `idna` pulled in by `requests`); see Current Vulnerabilities.

## Current Vulnerabilities

| Date | Package | Version | Severity | Advisory | Fix Available |
|------|---------|---------|----------|----------|---------------|
| 2026-07-09 | idna (via requests) | 3.11 | MEDIUM | [GHSA-65pc-fj4g-8rjx](https://github.com/advisories/GHSA-65pc-fj4g-8rjx) / CVE-2026-45409 | Yes — upgrade to ≥3.15 (`pip install 'idna>=3.15'`) |

## Resolved Vulnerabilities

| Package | Fixed Version | Severity | Advisory | Summary |
|---------|--------------|----------|----------|---------|
| requests | 2.34.2 | MODERATE | [GHSA-gc5v-m9x4-r6x2](https://github.com/advisories/GHSA-gc5v-m9x4-r6x2) | Predictable temp filename in `extract_zipped_paths()`: symlink attack |
| urllib3 | 2.7.0 | **HIGH** (7.5) | [GHSA-mf9v-mfxr-j63j](https://github.com/advisories/GHSA-mf9v-mfxr-j63j) / CVE-2026-44432 | Decompression-bomb safeguards bypassed in streaming API (DoS) |
| urllib3 | 2.7.0 | MEDIUM (5.3) | [GHSA-qccp-gfcp-xxvc](https://github.com/advisories/GHSA-qccp-gfcp-xxvc) / CVE-2026-44431 | Sensitive headers forwarded across origins in proxied redirects |
| pygments | 2.20.0 | MEDIUM | CVE-2026-4539 | Security flaw in `AdlLexer` |
| pillow | ≥ 12.2.0 | **HIGH** (8.9) | [GHSA-cfh3-3jmp-rvhc](https://github.com/advisories/GHSA-cfh3-3jmp-rvhc) / CVE-2026-25990 | Out-of-bounds write when loading crafted PSD images |
| pillow | ≥ 12.2.0 | **HIGH** (8.7) | [GHSA-whj4-6x5x-4v2j](https://github.com/advisories/GHSA-whj4-6x5x-4v2j) / CVE-2026-40192 | FITS GZIP decompression bomb: unbounded memory consumption |
| pillow | ≥ 12.2.0 | MEDIUM (5.1) | [GHSA-wjx4-4jcj-g98j](https://github.com/advisories/GHSA-wjx4-4jcj-g98j) / CVE-2026-42308 | Integer overflow in font glyph position tracking during text rendering |
| mistune | 3.2.0 | **HIGH** | [GHSA-fw3v-x4f2-v673](https://github.com/advisories/GHSA-fw3v-x4f2-v673) | ReDoS via catastrophic regex backtracking |
