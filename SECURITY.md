# Security Advisories

## Last Scan: 2026-05-24

Vulnerability check 2026-05-24 — All dependencies clean.

**Scope:** PySide6 6.10.1, psutil 7.1.3, pygame 2.6.1, shapely 2.1.2, requests 2.34.2, urllib3 2.7.0, pygments 2.20.0, pytest 9.0.3, pyinstaller 6.18.0

## Resolved Vulnerabilities

| Package | Fixed Version | Severity | Advisory | Summary |
|---------|--------------|----------|----------|---------|
| requests | 2.34.2 | MODERATE | [GHSA-gc5v-m9x4-r6x2](https://github.com/advisories/GHSA-gc5v-m9x4-r6x2) | Predictable temp filename in `extract_zipped_paths()` — symlink attack |
| urllib3 | 2.7.0 | **HIGH** (7.5) | [GHSA-mf9v-mfxr-j63j](https://github.com/advisories/GHSA-mf9v-mfxr-j63j) / CVE-2026-44432 | Decompression-bomb safeguards bypassed in streaming API (DoS) |
| urllib3 | 2.7.0 | MEDIUM (5.3) | [GHSA-qccp-gfcp-xxvc](https://github.com/advisories/GHSA-qccp-gfcp-xxvc) / CVE-2026-44431 | Sensitive headers forwarded across origins in proxied redirects |
| pygments | 2.20.0 | MEDIUM | CVE-2026-4539 | Security flaw in `AdlLexer` |
| pillow | 12.2.0 | **HIGH** | [GHSA-cfh3-3jmp-rvhc](https://github.com/advisories/GHSA-cfh3-3jmp-rvhc) | Out-of-bounds write loading crafted PSD images (RCE) |
| pillow | 12.2.0 | **HIGH** | [GHSA-whj4-6x5x-4v2j](https://github.com/advisories/GHSA-whj4-6x5x-4v2j) | FITS GZIP decompression bomb |
| mistune | 3.2.0 | **HIGH** | [GHSA-fw3v-x4f2-v673](https://github.com/advisories/GHSA-fw3v-x4f2-v673) | ReDoS via catastrophic regex backtracking |
