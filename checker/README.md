# MonoCruise background checker

`MonoCruiseChecker.exe` is a small optional helper that starts MonoCruise
automatically when ETS2/ATS is running. The installer offers it as a checkbox
("start MonoCruise automatically..."); nothing is added to Windows startup
without that consent, and the uninstaller removes the entry again.

## How it works

Once per second it tries to open the SCS telemetry shared-memory block
(`Local\SCSTelemetry`) that the in-game SDK plugin publishes. That block only
exists while the game is running, so this is a single cheap OS call — no
process listing, no polling of `tasklist`. When the game comes up and
MonoCruise is not already open (detected via the named mutex MonoCruise holds
while running, see `monocruise.py`), it launches `MonoCruise.exe` once for
that game session. Everything it does is written to `checker.log` next to the
exe.

It makes no network connections, reads no personal data, and touches no
registry keys. See the module docstring in [ets2_checker.py](ets2_checker.py)
for the full behaviour contract.

## Antivirus notes

The previous generation of this checker was occasionally flagged by AV
heuristics (`Trojan:Win32/Ravartar!rfn` style detections — behavioural, not a
signature match). The combination that tripped the heuristics: `shell=True`
process enumeration via `tasklist`, the app writing its own
`HKCU\...\CurrentVersion\Run` key, a silent background process that launches
other exes, and PyInstaller packaging. This rewrite removes every one of those
signals that can be removed:

- game detection opens existing shared memory instead of enumerating processes
- single-instance and "is MonoCruise open?" checks use named mutexes
  (`CreateMutexW`/`OpenMutexW`), no `subprocess`+`shell=True` anywhere
- the startup Run key is written by the (visible, consent-based) installer,
  never by this program
- the exe carries a proper version resource (`version_info.txt`) so it is not
  an anonymous binary
- one-folder PyInstaller build, no UPX, no one-file self-extraction

The remaining heuristic risk is the unsigned PyInstaller packaging itself;
code signing is the only real fix for that (plumbing already stubbed in
`.github/workflows/release.yml`).

## Build

```
pyinstaller --noconfirm checker/checker.spec
```

Output lands in `dist/MonoCruiseChecker/`. The installer places it at
`<install root>/checker/`. It is deliberately **not** part of `Update.zip`:
the checker runs at login and would hold file locks the auto-updater does not
manage, and it changes rarely — it ships and updates via the installer only.
