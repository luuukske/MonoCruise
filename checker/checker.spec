# PyInstaller spec for the MonoCruise background game checker.
# Build from the repo root with:  pyinstaller checker/checker.spec
#
# Produces dist/MonoCruiseChecker/ (one-folder mode: one-file exes unpack
# themselves at runtime, which both slows startup and trips AV heuristics).
#
# The checker is stdlib-only on purpose: a small import surface keeps the
# bundle tiny and the exe easy to audit. Do not add third-party imports to
# ets2_checker.py without a very good reason.
#
# Paths are relative to this spec file's directory (PyInstaller 6 behaviour).

import os

a = Analysis(
    ['ets2_checker.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MonoCruiseChecker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(SPECPATH, '..', 'icon.ico'),
    version='version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MonoCruiseChecker',
)
