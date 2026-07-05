# PyInstaller spec for the MonoCruise auto-updater.
# Build from the repo root with:  pyinstaller updater/updater.spec
#
# Produces dist/updater/updater.exe.

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = collect_data_files('PySide6')

# PyInstaller 6 resolves relative spec paths against the spec file's
# directory, not the invocation cwd ('updater/updater.py' stopped resolving).
# SPECPATH (provided by PyInstaller) makes the intent explicit either way.
import os
_REPO_ROOT = os.path.dirname(SPECPATH)

a = Analysis(
    [os.path.join(SPECPATH, 'updater.py')],
    # Repo root is on the path so the shared UI library (shared/) is found
    # and bundled; the updater imports it via `from shared import ...`.
    pathex=[SPECPATH, _REPO_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'shared',
        'shared.theme',
        'shared.markdown_renderer',
        'shared.dropdown',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='updater',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='updater',
)
