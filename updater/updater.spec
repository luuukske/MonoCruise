# PyInstaller spec for the MonoCruise auto-updater.
# Build from the repo root with:  pyinstaller updater/updater.spec
#
# Produces dist/updater/updater.exe.

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = collect_data_files('PySide6')

a = Analysis(
    ['updater/updater.py'],
    # Repo root ('.') is on the path so the shared UI library (shared/) is found
    # and bundled; the updater imports it via `from shared import ...`.
    pathex=['updater', '.'],
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
