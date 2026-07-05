# PyInstaller spec for the MonoCruise auto-updater.
# Build from the repo root with:  pyinstaller updater/updater.spec
#
# Produces dist/updater/updater.exe.

block_cipher = None

# PySide6 plugins/translations for the imported Qt modules (QtWidgets, QtSvg,
# QtMultimedia for the release-notes video player) come from PyInstaller's
# stock PySide6 hooks. Do NOT add collect_data_files('PySide6'): it copies
# the entire PySide6 package (WebEngine/Chromium, QML, all translations —
# ~600 MB) into the bundle.
datas = []

# Heavy Qt stacks the updater never touches.
QT_EXCLUDES = [
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebEngineQuick',
    'PySide6.QtWebChannel',
    'PySide6.QtQml',
    'PySide6.QtQuick',
    'PySide6.QtQuick3D',
    'PySide6.QtQuickWidgets',
    'PySide6.QtPdf',
    'PySide6.QtPdfWidgets',
    'PySide6.QtCharts',
    'PySide6.QtDataVisualization',
    'PySide6.Qt3DCore',
    'PySide6.Qt3DRender',
    'PySide6.QtLocation',
    'PySide6.QtPositioning',
    'PySide6.QtBluetooth',
    'PySide6.QtSensors',
    'PySide6.QtSerialPort',
    'PySide6.QtTest',
    'PySide6.QtDesigner',
]

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
    excludes=QT_EXCLUDES,
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
