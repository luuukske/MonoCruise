# PyInstaller spec for the MonoCruise main application.
# Build with:  pyinstaller monocruise.spec
#
# This produces dist/MonoCruise/MonoCruise.exe alongside its support files
# (one-folder mode: startup is significantly faster than one-file for a
# PySide6 app).

block_cipher = None

# Bundle every asset directory that ships with the app. Add new ones here
# when you introduce another directory of icons/sounds/etc.
#
# PySide6 plugins/translations for the Qt modules the app imports are pulled
# in by PyInstaller's stock PySide6 hooks. Do NOT add
# collect_data_files('PySide6') here: it copies the entire PySide6 package
# (WebEngine/Chromium, QML, all translations — ~600 MB) into the bundle.
datas = [
    ('ui/main_window/assets', 'ui/main_window/assets'),
    ('ui/popup/icons',          'ui/popup/icons'),
    ('ui/cc_panel/assets',      'ui/cc_panel/assets'),
    ('core/aeb/AEB_warning.wav','core/aeb'),
]

# The app only uses QtCore/QtGui/QtWidgets/QtSvgWidgets; keep the heavy Qt
# stacks out even if some transitive import mentions them.
QT_EXCLUDES = [
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebEngineQuick',
    'PySide6.QtWebChannel',
    'PySide6.QtQml',
    'PySide6.QtQuick',
    'PySide6.QtQuick3D',
    'PySide6.QtQuickWidgets',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
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

a = Analysis(
    ['monocruise.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
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
    name='MonoCruise',
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
    name='MonoCruise',
)

