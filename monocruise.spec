# PyInstaller spec for the MonoCruise main application.
# Build with:  pyinstaller monocruise.spec
#
# This produces dist/MonoCruise/MonoCruise.exe alongside its support files
# (one-folder mode — startup is significantly faster than one-file for a
# PySide6 app).

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Bundle every asset directory that ships with the app. Add new ones here
# when you introduce another directory of icons/sounds/etc.
datas = [
    ('ui/main_window/assets', 'ui/main_window/assets'),
    ('ui/popup/icons',          'ui/popup/icons'),
    ('ui/cc_panel/assets',      'ui/cc_panel/assets'),
    ('core/aeb/AEB_warning.wav','core/aeb'),
]

# PySide6 ships translations / Qt plugins that PyInstaller's stock hook
# misses occasionally; the hook below pulls in everything it knows about.
datas += collect_data_files('PySide6')

a = Analysis(
    ['monocruise.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
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
