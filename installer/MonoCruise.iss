; Inno Setup script for the MonoCruise bootstrap installer.
;
; The CI workflow compiles this with:
;     iscc /DMyAppVersion=X.Y.Z installer\MonoCruise.iss
;
; Expectations on the working tree at compile time:
;   - dist\MonoCruise\        (PyInstaller output for the main app)
;   - dist\updater\           (PyInstaller output for the auto-updater)
;
; Output: installer\output\MonoCruise installer.exe
;
; Files NOT listed in [Files] are never touched by the installer or
; uninstaller: that is how config.json and logs\ survive upgrades.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif

[Setup]
AppName=MonoCruise
AppVersion={#MyAppVersion}
AppPublisher=luuukske
AppPublisherURL=https://github.com/luuukske/MonoCruise
DefaultDirName={autopf}\MonoCruise
DefaultGroupName=MonoCruise
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=MonoCruise installer
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
UninstallDisplayIcon={app}\MonoCruise.exe
UninstallDisplayName=MonoCruise {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Main application: PyInstaller one-folder output.
Source: "..\dist\MonoCruise\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Auto-updater lives in the same install dir so updater.py:install_dir resolves correctly.
Source: "..\dist\updater\*";     DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\MonoCruise";        Filename: "{app}\MonoCruise.exe"
Name: "{group}\MonoCruise Updater"; Filename: "{app}\updater.exe"
Name: "{group}\Uninstall MonoCruise"; Filename: "{uninstallexe}"
Name: "{commondesktop}\MonoCruise"; Filename: "{app}\MonoCruise.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MonoCruise.exe"; Description: "Launch MonoCruise"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Only remove things the installer placed. config.json and logs\ are left
; behind on uninstall so the user can keep their settings/history.
Type: filesandordirs; Name: "{app}\_internal"

