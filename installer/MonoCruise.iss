; Inno Setup script for the MonoCruise bootstrap installer.
;
; The CI workflow compiles this with:
;     iscc /DMyAppVersion=X.Y.Z installer\MonoCruise.iss
;
; Expectations on the working tree at compile time:
;   - dist\MonoCruise\         (PyInstaller output for the main app)
;   - dist\updater\            (PyInstaller output for the auto-updater)
;   - dist\MonoCruiseChecker\  (PyInstaller output for the background checker)
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
AppPublisher=LD-Tech
AppPublisherURL=https://github.com/luuukske/MonoCruise
; Per-user install ({autopf} resolves to {localappdata}\Programs in lowest
; mode). The app writes config.json/installed_version.txt and the updater
; writes whole releases into the install dir — under an admin-owned
; Program Files none of that works from an unelevated process.
DefaultDirName={autopf}\MonoCruise
DefaultGroupName=MonoCruise
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=MonoCruise installer
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\MonoCruise.exe
UninstallDisplayName=MonoCruise {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"
; The checker is only added to Windows startup when the user leaves this
; checked: the Run key is written by the installer (visible, consent-based)
; instead of by the app, and removed again on uninstall.
Name: "autostart"; Description: "Start MonoCruise automatically when ETS2/ATS is running (small background checker)"; GroupDescription: "Startup:"

[Files]
; Main application: PyInstaller one-folder output.
Source: "..\dist\MonoCruise\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Auto-updater in its own subdir with its own _internal: its files never
; collide with the app's, and the self-update swap (updater_pending/) can
; replace the whole folder. Matches updater.py:install_root ({app} = parent
; of the updater dir).
Source: "..\dist\updater\*";     DestDir: "{app}\updater"; Flags: recursesubdirs createallsubdirs ignoreversion
; Background game checker, own subdir with its own _internal. Deliberately not
; part of Update.zip (it runs at login and would hold locks the auto-updater
; does not manage), so it ships and updates via this installer only.
Source: "..\dist\MonoCruiseChecker\*"; DestDir: "{app}\checker"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\MonoCruise";        Filename: "{app}\MonoCruise.exe"
Name: "{group}\MonoCruise Updater"; Filename: "{app}\updater\updater.exe"
Name: "{group}\Uninstall MonoCruise"; Filename: "{uninstallexe}"
Name: "{autodesktop}\MonoCruise"; Filename: "{app}\MonoCruise.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "MonoCruiseChecker"; ValueData: """{app}\checker\MonoCruiseChecker.exe"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\MonoCruise.exe"; Description: "Launch MonoCruise"; Flags: nowait postinstall skipifsilent
; Start the checker right away so autostart works without a reboot.
Filename: "{app}\checker\MonoCruiseChecker.exe"; Tasks: autostart; Flags: nowait

[UninstallRun]
; The checker has no window, so it cannot be asked to close; stop it so its
; files can be deleted. Harmless when it is not running.
Filename: "{cmd}"; Parameters: "/C taskkill /F /IM MonoCruiseChecker.exe"; Flags: runhidden; RunOnceId: "StopChecker"

[UninstallDelete]
; Only remove things the installer placed (plus updater self-update leftovers).
; config.json and logs\ are left behind on uninstall so the user can keep
; their settings/history.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\updater"
Type: filesandordirs; Name: "{app}\checker"
Type: filesandordirs; Name: "{app}\updater_pending"
Type: filesandordirs; Name: "{app}\updater_old"
Type: filesandordirs; Name: "{app}\update_staging.tmp"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  { Upgrade over an existing install: the running checker holds a lock on its
    own exe and has no window for Restart Manager to close, so stop it before
    files are replaced. Harmless when it is not running. }
  Exec(ExpandConstant('{cmd}'), '/C taskkill /F /IM MonoCruiseChecker.exe', '',
       SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := '';
end;

