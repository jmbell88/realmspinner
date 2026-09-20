#ifndef AppVersion
  #error AppVersion must be supplied by installer/build.ps1
#endif
#ifndef StageDir
  #error StageDir must be supplied by installer/build.ps1
#endif

#define ProjectRoot SourcePath + "\.."

[Setup]
; A *new* AppId for the 2026-09-19 rename, deliberately. Inno keys an in-place
; upgrade off this GUID, so keeping the old one would have left the product
; installed in a directory named after a name it no longer has, with one
; Add/Remove Programs entry whose display name changed under the user. A new
; identity installs cleanly into {localappdata}\Programs\Realmspinner;
; PrepareToInstall below is what stops that leaving two entries behind.
AppId={{580F8E4B-F507-4CEA-B6B5-A92DA1A70FDC}
AppName=Realmspinner
AppVersion={#AppVersion}
AppPublisher=Realmspinner
DefaultDirName={localappdata}\Programs\Realmspinner
DefaultGroupName=Realmspinner
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
DiskSpanning=no
DiskSliceSize=2100000000
OutputDir={#ProjectRoot}\dist
OutputBaseFilename=RealmspinnerSetup-v{#AppVersion}
; Shown as the wizard's first page. Without it Inno Setup shows the user no
; terms at all, which -- for everyone who installs rather than clones -- is the
; same posture as having no licence. The third-party notices are not shown
; here (a wizard pane renders Markdown as raw text); they are staged into the
; install root and beside vendor\, which is what MIT requires for the two
; binaries this installer still carries (gltfpack, realmspinnerc) -- see
; installer\build.ps1. The reconstruction engine (trellis.cpp/ggml, MIT, plus
; the NVIDIA CUDA redistributables under NVIDIA's own EULA) no longer travels
; with the installer as of 2026-09-10 -- it is a Settings -> Models download,
; fetched from trellis.cpp's own release page -- so this project no longer
; redistributes it and the notice is documentation rather than an obligation.
; THIRD-PARTY-NOTICES.md still describes it, and is staged either way.
LicenseFile={#ProjectRoot}\LICENSE
; A genuine multi-size Windows ICO (16-256 px), and it has to be a separate
; file: src\realmspinner\assets\icon.ico is a 1024x1024 PNG despite its extension,
; which pygame loads happily (it reads content, not names) and Inno Setup
; cannot turn into an icon resource at all -- it failed the first-ever compile
; with "Resource update error: File is too large". The two cannot be merged:
; SDL2's ICO loader rejects 32-bit RGBA ICOs, so a real .ico would leave the
; app window with no icon. Regenerate with Pillow from the PNG beside it.
SetupIconFile={#ProjectRoot}\installer\realmspinner.ico
; Staged into the install root by installer\build.ps1, so Add/Remove Programs
; has an icon to show -- it previously pointed at the PNG above and showed none.
UninstallDisplayIcon={app}\realmspinner.ico
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\python\Lib\site-packages"
Type: filesandordirs; Name: "{app}\src"
; The bundled wheels of the *previous* version. Their filenames carry a
; version, so an upgrade would otherwise leave last release's docopt beside
; this one's -- files nothing will ever install, pinned by a manifest that no
; longer names them. packs.json is replaced in place by [Files].
Type: filesandordirs; Name: "{app}\packs"
; The 2026-09-15 audit (pipelines-01): build.ps1 stopped staging vendor\trellis
; on 2026-09-10 -- the engine is a Settings -> Models download now -- but this
; list was never told, so an in-place upgrade from <=0.0.41 left the old 838 MB
; engine sitting in {app}\vendor\trellis forever. Worse than dead weight:
; Config.resolve_trellis_exe() falls back to exactly that path, so the stale,
; unpinned engine from the previous release keeps being *used* on every
; upgraded install that never explicitly downloads engine:trellis_runtime.
Type: filesandordirs; Name: "{app}\vendor\trellis"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Realmspinner"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m realmspinner"; WorkingDir: "{app}"; IconFilename: "{app}\realmspinner.ico"
Name: "{group}\Realmspinner Doctor"; Filename: "{app}\bin\realmspinner-doctor.cmd"; WorkingDir: "{app}"
Name: "{userdesktop}\Realmspinner"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m realmspinner"; WorkingDir: "{app}"; IconFilename: "{app}\realmspinner.ico"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\python"

[Code]
// The AppId this product shipped under as Warlock Studio, up to and including
// 0.0.51. Inno registers its uninstaller under "<AppId>_is1", so this is the
// key to look for. A literal rather than something derived: it is a historical
// fact about already-installed copies, and it must not follow AppId above the
// next time that changes.
const
  LegacyAppId = '{C64355D5-8A1F-4A10-8DBB-7E72BCE2C297}_is1';

function LegacyUninstaller(): String;
var
  Key: String;
  Value: String;
begin
  Result := '';
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + LegacyAppId;
  // Per-user install (PrivilegesRequired=lowest), so HKCU is where it lives --
  // but an older machine may carry a per-machine copy, so both are asked.
  if RegQueryStringValue(HKEY_CURRENT_USER, Key, 'QuietUninstallString', Value) then
    Result := Value
  else if RegQueryStringValue(HKEY_LOCAL_MACHINE, Key, 'QuietUninstallString', Value) then
    Result := Value;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Command: String;
  ExecName: String;
  Params: String;
  ResultCode: Integer;
  Cut: Integer;
begin
  // Never fails the install. A machine that cannot remove the old product is
  // still a machine that can have the new one -- the cost is one stale
  // Add/Remove Programs entry, which is not worth refusing an install over.
  Result := '';
  Command := LegacyUninstaller();
  if Command = '' then
    exit;

  if MsgBox(
       'Warlock Studio is installed on this PC. It was renamed to Realmspinner,'
       + ' so this is the same program under a new name rather than a second copy.'
       + #13#10#13#10
       + 'Remove Warlock Studio now?'
       + #13#10#13#10
       + 'Your assets and downloaded models are not touched either way --'
       + ' Realmspinner moves them to their new home the first time you run it.',
       mbConfirmation, MB_YESNO) <> IDYES then
    exit;

  // QuietUninstallString is already '"<path>\unins000.exe" /SILENT'; split it
  // so Exec gets the executable and its arguments apart, the way it wants them.
  ExecName := Command;
  Params := '';
  if Copy(ExecName, 1, 1) = '"' then
  begin
    Delete(ExecName, 1, 1);
    Cut := Pos('"', ExecName);
    if Cut > 0 then
    begin
      Params := Trim(Copy(ExecName, Cut + 1, Length(ExecName)));
      ExecName := Copy(ExecName, 1, Cut - 1);
    end;
  end
  else
  begin
    Cut := Pos(' ', ExecName);
    if Cut > 0 then
    begin
      Params := Trim(Copy(ExecName, Cut + 1, Length(ExecName)));
      ExecName := Copy(ExecName, 1, Cut - 1);
    end;
  end;

  if Params = '' then
    Params := '/SILENT';
  if Pos('/NORESTART', Uppercase(Params)) = 0 then
    Params := Params + ' /NORESTART';

  // Waited on, not fired and forgotten: the old uninstaller deletes its own
  // install tree, and letting that race this installer's file copy is how a
  // fresh install ends up missing files it had already written.
  Exec(ExecName, Params, '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataPath: String;
  HomeOverride: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // The 2026-09-07 audit found this always naming %USERPROFILE%\.realmspinner,
    // ignoring REALMSPINNER_HOME -- so a user who relocated their data root was
    // told it survived at a path it had never lived at. Mirrors
    // config._home()'s precedence (REALMSPINNER_HOME wins when set); that
    // function is the one to read if the precedence ever changes, not this.
    HomeOverride := GetEnv('REALMSPINNER_HOME');
    if HomeOverride <> '' then
      DataPath := HomeOverride
    else
      DataPath := AddBackslash(GetEnv('USERPROFILE')) + '.realmspinner';
    MsgBox(
      'Realmspinner was removed. Your assets and downloaded models remain at ' + DataPath + '.',
      mbInformation,
      MB_OK
    );
  end;
end;
