#define AppName "ChatKodi"
#define AppVersion GetEnv("CHATKODI_BUILD_VERSION")
#if AppVersion == ""
#define AppVersion "2.0.0"
#endif
#define SourceDir GetEnv("CHATKODI_DIST_DIR")
#if SourceDir == ""
#define SourceDir "..\dist\ChatKodi"
#endif

[Setup]
AppId={{A4B7D8F0-79C8-4E5B-89C8-8A7F4D0180C8}}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\ChatKodi
DefaultGroupName=ChatKodi
OutputDir=..\release
OutputBaseFilename=ChatKodi-{#AppVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
UninstallDisplayIcon={app}\ChatKodiLauncher.exe
SetupLogging=yes

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ChatKodi"; Filename: "{app}\ChatKodiLauncher.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\ChatKodi"; Filename: "{app}\ChatKodiLauncher.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\ChatKodiLauncher.exe"; Description: "Launch ChatKodi"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
