; Inno Setup Installer for Ollama TurboQuant Fork
;
; Build with: docker run --rm -v .:/work amake/innosetup app/ollama-turboquant.iss

#define MyAppName "Ollama"
#define MyAppVersion "2.0.2-turboquant"
#define MyAppPublisher "Ollama TurboQuant Fork"
#define MyAppURL "https://github.com/amarce/ollama"
#define MyAppExeName "ollama.exe"
#define MyIcon ".\assets\app.ico"

[Setup]
AppId={{44E83376-CE68-45EB-8FC1-393500EB558C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion=2.0.2.0
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=OllamaSetup-turboquant
SetupIconFile={#MyIcon}
UninstallDisplayIcon={uninstallexe}
Compression=lzma2/ultra64
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
OutputDir=..\dist\

SetupLogging=yes
CloseApplications=no
RestartApplications=no
RestartIfNeededByRun=no

WizardSmallImageFile=.\assets\setup.bmp

MinVersion=10.0.10240

DisableDirPage=yes
DisableFinishedPage=yes
DisableReadyMemo=yes
DisableReadyPage=yes
DisableStartupPrompt=yes

SetupMutex=OllamaSetupMutex

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[LangOptions]
DialogFontSize=12

#define MyTrayAppExeName "ollama app.exe"

[Files]
Source: "..\dist\windows-ollama-app-amd64.exe"; DestDir: "{app}"; DestName: "{#MyTrayAppExeName}"; Flags: ignoreversion 64bit; BeforeInstall: TaskKill('{#MyTrayAppExeName}')
Source: "..\dist\windows-amd64\vc_redist.x64.exe"; DestDir: "{tmp}"; Check: vc_redist_needed(); Flags: deleteafterinstall
Source: "..\dist\windows-amd64\ollama.exe"; DestDir: "{app}"; Flags: ignoreversion 64bit; BeforeInstall: TaskKill('{#MyAppExeName}')
Source: ".\assets\app.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyTrayAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{app}\lib\{#MyAppName}"; Filename: "{app}\{#MyTrayAppExeName}"; IconFilename: "{app}\app.ico"
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyTrayAppExeName}"; IconFilename: "{app}\app.ico"

[InstallDelete]
Type: filesandordirs; Name: "{%TEMP}\ollama*"
Type: filesandordirs; Name: "{app}\lib\ollama"
Type: files; Name: "{%LOCALAPPDATA}\Ollama\updates"

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /passive /norestart"; Check: vc_redist_needed(); StatusMsg: "Installing VC++ Redistributables..."; Flags: waituntilterminated
Filename: "{cmd}"; Parameters: "/C set PATH={app};%PATH% & ""{app}\{#MyTrayAppExeName}"""; Flags: postinstall nowait runhidden

[UninstallRun]
Filename: "taskkill"; Parameters: "/im ""{#MyTrayAppExeName}"" /f /t"; Flags: runhidden
Filename: "taskkill"; Parameters: "/im ""{#MyAppExeName}"" /f /t"; Flags: runhidden
Filename: "{cmd}"; Parameters: "/c timeout 5"; Flags: runhidden

[UninstallDelete]
Type: filesandordirs; Name: "{%TEMP}\ollama*"
Type: filesandordirs; Name: "{%LOCALAPPDATA}\Ollama"
Type: filesandordirs; Name: "{%LOCALAPPDATA}\Programs\Ollama"
Type: filesandordirs; Name: "{%USERPROFILE}\.ollama\history"

[Messages]
WizardReady=Ollama TurboQuant
ReadyLabel1=%nLet's get you up and running with your own large language models.%n%nTurboQuant KV cache compression auto-enables on NVIDIA CUDA GPUs.

[Registry]
Root: HKCU; Subkey: "Environment"; \
    ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
    Check: NeedsAddPath('{app}')
Root: HKCU; Subkey: "Software\Classes\ollama"; ValueType: string; ValueName: ""; ValueData: "URL:Ollama Protocol"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\ollama"; ValueType: string; ValueName: "URL Protocol"; ValueData: ""; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\ollama\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey

[Code]

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER,
    'Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + ExpandConstant(Param) + ';', ';' + OrigPath + ';') = 0;
end;

const VCRTL_MIN_V1 = 14;
const VCRTL_MIN_V2 = 40;
const VCRTL_MIN_V3 = 33807;
const VCRTL_MIN_V4 = 0;

function vc_redist_needed(): Boolean;
var
  sRegKey: string;
  v1, v2, v3, v4: Cardinal;
begin
  sRegKey := 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  if (RegQueryDWordValue(HKEY_LOCAL_MACHINE, sRegKey, 'Major', v1) and
      RegQueryDWordValue(HKEY_LOCAL_MACHINE, sRegKey, 'Minor', v2) and
      RegQueryDWordValue(HKEY_LOCAL_MACHINE, sRegKey, 'Bld', v3) and
      RegQueryDWordValue(HKEY_LOCAL_MACHINE, sRegKey, 'RBld', v4)) then
  begin
    Result := not (
        (v1 > VCRTL_MIN_V1) or ((v1 = VCRTL_MIN_V1) and
         ((v2 > VCRTL_MIN_V2) or ((v2 = VCRTL_MIN_V2) and
          ((v3 > VCRTL_MIN_V3) or ((v3 = VCRTL_MIN_V3) and
           (v4 >= VCRTL_MIN_V4)))))));
  end
  else
    Result := TRUE;
end;

procedure TaskKill(FileName: String);
var
  ResultCode: Integer;
begin
    Exec('taskkill.exe', '/f /im ' + '"' + FileName + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
