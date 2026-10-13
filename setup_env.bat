@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  MusicRecorder 运行环境一键安装（.NET 8 SDK + Python 3.10+）
rem
rem  给第一次 clone 本仓库、机器上什么开发环境都没有的人：
rem    1. 自动检测已有的 .NET 8 SDK / Python（能识别 Microsoft Store 假别名）
rem    2. 缺什么装什么：优先 winget；不可用 / 失败时自动回退到免管理员方案
rem         .NET  → dotnet-install.ps1（用户级，装到 %LOCALAPPDATA%，无需 UAC）
rem         Python→ python.org 官方安装包静默安装（InstallAllUsers=0，无需 UAC）
rem    3. 装完在当前窗口立即生效（从注册表合并机器 + 用户 PATH），随后自动构建
rem
rem  用法：
rem    setup_env.bat              检测 → 安装缺失环境 → 调用 install.bat 构建
rem    setup_env.bat --check      仅检测并报告，不安装、不构建
rem    setup_env.bat --no-build   只安装环境，不自动构建
rem    setup_env.bat --no-pause   结束时不暂停（自动化 / CI 用）
rem
rem  备注：Python 仅「MT3 模型导出」(install_mt3.bat) 需要；
rem        BasicPitch / Madmom 的构建运行只需 .NET。
rem ============================================================

set "ROOT=%~dp0"
set "MODE=%~1"
set "EXTRA=%~2"
set "ENVCHECK=%TEMP%\mr_envcheck.txt"
set "NOPAUSE="
set "NOBUILD="
if /i "%MODE%"=="--no-pause"   set "NOPAUSE=1"
if /i "%EXTRA%"=="--no-pause"  set "NOPAUSE=1"
if /i "%MODE%"=="--no-build"   set "NOBUILD=1"
if /i "%EXTRA%"=="--no-build"  set "NOBUILD=1"

pushd "%ROOT%"

echo ================================================
echo  MusicRecorder 环境检测与安装
echo ================================================
echo.

rem ---------- 检测 ----------
call :detect
echo [检测结果]
if "%FOUND_DOTNET%"=="1" (
    echo   .NET 8 SDK ........... 已安装 ^(8.x^)
) else (
    echo   .NET 8 SDK ........... 未安装
)
if defined FOUND_PY (
    echo   Python ^(^>=3.10^) ...... 已安装 %FOUND_PYVER%
    echo                            %FOUND_PY%
) else (
    echo   Python ^(^>=3.10^) ...... 未安装 ^(已忽略 Microsoft Store 占位别名^)
)
echo.

if /i "%MODE%"=="--check" goto :end_ok

if "%FOUND_DOTNET%"=="1" if defined FOUND_PY goto :env_ready

winget --version >nul 2>nul
if not errorlevel 1 set "HAS_WINGET=1"

rem ---------- 安装 .NET 8 SDK ----------
if "%FOUND_DOTNET%"=="1" goto :dotnet_ok
echo [安装] .NET 8 SDK ...
if defined HAS_WINGET (
    echo        方式 1/2：winget（系统级安装，若弹 UAC 请点「是」）...
    winget install --id Microsoft.DotNet.SDK.8 -e --source winget --accept-source-agreements --accept-package-agreements
    call :refresh_path
    call :detect
)
if "%FOUND_DOTNET%"=="1" goto :dotnet_ok
echo        方式 2/2：官方 dotnet-install 脚本（用户级，无需管理员）...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $dir=Join-Path $env:LOCALAPPDATA 'Microsoft\dotnet'; $s=Join-Path $env:TEMP 'mr_dotnet_install.ps1'; Invoke-WebRequest -UseBasicParsing 'https://dot.net/v1/dotnet-install.ps1' -OutFile $s; & $s -Channel 8.0 -InstallDir $dir; $up=[Environment]::GetEnvironmentVariable('Path','User'); if(-not $up){$up=''}; if($up -notlike ('*'+$dir+'*')){[Environment]::SetEnvironmentVariable('Path',($up.TrimEnd(';')+';'+$dir),'User')}"
if errorlevel 1 goto :dotnet_fail
call :refresh_path
call :detect
if not "%FOUND_DOTNET%"=="1" goto :dotnet_fail
:dotnet_ok
echo        [OK] .NET 8 SDK 就绪。

rem ---------- 安装 Python 3.11 ----------
if defined FOUND_PY goto :python_ok
echo.
echo [安装] Python 3.11（3.10+ 均可，选 3.11 兼容性最好）...
if defined HAS_WINGET (
    echo        方式 1/2：winget（用户级安装）...
    winget install --id Python.Python.3.11 -e --source winget --scope user --accept-source-agreements --accept-package-agreements
    call :refresh_path
    call :detect
)
if defined FOUND_PY goto :python_ok
echo        方式 2/2：python.org 官方包静默安装（用户级，无需管理员）...
echo        正在下载，请稍候 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $url='https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'; $out=Join-Path $env:TEMP 'mr_python-3.11.9-amd64.exe'; Invoke-WebRequest -UseBasicParsing $url -OutFile $out; $a=@('/quiet','InstallAllUsers=0','PrependPath=1','Include_pip=1','Include_launcher=1','InstallLauncherAllUsers=0'); $pr=Start-Process -FilePath $out -ArgumentList $a -Wait -PassThru; if($pr.ExitCode -ne 0 -and $pr.ExitCode -ne 3010){exit 1}"
if errorlevel 1 goto :python_fail
call :refresh_path
call :detect
if not defined FOUND_PY goto :python_fail
:python_ok
echo        [OK] Python %FOUND_PYVER% 就绪。

rem ---------- 完成，开始构建 ----------
:env_ready
echo.
echo [完成] 运行环境已就绪：.NET 8 SDK + Python 3.10+。
if defined NOBUILD goto :end_ok
echo.
echo 接下来执行 install.bat 构建全部工程并运行冒烟自检 ...
echo.
call "%ROOT%install.bat"
if errorlevel 1 goto :end_fail
echo.
echo 提示：需要 MT3 多乐器转谱时，再运行 install_mt3.bat（下载并导出约 350 MB 模型）。
goto :end_ok

rem ============================================================
rem  子程序：环境检测（结果写入 FOUND_DOTNET / FOUND_PY / FOUND_PYVER）
rem  候选顺序：py -3 → PATH 中的真 python（排除 WindowsApps 假别名）→ 常见用户目录
rem ============================================================
:detect
set "FOUND_DOTNET=0"
set "FOUND_PY="
set "FOUND_PYVER="
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $dn=0; try{ $sl=((& dotnet --list-sdks) -join [Environment]::NewLine); if([regex]::IsMatch($sl,'(?m)^8\.')){$dn=1} }catch{}; $cands=@(); $cands+=,@('py',@('-3')); try{ $gc=Get-Command python | Where-Object { $_.Source -notmatch 'WindowsApps' } | Select-Object -First 1; if($gc){$cands+=,@($gc.Source,@())} }catch{}; foreach($d in @('Python313','Python312','Python311','Python310')){ $p=Join-Path $env:LOCALAPPDATA ('Programs\Python\'+$d+'\python.exe'); if(Test-Path $p){$cands+=,@($p,@())} }; $fp=''; $fv=''; foreach($c in $cands){ try{ $v=[string](& $c[0] @($c[1]) --version) }catch{ $v='' }; if($v -match 'Python 3\.(\d+)' -and [int]$matches[1] -ge 10){ $fp=$c[0]; if($fp -eq 'py'){$fp='py -3'}; $fv=($v -replace 'Python ','').Trim(); break } }; 'DOTNET='+$dn; 'PY='+$fp; 'PYVER='+$fv" > "%ENVCHECK%" 2>nul
for /f "usebackq delims=" %%L in ("%ENVCHECK%") do set "FOUND_%%L"
exit /b 0

rem ============================================================
rem  子程序：从注册表重新合并 PATH（机器 + 用户），让新装软件本窗口可用
rem ============================================================
:refresh_path
powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')" > "%ENVCHECK%" 2>nul
set "NEWPATH="
set /p "NEWPATH=" < "%ENVCHECK%"
if defined NEWPATH set "PATH=%NEWPATH%"
exit /b 0

:dotnet_fail
echo        [错误] .NET 8 SDK 安装失败，请手动安装：
echo               https://dotnet.microsoft.com/download/dotnet/8.0
goto :end_fail

:python_fail
echo        [错误] Python 安装失败，请手动安装 Python 3.10+ 并勾选 "Add python.exe to PATH"：
echo               https://www.python.org/downloads/
goto :end_fail

:end_ok
popd
if not defined NOPAUSE (
    echo.
    pause
)
endlocal
exit /b 0

:end_fail
popd
echo.
echo [失败] 环境安装未完成，请根据上方提示处理后重新运行本脚本。
if not defined NOPAUSE (
    echo.
    pause
)
endlocal
exit /b 1