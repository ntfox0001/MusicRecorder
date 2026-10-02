@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  MusicRecorder 一键安装脚本
rem
rem  适用：第一次 clone 本仓库后，一条命令完成环境检查 + 构建 + 自检。
rem  用法：install.bat
rem
rem  说明：
rem    - 本脚本只依赖 .NET 8 SDK，不需要 Python，也不需要 madmom。
rem      BasicPitch / Madmom 的模型参数已随源码分发（NmpIr.g.cs / *Ir.g.cs）。
rem    - MT3（多乐器转谱）需要额外的 ONNX 模型文件（约 350 MB），
rem      体积过大未随仓库分发，请另行运行 install_mt3.bat。
rem ============================================================

set "ROOT=%~dp0"
pushd "%ROOT%"

echo ================================================
echo  MusicRecorder 安装
echo ================================================
echo.

rem ---------- 1/4 检查 .NET SDK ----------
echo [1/4] 检查 .NET SDK ...
where dotnet >nul 2>nul
if errorlevel 1 (
    echo       [错误] 未找到 dotnet 命令。
    echo              请先安装 .NET 8 SDK：https://dotnet.microsoft.com/download/dotnet/8.0
    goto :fail
)
for /f "tokens=*" %%V in ('dotnet --version') do set "DOTNETVER=%%V"
echo       dotnet 版本：%DOTNETVER%
dotnet --list-sdks | findstr /b "8." >nul
if errorlevel 1 (
    echo       [警告] 未检测到 .NET 8 SDK。本工程目标框架为 net8.0 / net8.0-windows，
    echo              缺少 8.x SDK 时构建会失败。
)

rem ---------- 2/4 构建全部工程 ----------
echo.
echo [2/4] 构建全部工程（Release）...
set "FAILED=0"
for %%P in (BasicPitch Mt3 Madmom Madmom.Test Madmom.EndToEnd BasicPitchSharp.Test Mp3ToSheet.Bench Mp3ToSheet) do (
    echo.
    echo       --- %%P ---
    dotnet build "%%P\%%P.csproj" -c Release --nologo
    if errorlevel 1 set "FAILED=1"
)
if "%FAILED%"=="1" (
    echo.
    echo       [错误] 部分工程构建失败，请查看上方输出。
    goto :fail
)

rem ---------- 3/4 冒烟自检 ----------
echo.
echo [3/4] 运行冒烟自检（Madmom.Test：合成权重，不依赖任何模型文件）...
dotnet run --project Madmom.Test -c Release --no-build
if errorlevel 1 (
    echo.
    echo       [错误] 冒烟自检未通过。
    goto :fail
)

rem ---------- 4/4 检查模型 / 资源 ----------
echo.
echo [4/4] 检查模型 / 资源 ...
set "MISSING=0"

if exist "%ROOT%BasicPitch\nmp.onnx" (
    echo       [OK]   BasicPitch\nmp.onnx
) else (
    echo       [缺失] BasicPitch\nmp.onnx
    set "MISSING=1"
)

if exist "%ROOT%Madmom\MadmomDownBeatIr.g.cs" (
    if exist "%ROOT%Madmom\MadmomBeatIr.g.cs" (
        echo       [OK]   Madmom 权重（MadmomDownBeatIr.g.cs / MadmomBeatIr.g.cs）
    ) else (
        echo       [缺失] Madmom\MadmomBeatIr.g.cs
        set "MISSING=1"
    )
) else (
    echo       [缺失] Madmom\MadmomDownBeatIr.g.cs
    set "MISSING=1"
)

if exist "%ROOT%mt3_onnx\mt3_encoder.onnx" (
    echo       [OK]   mt3_onnx\（MT3 模型已就绪）
) else (
    echo       [提示] 未找到 mt3_onnx\：MT3 多乐器转谱需要额外的 ONNX 模型，
    echo              请运行 install_mt3.bat（下载 checkpoint 并导出，约 350 MB）。
)

echo.
if "%MISSING%"=="1" (
    echo [警告] 有资源缺失（正常 clone 不会出现，请检查仓库完整性）。
) else (
    echo [完成] 安装成功，仓库已就绪。
)
echo.
echo 产物：
echo   BasicPitch\bin\Release\netstandard2.1\BasicPitch.dll
echo   Mt3\bin\Release\netstandard2.1\Mt3.dll
echo   Madmom\bin\Release\netstandard2.1\Madmom.dll
echo   Mp3ToSheet\bin\Release\net8.0-windows\Mp3ToSheet.exe        （WPF 桌面应用）
echo   Mp3ToSheet.Bench\bin\Release\net8.0\Mp3ToSheet.Bench.exe   （命令行测速）
echo.
echo 运行桌面应用： Mp3ToSheet\bin\Release\net8.0-windows\Mp3ToSheet.exe
echo 命令行测速：   dotnet run --project Mp3ToSheet.Bench -c Release -- 音频文件
echo.

popd
endlocal
exit /b 0

:fail
echo.
echo [失败] 安装中断。
popd
endlocal
exit /b 1