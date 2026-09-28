@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  MusicRecorder 一键构建脚本
rem  用法: build.bat [Debug/Release]     默认 Release
rem ============================================================

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Release"

set "ROOT=%~dp0"
pushd "%ROOT%"

echo ================================================
echo  MusicRecorder 构建  [%CONFIG%]
echo ================================================

set "FAILED=0"
for %%P in (BasicPitch Mt3 BasicPitchSharp.Test Mp3ToSheet) do (
    echo.
    echo --- %%P ---
    dotnet build "%%P\%%P.csproj" -c %CONFIG% --nologo
    if errorlevel 1 set "FAILED=1"
)

echo.
if "%FAILED%"=="1" (
    echo [失败] 部分工程构建出错，请查看上方输出。
    popd
    exit /b 1
)

echo [完成] 构建产物：
echo   BasicPitch\bin\%CONFIG%\netstandard2.1\BasicPitch.dll
echo   Mt3\bin\%CONFIG%\netstandard2.1\Mt3.dll
echo   BasicPitchSharp.Test\bin\%CONFIG%\net8.0\BasicPitchSharp.Test.exe
echo   Mp3ToSheet\bin\%CONFIG%\net8.0-windows\Mp3ToSheet.exe

popd
endlocal
exit /b 0