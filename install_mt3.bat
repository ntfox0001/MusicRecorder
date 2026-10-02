@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  MT3 模型安装脚本（可选组件）
rem
rem  用途：为「MT3 多乐器转谱」准备 ONNX 模型文件。
rem  步骤：下载 MR-MT3 checkpoint → 导出 ONNX → 写入 mt3_onnx\
rem
rem  前置条件：已安装 Python 3.10+ 并加入 PATH。
rem            首次运行会自动创建 .onnx_conv_venv 虚拟环境并安装
rem            torch / transformers / onnx / onnxruntime / mt3-infer。
rem  用法：install_mt3.bat
rem
rem  产物：mt3_onnx\mt3_encoder.onnx / mt3_decoder.onnx（含 .onnx.data，约 350 MB）
rem  注意：模型未随仓库分发，本脚本会联网下载（HuggingFace）。
rem ============================================================

set "ROOT=%~dp0"
set "VENV=%ROOT%.onnx_conv_venv"
set "PY=%VENV%\Scripts\python.exe"
set "CKPT=%ROOT%mr_mt3_model\mt3.pth"
set "REPO=gudgud1014/MR-MT3"

pushd "%ROOT%"

echo ================================================
echo  MT3 模型安装
echo ================================================
echo.

rem ---------- 1/3 准备 Python 环境 ----------
echo [1/3] 检查 Python ...
where python >nul 2>nul
if errorlevel 1 (
    echo       [错误] 未找到 python 命令，请安装 Python 3.10+ 并加入 PATH。
    goto :fail
)

if exist "%PY%" goto :venv_ready
echo       创建虚拟环境 .onnx_conv_venv 并安装依赖，耗时较长 ...
python -m venv "%VENV%"
if errorlevel 1 (
    echo       [错误] 创建虚拟环境失败，请确认已安装 Python 3.10+。
    goto :fail
)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install torch "transformers==4.44.0" onnx onnxruntime mt3-infer huggingface_hub
if errorlevel 1 (
    echo       [错误] 依赖安装失败。
    goto :fail
)
goto :venv_ready

:venv_ready

rem ---------- 2/3 下载 checkpoint ----------
echo.
if exist "%CKPT%" echo [2/3] 已存在 mr_mt3_model\mt3.pth，跳过下载。
if exist "%CKPT%" goto :ckpt_ready
echo [2/3] 下载 MR-MT3 checkpoint（HuggingFace: %REPO%）...
"%PY%" -m pip install --upgrade huggingface_hub
if errorlevel 1 (
    echo       [错误] huggingface_hub 安装失败。
    goto :fail
)
"%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='gudgud1014/MR-MT3', local_dir='mr_mt3_model')"
if errorlevel 1 (
    echo       [错误] 下载失败，请检查网络或代理设置。
    goto :fail
)

:ckpt_ready
if exist "%CKPT%" goto :export
echo       [错误] 未找到 mr_mt3_model\mt3.pth，下载可能不完整。
echo              也可手动下载后放到该路径，再重新运行本脚本。
goto :fail

rem ---------- 3/3 导出 ONNX ----------
:export
echo.
echo [3/3] 导出 ONNX（调用 export_mt3_onnx.bat）...
call "%ROOT%export_mt3_onnx.bat"
if errorlevel 1 (
    echo       [错误] ONNX 导出失败。
    goto :fail
)

echo.
echo [完成] mt3_onnx\ 已就绪，MT3 多乐器转谱可用。
echo       试用：dotnet run --project Mp3ToSheet.Bench -c Release -- 音频文件
echo.

popd
endlocal
exit /b 0

:fail
echo.
echo [失败] MT3 安装中断。
popd
endlocal
exit /b 1