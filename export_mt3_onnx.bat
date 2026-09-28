@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  将 MR-MT3 的 PyTorch checkpoint 导出为 ONNX（带 KV-cache）
rem
rem  前置条件：
rem    1. mr_mt3_model\mt3.pth 已就位（下载地址见 README）
rem    2. 已安装 Python 3.10+（首次运行会自动建 .onnx_conv_venv 并装依赖）
rem
rem  产物：mt3_onnx\mt3_encoder.onnx / mt3_decoder.onnx（含 .onnx.data）
rem ============================================================

set "ROOT=%~dp0"
set "VENV=%ROOT%.onnx_conv_venv"
set "PY=%VENV%\Scripts\python.exe"

pushd "%ROOT%"

if not exist "%PY%" (
    echo [1/2] 未找到虚拟环境，正在创建 .onnx_conv_venv ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请确认已安装 Python 3.10+ 并加入 PATH。
        popd
        exit /b 1
    )
    echo       安装依赖（torch / transformers / onnx / onnxruntime / mt3-infer）...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install torch "transformers==4.44.0" onnx onnxruntime mt3-infer
    if errorlevel 1 (
        echo [错误] 依赖安装失败。
        popd
        exit /b 1
    )
)

if not exist "%ROOT%mr_mt3_model\mt3.pth" (
    echo [错误] 未找到 mr_mt3_model\mt3.pth
    echo        请先按 README「MT3 模型下载」准备好 checkpoint。
    popd
    exit /b 1
)

echo [2/2] 导出 ONNX ...
"%PY%" "%ROOT%convert_mt3_onnx.py"
set "RC=%ERRORLEVEL%"

popd
if not "%RC%"=="0" (
    echo [失败] 导出出错，返回码 %RC%
    exit /b %RC%
)

echo.
echo [完成] 产物已写入 mt3_onnx\
endlocal
exit /b 0