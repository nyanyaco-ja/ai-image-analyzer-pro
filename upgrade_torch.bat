@echo off
chcp 65001 > nul
title Upgrade PyTorch to v2.6+

echo ==========================================
echo    Upgrading PyTorch to v2.6.0+
echo ==========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [ERROR] Virtual environment not found
    echo Please run setup.bat first
    pause
    exit
)

echo.
echo ==========================================
echo Upgrading torch and torchvision...
echo ==========================================
echo.

REM Uninstall old versions first
echo Uninstalling old PyTorch versions...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch 2.6.0+ with CUDA 12.1 support...
echo.

REM Install PyTorch with CUDA support
pip install torch>=2.6.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu121

echo.
echo ==========================================
echo Verifying installation...
echo ==========================================
echo.

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ==========================================
echo Upgrade completed!
echo ==========================================
echo.
echo Next step: Run test_clip.py to verify CLIP functionality
echo   python test_clip.py
echo.
pause
