@echo off
echo ==========================================
echo    UT - Dependency Installer
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or later.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Python version:
python --version
echo.

REM Check virtual environment
if exist "venv\Scripts\python.exe" (
    echo [2/3] Virtual environment found, activating...
    call venv\Scripts\activate.bat
) else (
    echo [2/3] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)
echo.

echo [3/3] Installing dependencies (using Tsinghua mirror)...
echo Mirror: https://pypi.tuna.tsinghua.edu.cn/simple
echo.

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed. Check network or requirements.txt.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo    Dependencies installed successfully!
echo ==========================================
echo.
echo You can now run the program:
echo   python main.py
echo.
pause