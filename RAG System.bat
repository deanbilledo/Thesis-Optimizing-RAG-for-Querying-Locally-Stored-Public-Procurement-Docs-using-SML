@echo off
chcp 65001 >nul
title RAG Document Analysis System

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║     RAG Document Analysis System - Starting...           ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Set paths relative to this script
set "ROOT_DIR=%~dp0"
set "PYTHON_DIR=%ROOT_DIR%python"
set "APP_DIR=%ROOT_DIR%app"
set "TESSERACT_DIR=%ROOT_DIR%tesseract"

:: Add Python and Tesseract to PATH
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%TESSERACT_DIR%;%PATH%"

:: Set Python environment variables
set "PYTHONHOME=%PYTHON_DIR%"
set "PYTHONPATH=%APP_DIR%;%PYTHON_DIR%\Lib\site-packages"

:: Set Tesseract path for pytesseract
set "TESSERACT_CMD=%TESSERACT_DIR%\tesseract.exe"

:: Change to app directory
cd /d "%APP_DIR%"

echo  Starting application...
echo  (This may take 1-2 minutes on first launch)
echo.

:: Run the desktop application
"%PYTHON_DIR%\python.exe" desktop_app.py

:: If desktop_app fails, fall back to streamlit
if errorlevel 1 (
    echo.
    echo  Desktop mode failed, trying web browser mode...
    "%PYTHON_DIR%\python.exe" -m streamlit run app.py --server.headless true
)

pause
