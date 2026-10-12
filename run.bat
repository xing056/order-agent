@echo off
REM ============================================================
REM  agent-lab one-click launcher  (Windows)
REM  Double-click this file to run the demo.
REM  It always uses the project-local .venv, so you can never
REM  accidentally launch with the wrong Python interpreter again.
REM ============================================================
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [*] No .venv found - creating one...
    python -m venv .venv
    if errorlevel 1 (
        echo [!] Failed to create venv. Is Python on your PATH?
        pause
        exit /b 1
    )
)

"%PY%" -c "import streamlit, langchain, openai, dotenv" 2>nul
if errorlevel 1 (
    echo [*] Dependencies missing - installing from requirements.txt ...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [!] Install failed. Check your network / pip mirror.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    echo [!] .env not found. Copy .env.example to .env and fill in your API Key.
    pause
    exit /b 1
)

echo.
echo [*] Resetting demo data (no API calls, costs nothing)...
"%PY%" -m agent.db

echo.
echo [*] Starting the demo UI - your browser should open shortly.
echo     Press Ctrl+C in this window to stop.
echo.
"%PY%" -m streamlit run app.py

pause
