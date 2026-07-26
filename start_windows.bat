@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Missing .venv. Run install_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
if "%GAUSSCAPTURE_HOST%"=="" set GAUSSCAPTURE_HOST=127.0.0.1
if "%GAUSSCAPTURE_PORT%"=="" set GAUSSCAPTURE_PORT=7860
start "GaussCapture Backend" cmd /k python -m uvicorn backend.main:app --host %GAUSSCAPTURE_HOST% --port %GAUSSCAPTURE_PORT%
if exist frontend\node_modules (
  cd frontend
  npm run dev
) else (
  echo Backend started at http://localhost:7860
  echo Install frontend dependencies with: cd frontend && npm install
)
