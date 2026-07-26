@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
where npm >nul 2>nul
if %errorlevel%==0 (
  cd frontend
  npm install
  cd ..
) else (
  echo npm was not found. Install Node.js 18+ to run the React frontend.
)
echo Install complete. Run start_windows.bat

