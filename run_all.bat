@echo off
cd /d "%~dp0"

if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if not exist models\latest.json (
  python -m churnguard.train
)

if "%CHURNGUARD_API_KEY%"=="" set CHURNGUARD_API_KEY=demo-key
if "%CHURNGUARD_API_URL%"=="" set CHURNGUARD_API_URL=http://localhost:8000

start "ChurnGuard API" cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000"
timeout /t 8 /nobreak > nul
streamlit run frontend\streamlit_app.py --server.port 8501
