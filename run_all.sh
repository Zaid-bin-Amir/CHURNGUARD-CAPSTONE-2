#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f "models/latest.json" ]; then
  python -m churnguard.train
fi

export CHURNGUARD_API_KEY="${CHURNGUARD_API_KEY:-demo-key}"
export CHURNGUARD_API_URL="${CHURNGUARD_API_URL:-http://localhost:8000}"

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

sleep 5
streamlit run frontend/streamlit_app.py --server.port 8501
