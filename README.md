# ChurnGuard: Customer Churn Prediction and Retention Support System

ChurnGuard scores every telecom customer with a calibrated churn probability and explains the three biggest drivers in plain English, so a retention team knows whom to call first and why.

Pipeline: raw CSV, feature engineering, tuned and calibrated XGBoost, SHAP explanations, FastAPI backend with API-key auth and structured logging, Streamlit frontend, Docker deployment.

## Quick start (Google Colab, easiest)

1. Open `ChurnGuard_Colab.ipynb` in Google Colab.
2. Run the cells from top to bottom. The first cell asks you to upload `ChurnGuard.zip`.
3. The notebook installs packages, trains the model, shows every metric and plot, starts the API and sends a test prediction.

Training takes a few minutes on a free CPU runtime.

## Quick start (your own computer)

Requirements: Python 3.10 or newer.

Windows: double-click `run_all.bat`.
Mac or Linux: `bash run_all.sh`

This creates a virtual environment, installs packages, trains the model on first run, then starts the API on http://localhost:8000 and the app on http://localhost:8501.

Manual steps:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m churnguard.train
uvicorn app.main:app --port 8000
streamlit run frontend/streamlit_app.py
```

On Windows activate with `.venv\Scripts\activate`.

## Project structure

```
ChurnGuard/
  churnguard/        data, features, modeling, evaluation, explain, registry, train, retrain
  app/               FastAPI backend (POST /predict, GET /health, auth, logging)
  frontend/          Streamlit interface
  tests/             pytest suite
  data/              Telco dataset, data/new for retraining files
  models/            versioned model files (joblib and skops), latest.json
  reports/           plots, metrics.json, comparison tables
  logs/              predictions.jsonl written by the API
  ChurnGuard_Colab.ipynb
  Dockerfile
  MODEL_CARD.md  ARCHITECTURE.md  EVALUATION_REPORT.md  MONITORING_NOTES.md
```

## Using the API

Start it: `uvicorn app.main:app --port 8000`. Interactive documentation is generated at http://localhost:8000/docs.

The API key is read from the environment variable `CHURNGUARD_API_KEY` (default for local use: `demo-key`). Always set your own value when deploying.

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```

Response: probability, label, risk level, threshold, the top three SHAP drivers and a plain-English summary.

Endpoints:

- `POST /predict`: one customer, with explanation (API key required)
- `POST /predict/batch`: up to 500 customers, probabilities only (API key required)
- `GET /health`: status, model version, threshold

Every prediction is appended to `logs/predictions.jsonl` and every request is logged as one JSON line on the console.

## Tests

```
python -m pytest
```

Covers a valid prediction, missing and invalid input, an unseen category, an authentication failure, a regression test against a stored input and output pair, plus feature engineering and model file checks.

## Docker and deployment

```
docker build -t churnguard .
docker run -p 7860:7860 -e CHURNGUARD_API_KEY=your-secret churnguard
```

The image trains the model during the build, so no model file has to be committed. Port 7860 matches Hugging Face Spaces.

Hugging Face Spaces: create a Space with the Docker SDK, upload the repository files, and add `CHURNGUARD_API_KEY` under Settings, Secrets. Render: create a Web Service from the repository with the Docker runtime and add `CHURNGUARD_API_KEY` as an environment variable. Then set the Streamlit sidebar API URL (or `CHURNGUARD_API_URL`) to your public URL.

## Retraining

Put new CSV files in the original schema (including `Churn`) into `data/new/`, then run:

```
python -m churnguard.retrain
```

It combines the base data with the new files, removes duplicate customer IDs, runs the same pipeline and saves a new versioned model.

## Extra: LightGBM comparison

After training, `python -m churnguard.boosting_comparison` compares XGBoost with LightGBM on identical settings and writes `reports/boosting_comparison.csv`.

## Reproducibility

Seed 42 everywhere, stratified splits, package list in `requirements.txt`. Experiments are tracked locally with MLflow in `mlruns/` (run `mlflow ui` to browse them).

## Documentation

- `MODEL_CARD.md`: problem, data, metrics, limitations, intended and out-of-scope use
- `ARCHITECTURE.md`: how the pieces fit together
- `EVALUATION_REPORT.md`: full evaluation results, regenerated on every training run
- `MONITORING_NOTES.md`: drift checks and retraining trigger
