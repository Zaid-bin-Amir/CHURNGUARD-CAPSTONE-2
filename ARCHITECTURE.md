# ChurnGuard Architecture

```
Telco CSV
   |
   v
churnguard.data      clean types, blank TotalCharges to 0, Churn Yes/No to 1/0, stratified 80/20 split
   |
   v
sklearn Pipeline
   FeatureEngineer   tenure_bucket, total_services, charges_per_tenure, contract_risk, auto_pay, security_support
   ColumnTransformer one-hot (unseen categories ignored), numeric passthrough or scaling
   CalibratedClassifierCV(XGBoost)   sigmoid calibration on out-of-fold predictions
   |
   v
churnguard.train     comparison, tuning, threshold, hold-out evaluation, SHAP, reports, saving
   |
   v
models/              churn_pipeline_<date>_<experiment>.joblib / .skops / .json, latest.json
   |
   v
app (FastAPI)  <---- HTTP ---->  frontend (Streamlit)
   |
   v
logs/predictions.jsonl
```

## Components

**churnguard/config.py** holds paths, the seed, feature lists and the sample customer. Every other module reads from it.

**churnguard/features.py** contains `FeatureEngineer`, a scikit-learn transformer that works row by row and keeps no state, so it cannot leak information from other rows or from the target. It is the first step of the saved pipeline, so training and serving use identical code.

**churnguard/modeling.py** builds the candidate pipelines (majority baseline, Logistic Regression, Random Forest, XGBoost), evaluates them with the same stratified 5-fold cross-validation, tunes XGBoost with `RandomizedSearchCV` scored by PR-AUC, calibrates probabilities and picks the decision threshold by maximising F2 on out-of-fold training predictions.

**churnguard/evaluation.py** computes hold-out metrics, the calibration table, the error analysis by segment, the leakage check and the plots.

**churnguard/explain.py** uses exact TreeSHAP values from the XGBoost model inside the calibrated pipeline. One-hot columns are summed back to their source feature, and each driver is turned into one plain-English sentence.

**churnguard/registry.py** saves the full pipeline with joblib and skops under a name containing the date and a short experiment description, writes `latest.json` with the threshold and library versions, and loads the model for the API. If no saved model exists, the API trains one automatically at startup.

**churnguard/tracking.py** logs parameters, metrics and artifacts to a local MLflow store in `mlruns/`.

**app/** is the FastAPI backend. `main.py` defines the routes and request logging middleware, `schemas.py` validates input with Pydantic, `security.py` checks the `X-API-Key` header with a constant-time comparison, `service.py` wraps prediction and explanation, `logging_utils.py` writes JSON logs and the prediction history.

**frontend/streamlit_app.py** is a thin client. It never imports the model and calls the backend only over HTTP. It offers single-customer scoring with explanation, a what-if panel and batch CSV scoring.

## Design decisions

- The saved object is one pipeline (features, encoding, model, calibration), so the API sends raw customer fields and cannot drift from training.
- The threshold is stored next to the model in `latest.json`, so changing the business trade-off does not need new code.
- Hold-out data is touched once, after every choice has been made.
- Pinned package versions are recorded with each model in its JSON file for reproducibility checks.
