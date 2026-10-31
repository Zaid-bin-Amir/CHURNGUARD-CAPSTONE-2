from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NEW_DATA_DIR = ROOT / "data" / "new"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLRUNS_DIR = ROOT / "mlruns"
LATEST_FILE = MODELS_DIR / "latest.json"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SEARCH_ITER = 30
BETA = 2.0

TARGET = "Churn"
ID_COLUMN = "customerID"

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RAW_CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERIC

ENGINEERED_CATEGORICAL = ["tenure_bucket"]
ENGINEERED_NUMERIC = [
    "total_services",
    "charges_per_tenure",
    "contract_risk",
    "auto_pay",
    "security_support",
]

CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}
