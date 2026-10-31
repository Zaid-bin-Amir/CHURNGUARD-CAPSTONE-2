from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard import config


def load_raw(path=None):
    return pd.read_csv(Path(path) if path else config.DATA_PATH)


def clean(raw):
    frame = raw.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    frame["tenure"] = frame["tenure"].astype(int)
    frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(float)
    frame[config.TARGET] = frame[config.TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return frame


def load_clean(path=None):
    return clean(load_raw(path))


def features_and_target(frame):
    return frame[config.RAW_FEATURES].copy(), frame[config.TARGET].copy()


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.SEED,
    )


def summarize(X, y, X_train, X_test, y_train, y_test):
    return {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
    }
