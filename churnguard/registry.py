import json
import logging
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from churnguard import config

logger = logging.getLogger(__name__)


def default_tag(description="xgb-calibrated"):
    return f"{datetime.now().strftime('%Y%m%d')}_{description}"


def library_versions():
    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    for name in ("xgboost", "shap", "skops", "mlflow", "lightgbm"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def save_skops(pipeline, path):
    try:
        import skops.io as sio

        sio.dump(pipeline, path)
        restored = load_skops(path)
        frame = pd.DataFrame([config.SAMPLE_CUSTOMER])[config.RAW_FEATURES]
        np.testing.assert_allclose(
            pipeline.predict_proba(frame), restored.predict_proba(frame), atol=1e-6
        )
        return True
    except Exception as exc:
        logger.warning("skops serialization skipped: %s", exc)
        try:
            path.unlink()
        except OSError:
            pass
        return False


def load_skops(path):
    import skops.io as sio

    return sio.load(path, trusted=sio.get_untrusted_types(file=str(path)))


def save_all(pipeline, tag, threshold, summary=None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib_name = f"churn_pipeline_{tag}.joblib"
    skops_name = f"churn_pipeline_{tag}.skops"
    joblib.dump(pipeline, config.MODELS_DIR / joblib_name)
    skops_saved = save_skops(pipeline, config.MODELS_DIR / skops_name)
    meta = {
        "tag": tag,
        "joblib_file": joblib_name,
        "skops_file": skops_name if skops_saved else None,
        "threshold": float(threshold),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "versions": library_versions(),
        "summary": summary or {},
    }
    text = json.dumps(meta, indent=2, default=str)
    (config.MODELS_DIR / f"churn_pipeline_{tag}.json").write_text(text, encoding="utf-8")
    config.LATEST_FILE.write_text(text, encoding="utf-8")
    return meta


def save_regression_case(pipeline):
    record = dict(config.SAMPLE_CUSTOMER)
    probability = float(pipeline.predict_proba(pd.DataFrame([record])[config.RAW_FEATURES])[0, 1])
    payload = {"input": record, "expected_probability": round(probability, 6)}
    path = config.MODELS_DIR / "regression_case.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_latest():
    if not config.LATEST_FILE.exists():
        raise FileNotFoundError("No trained model found. Run: python -m churnguard.train")
    meta = json.loads(config.LATEST_FILE.read_text(encoding="utf-8"))
    pipeline = joblib.load(config.MODELS_DIR / meta["joblib_file"])
    pipeline.predict_proba(pd.DataFrame([config.SAMPLE_CUSTOMER])[config.RAW_FEATURES])
    return pipeline, meta


def load_or_train():
    try:
        return load_latest()
    except Exception as exc:
        logger.warning("Saved model not usable (%s). Training a fresh model now.", exc)
        from churnguard import train

        train.run(track=False)
        return load_latest()
