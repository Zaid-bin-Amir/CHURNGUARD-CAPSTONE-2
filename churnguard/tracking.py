import logging

from churnguard import config

logger = logging.getLogger(__name__)


def flatten(prefix, values):
    flat = {}
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def log_run(tag, params, metrics, artifact_paths):
    try:
        import mlflow

        config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(config.MLRUNS_DIR.as_uri())
        mlflow.set_experiment("churnguard")
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.log_metrics(metrics)
            for path in artifact_paths:
                if path is not None and path.exists():
                    mlflow.log_artifact(str(path))
        return True
    except Exception as exc:
        logger.warning("MLflow tracking skipped: %s", exc)
        return False
