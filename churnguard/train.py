import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import random
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from churnguard import config, data, evaluation, explain, modeling, registry, reporting, tracking

logger = logging.getLogger("churnguard.train")


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def write_json(name, payload):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=to_jsonable), encoding="utf-8")
    return path


def run(raw_df=None, tag=None, search_iter=None, track=True, best_params=None, xgb_factory=None, param_space=None):
    warnings.filterwarnings("ignore")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    factory = xgb_factory or modeling.make_xgb
    started = time.time()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = data.clean(raw_df) if raw_df is not None else data.load_clean()
    X, y = data.features_and_target(frame)
    X_train, X_test, y_train, y_test = data.split_dataset(X, y)
    summary = data.summarize(X, y, X_train, X_test, y_train, y_test)
    logger.info("Data ready: %s rows, churn rate %.3f", summary["rows"], summary["churn_rate"])

    leakage = evaluation.leakage_check(X_train, y_train)

    logger.info("Comparing candidate models with stratified %s-fold cross-validation", config.CV_FOLDS)
    comparison = modeling.compare_models(X_train, y_train, factory)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    improvements = reporting.improvement_statements(comparison.to_dict("records"))

    if best_params is None:
        logger.info("Tuning XGBoost with RandomizedSearchCV")
        best_params, best_cv_score, top = modeling.tune_xgb(
            X_train, y_train, factory, n_iter=search_iter, param_space=param_space
        )
        top.to_csv(config.REPORTS_DIR / "tuning_top_configs.csv", index=False)
        n_iter = search_iter or config.SEARCH_ITER
    else:
        best_cv_score = float("nan")
        n_iter = 0

    logger.info("Fitting calibrated final model and choosing the decision threshold")
    pipeline, threshold, oof_f2, oof_probabilities = modeling.fit_final(X_train, y_train, best_params, factory)
    threshold_rows = modeling.threshold_table(y_train, oof_probabilities)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    holdout = evaluation.holdout_metrics(y_test, probabilities, threshold)
    baseline = evaluation.baseline_holdout_metrics(y_train, y_test)
    calibration, calibration_summary = evaluation.calibration_table(y_test, probabilities)
    calibration.to_csv(config.REPORTS_DIR / "calibration_table.csv", index=False)
    errors = evaluation.error_analysis(X_test, y_test, probabilities, threshold)

    logger.info("Computing SHAP explanations")
    sample = X_train.sample(n=min(1000, len(X_train)), random_state=config.SEED)
    importance = explain.global_importance(pipeline, sample)
    importance.to_csv(config.REPORTS_DIR / "global_importance.csv", index=False)
    examples = explain.local_examples(pipeline, X_test.reset_index(drop=True), y_test.reset_index(drop=True), probabilities, threshold)

    figures = [
        evaluation.save_figure(evaluation.plot_roc_pr(y_test, probabilities), "roc_pr_curves.png"),
        evaluation.save_figure(evaluation.plot_confusion(holdout), "confusion_matrix.png"),
        evaluation.save_figure(evaluation.plot_calibration(calibration), "calibration.png"),
        evaluation.save_figure(explain.summary_figure(pipeline, sample, importance), "shap_summary.png"),
    ]

    resolved_tag = tag or registry.default_tag()
    results = {
        "tag": resolved_tag,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data": summary,
        "leakage": leakage,
        "comparison": comparison.to_dict("records"),
        "improvements": improvements,
        "tuning": {
            "best_params": best_params,
            "best_cv_pr_auc": best_cv_score,
            "n_iter": n_iter,
            "search_space": modeling.SEARCH_SPACE_DESCRIPTION,
        },
        "threshold": {"value": threshold, "oof_f2": oof_f2, "table": threshold_rows},
        "holdout": holdout,
        "baseline_holdout": baseline,
        "calibration": {"table": calibration.to_dict("records"), "summary": calibration_summary},
        "error_analysis": errors,
        "importance": importance.to_dict("records"),
        "local_examples": examples,
        "versions": registry.library_versions(),
    }

    meta = registry.save_all(
        pipeline,
        resolved_tag,
        threshold,
        summary={
            "best_params": best_params,
            "holdout_pr_auc": holdout["pr_auc"],
            "holdout_roc_auc": holdout["roc_auc"],
            "holdout_recall": holdout["recall"],
            "rows": summary["rows"],
        },
    )
    registry.save_regression_case(pipeline)
    metrics_path = write_json("metrics.json", results)
    write_json("error_analysis.json", errors)
    write_json("local_explanations.json", examples)
    reporting.write_docs(results)

    if track:
        params = {"tag": resolved_tag, "threshold": threshold, "seed": config.SEED, "cv_folds": config.CV_FOLDS}
        params.update({f"xgb_{key}": value for key, value in best_params.items()})
        metrics = tracking.flatten("holdout_", {k: v for k, v in holdout.items() if k != "classification_report"})
        metrics["oof_f2"] = oof_f2
        tracking.log_run(
            resolved_tag,
            params,
            metrics,
            [metrics_path, config.MODELS_DIR / meta["joblib_file"], *figures],
        )

    logger.info(
        "Finished in %.0f s. Hold-out recall %.3f, PR-AUC %.3f, ROC-AUC %.3f at threshold %.2f",
        time.time() - started,
        holdout["recall"],
        holdout["pr_auc"],
        holdout["roc_auc"],
        threshold,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train the ChurnGuard model and generate reports.")
    parser.add_argument("--search-iter", type=int, default=None, help="Number of RandomizedSearchCV samples")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow tracking")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    results = run(search_iter=arguments.search_iter, track=not arguments.no_mlflow)
    print(f"Model version: {results['tag']}")
    print(f"Hold-out recall {results['holdout']['recall']:.3f} | PR-AUC {results['holdout']['pr_auc']:.3f} | ROC-AUC {results['holdout']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
