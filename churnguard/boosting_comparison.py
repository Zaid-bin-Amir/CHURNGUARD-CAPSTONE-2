import json
import logging
import sys

import pandas as pd

from churnguard import config, data, modeling


def lightgbm_factory(params):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 4)),
        num_leaves=2 ** int(params.get("max_depth", 4)) - 1,
        subsample=float(params.get("subsample", 0.8)),
        subsample_freq=1,
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        min_child_weight=float(params.get("min_child_weight", 1)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        scale_pos_weight=float(params.get("scale_pos_weight", 2.0)),
        random_state=config.SEED,
        n_jobs=1,
        verbose=-1,
    )


def run():
    meta = json.loads(config.LATEST_FILE.read_text(encoding="utf-8"))
    params = meta["summary"]["best_params"]
    X, y = data.features_and_target(data.load_clean())
    X_train, X_test, y_train, y_test = data.split_dataset(X, y)
    rows = []
    xgb_pipeline = modeling.make_pipeline(modeling.make_xgb(**params), scale=False)
    rows.append(modeling.evaluate_cv("XGBoost (tuned)", xgb_pipeline, X_train, y_train))
    lgbm_pipeline = modeling.make_pipeline(lightgbm_factory(params), scale=False)
    rows.append(modeling.evaluate_cv("LightGBM (same settings)", lgbm_pipeline, X_train, y_train))
    table = pd.DataFrame(rows)
    path = config.REPORTS_DIR / "boosting_comparison.csv"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    table = run()
    columns = ["model", "accuracy", "recall", "roc_auc", "pr_auc", "f2_at_best_threshold"]
    print(table[columns].round(4).to_string(index=False))
    print(f"Saved to {config.REPORTS_DIR / 'boosting_comparison.csv'}")


if __name__ == "__main__":
    main()
