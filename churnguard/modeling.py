import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard import config
from churnguard.features import FeatureEngineer

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

PARAM_SPACE = {
    "clf__n_estimators": randint(150, 600),
    "clf__max_depth": randint(2, 7),
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0.0, 2.0),
    "clf__reg_lambda": loguniform(0.5, 20.0),
    "clf__scale_pos_weight": uniform(1.0, 2.5),
}

SEARCH_SPACE_DESCRIPTION = {
    "n_estimators": "integer 150 to 599",
    "max_depth": "integer 2 to 6",
    "learning_rate": "log-uniform 0.01 to 0.2",
    "subsample": "uniform 0.6 to 1.0",
    "colsample_bytree": "uniform 0.5 to 1.0",
    "min_child_weight": "integer 1 to 9",
    "gamma": "uniform 0.0 to 2.0",
    "reg_lambda": "log-uniform 0.5 to 20",
    "scale_pos_weight": "uniform 1.0 to 3.5 (class imbalance weight)",
}


def make_cv():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def make_preprocessor(scale=False, engineered=True):
    numeric = config.NUMERIC_FEATURES if engineered else config.RAW_NUMERIC
    categorical = config.CATEGORICAL_FEATURES if engineered else config.RAW_CATEGORICAL
    numeric_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator, scale=False, engineered=True):
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("prep", make_preprocessor(scale=scale, engineered=engineered)),
            ("clf", estimator),
        ]
    )


def make_xgb(**params):
    from xgboost import XGBClassifier

    settings = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": config.SEED,
        "verbosity": 0,
    }
    settings.update(params)
    return XGBClassifier(**settings)


def candidate_specs(xgb_factory=make_xgb):
    return [
        {
            "name": "Majority baseline",
            "estimator": DummyClassifier(strategy="most_frequent"),
            "scale": False,
            "engineered": True,
        },
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=config.SEED),
            "scale": True,
            "engineered": True,
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=config.SEED,
            ),
            "scale": False,
            "engineered": True,
        },
        {
            "name": "XGBoost (default settings)",
            "estimator": xgb_factory(scale_pos_weight=2.0),
            "scale": False,
            "engineered": True,
        },
        {
            "name": "XGBoost (raw features only)",
            "estimator": xgb_factory(scale_pos_weight=2.0),
            "scale": False,
            "engineered": False,
        },
    ]


def best_threshold(y, probabilities):
    thresholds = np.linspace(0.05, 0.95, 91)
    actual = np.asarray(y)
    scores = np.array(
        [
            fbeta_score(actual, (probabilities >= value).astype(int), beta=config.BETA, zero_division=0)
            for value in thresholds
        ]
    )
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def evaluate_cv(name, pipeline, X, y):
    cv = make_cv()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    probabilities = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold, f2 = best_threshold(y, probabilities)
    row = {"model": name}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[metric] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    row["f2_at_best_threshold"] = f2
    row["best_threshold"] = threshold
    return row


def compare_models(X, y, xgb_factory=make_xgb):
    rows = []
    for spec in candidate_specs(xgb_factory):
        pipeline = make_pipeline(spec["estimator"], scale=spec["scale"], engineered=spec["engineered"])
        rows.append(evaluate_cv(spec["name"], pipeline, X, y))
    return pd.DataFrame(rows)


def tune_xgb(X, y, xgb_factory=make_xgb, n_iter=None, param_space=None):
    pipeline = make_pipeline(xgb_factory(), scale=False, engineered=True)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space or PARAM_SPACE,
        n_iter=n_iter or config.SEARCH_ITER,
        scoring="average_precision",
        cv=make_cv(),
        n_jobs=-1,
        random_state=config.SEED,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    best = {key.replace("clf__", "", 1): value for key, value in search.best_params_.items()}
    results = pd.DataFrame(search.cv_results_)
    keep = ["mean_test_score", "std_test_score", "rank_test_score"] + [c for c in results.columns if c.startswith("param_")]
    top = results.sort_values("rank_test_score")[keep].head(10).reset_index(drop=True)
    top.columns = [c.replace("param_clf__", "") for c in top.columns]
    return best, float(search.best_score_), top


def build_final(best_params, xgb_factory=make_xgb):
    calibrated = CalibratedClassifierCV(
        xgb_factory(**best_params),
        method="sigmoid",
        cv=make_cv(),
        ensemble=False,
    )
    return make_pipeline(calibrated, scale=False, engineered=True)


def select_threshold(pipeline, X, y):
    probabilities = cross_val_predict(pipeline, X, y, cv=make_cv(), method="predict_proba", n_jobs=-1)[:, 1]
    threshold, f2 = best_threshold(y, probabilities)
    return threshold, f2, probabilities


def threshold_table(y, probabilities, values=None):
    actual = np.asarray(y)
    rows = []
    for value in values or [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        predicted = (probabilities >= value).astype(int)
        rows.append(
            {
                "threshold": float(value),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
                "f2": float(fbeta_score(actual, predicted, beta=config.BETA, zero_division=0)),
                "flagged_share": float(predicted.mean()),
            }
        )
    return rows


def fit_final(X_train, y_train, best_params, xgb_factory=make_xgb):
    pipeline = build_final(best_params, xgb_factory)
    threshold, f2, oof = select_threshold(pipeline, X_train, y_train)
    pipeline.fit(X_train, y_train)
    return pipeline, threshold, f2, oof
