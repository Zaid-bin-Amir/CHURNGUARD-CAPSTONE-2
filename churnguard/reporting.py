import pandas as pd

from churnguard import config
from churnguard.features import FEATURE_JUSTIFICATION

PENDING = "Not generated yet. Run `python -m churnguard.train` (or the Colab notebook) and this section is filled in automatically."


def num(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value, digits=1):
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def md_table(rows, columns):
    header = "| " + " | ".join(title for _, title, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _, formatter in columns:
            value = row.get(key)
            cells.append(formatter(value) if formatter else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def improvement_statements(comparison):
    table = pd.DataFrame(comparison).set_index("model")
    statements = []

    def verdict(delta, reference_std):
        return "a meaningful improvement" if delta > max(0.01, reference_std) else "not a meaningful improvement"

    def compare(name, reference):
        row = table.loc[name]
        base = table.loc[reference]
        delta_pr = row["pr_auc"] - base["pr_auc"]
        delta_roc = row["roc_auc"] - base["roc_auc"]
        delta_recall = row["recall"] - base["recall"]
        delta_f2 = row["f2_at_best_threshold"] - base["f2_at_best_threshold"]
        return (
            f"{name} versus {reference}: PR-AUC {delta_pr:+.3f}, ROC-AUC {delta_roc:+.3f}, recall {delta_recall:+.3f}, "
            f"F2 at best threshold {delta_f2:+.3f} (fold standard deviation of the reference PR-AUC: {base['pr_auc_std']:.3f}). "
            f"This is {verdict(delta_pr, base['pr_auc_std'])}."
        )

    for name in ["Logistic Regression", "Random Forest", "XGBoost (default settings)"]:
        if name in table.index:
            statements.append(compare(name, "Majority baseline"))
    for name in ["Random Forest", "XGBoost (default settings)"]:
        if name in table.index and "Logistic Regression" in table.index:
            statements.append(compare(name, "Logistic Regression"))
    if "XGBoost (default settings)" in table.index and "XGBoost (raw features only)" in table.index:
        statements.append(
            compare("XGBoost (default settings)", "XGBoost (raw features only)").replace(
                "XGBoost (default settings) versus XGBoost (raw features only)",
                "Engineered features (XGBoost with them versus without them)",
            )
        )
    return statements


def selection_text(comparison):
    table = pd.DataFrame(comparison).set_index("model")
    candidates = table.drop(index=[name for name in ["Majority baseline", "XGBoost (raw features only)"] if name in table.index])
    by_f2 = candidates.sort_values("f2_at_best_threshold", ascending=False)
    by_pr = candidates.sort_values("pr_auc", ascending=False)
    order_f2 = ", ".join(f"{name} ({row['f2_at_best_threshold']:.3f})" for name, row in by_f2.iterrows())
    order_pr = ", ".join(f"{name} ({row['pr_auc']:.3f})" for name, row in by_pr.iterrows())
    baseline_accuracy = table.loc["Majority baseline", "accuracy"]
    return (
        f"Candidates were ranked by recall-weighted F2 (beta = {config.BETA:g}) at each model's own best threshold: {order_f2}. "
        f"Ranked by PR-AUC: {order_pr}. Accuracy was deliberately not used for selection because the majority baseline already reaches "
        f"{pct(baseline_accuracy)} accuracy while catching no churners. A missed churner costs the business a customer, while an unnecessary "
        f"retention call costs a few minutes of an agent, so recall carries more weight than precision."
    )


def data_section(results):
    info = results["data"]
    return (
        f"- Rows: {info['rows']:,} customers, {info['features']} raw input features, one binary target (`Churn`).\n"
        f"- Churn rate: {pct(info['churn_rate'])} (imbalanced, as expected for real churn data).\n"
        f"- Split: stratified {info['train_rows']:,} training rows ({pct(info['train_churn_rate'])} churn) and an untouched "
        f"hold-out of {info['test_rows']:,} rows ({pct(info['test_churn_rate'])} churn). Seed: {config.SEED}.\n"
        f"- Cross-validation: stratified {config.CV_FOLDS}-fold on the training rows only."
    )


def evaluation_report(results):
    lines = ["# ChurnGuard Evaluation Report", ""]
    if results is None:
        lines += ["All numbers in this report are produced by the training run.", "", PENDING, ""]
        return "\n".join(lines)
    holdout = results["holdout"]
    baseline = results["baseline_holdout"]
    report = holdout["classification_report"]
    lines += [
        f"Model version: `{results['tag']}`. Trained at {results['trained_at']}. Every number below is regenerated by `python -m churnguard.train`.",
        "",
        "## 1. Data and split",
        "",
        data_section(results),
        "",
        "## 2. Leakage check of engineered features",
        "",
        f"- Target column present in model inputs: {results['leakage']['target_in_inputs']}.",
        "- Every engineered feature is computed row by row from input columns only (no target, no statistics of other rows).",
        "- The feature step and all encoders are fitted inside each cross-validation fold, never on the hold-out.",
        f"- Univariate strength check (flag if above {results['leakage']['flag_threshold']}): "
        + ("no feature flagged." if not results["leakage"]["flagged_features"] else "flagged: " + ", ".join(results["leakage"]["flagged_features"]) + "."),
        "",
        md_table(
            results["leakage"]["checks"],
            [
                ("feature", "Engineered feature", None),
                ("univariate_auc", "Univariate AUC", lambda v: num(v, 4)),
                ("strength", "Strength", lambda v: num(v, 4)),
                ("flagged", "Flagged", lambda v: "yes" if v else "no"),
            ],
        ),
        "",
        "Justification of each engineered feature:",
        "",
    ]
    for name, text in FEATURE_JUSTIFICATION.items():
        lines.append(f"- `{name}`: {text}")
    lines += [
        "",
        "## 3. Model comparison (stratified 5-fold cross-validation on training data)",
        "",
        md_table(
            results["comparison"],
            [
                ("model", "Model", None),
                ("accuracy", "Accuracy", num),
                ("precision", "Precision", num),
                ("recall", "Recall", num),
                ("f1", "F1", num),
                ("roc_auc", "ROC-AUC", num),
                ("pr_auc", "PR-AUC", num),
                ("f2_at_best_threshold", "F2 (best threshold)", num),
            ],
        ),
        "",
        "Precision, recall and F1 above use the default 0.5 decision threshold. The F2 column lets each model use its own best threshold.",
        "",
        "### Is each change a meaningful improvement?",
        "",
    ]
    for statement in results["improvements"]:
        lines.append(f"- {statement}")
    lines += [
        "",
        "### Selection against the business priority",
        "",
        selection_text(results["comparison"]),
        "",
        "The final model is XGBoost, as named by the project brief, tuned and probability-calibrated. It is kept alongside the interpretable "
        "Logistic Regression, which stays in the repository as the explainable reference model.",
        "",
        "## 4. Hyperparameter tuning",
        "",
        f"RandomizedSearchCV, {results['tuning']['n_iter']} sampled configurations, stratified {config.CV_FOLDS}-fold, scored by PR-AUC (average precision). "
        "PR-AUC was chosen because it is sensitive to the minority class and is not inflated by the many easy non-churners. "
        "`scale_pos_weight` is part of the search so the model can trade off the class imbalance itself.",
        "",
        "Search space:",
        "",
    ]
    for key, text in results["tuning"]["search_space"].items():
        lines.append(f"- `{key}`: {text}")
    best = ", ".join(f"`{key}` = {num(value, 4) if isinstance(value, float) else value}" for key, value in results["tuning"]["best_params"].items())
    lines += [
        "",
        f"Chosen configuration (best cross-validated PR-AUC {results['tuning']['best_cv_pr_auc']:.4f}; slightly optimistic because it is the maximum over the search): {best}.",
        "",
        "## 5. Decision threshold",
        "",
        f"The threshold was chosen on out-of-fold training predictions by maximising F2 (recall counts {config.BETA:g} times as much as precision). "
        f"Chosen threshold: **{results['threshold']['value']:.2f}** (out-of-fold F2 {results['threshold']['oof_f2']:.3f}). The hold-out was not used.",
        "",
        md_table(
            results["threshold"]["table"],
            [
                ("threshold", "Threshold", lambda v: num(v, 2)),
                ("precision", "Precision", num),
                ("recall", "Recall", num),
                ("f2", "F2", num),
                ("flagged_share", "Share of customers flagged", pct),
            ],
        ),
        "",
        "## 6. Final hold-out results (untouched 20% of customers)",
        "",
        md_table(
            [
                {"name": "Majority baseline", **baseline},
                {"name": "ChurnGuard (calibrated XGBoost)", **holdout},
            ],
            [
                ("name", "Model", None),
                ("accuracy", "Accuracy", num),
                ("precision", "Precision", num),
                ("recall", "Recall", num),
                ("f1", "F1", num),
                ("f2", "F2", num),
                ("roc_auc", "ROC-AUC", num),
                ("pr_auc", "PR-AUC", num),
                ("brier", "Brier score", num),
            ],
        ),
        "",
        f"Operating threshold {holdout['threshold']:.2f}. The baseline row uses its own constant prediction.",
        "",
        "### Classification report (per class)",
        "",
        md_table(
            [
                {"label": "No churn", **report["No churn"]},
                {"label": "Churn", **report["Churn"]},
            ],
            [
                ("label", "Class", None),
                ("precision", "Precision", num),
                ("recall", "Recall", num),
                ("f1-score", "F1", num),
                ("support", "Support", lambda v: str(int(v))),
            ],
        ),
        "",
        "### Confusion matrix at the chosen threshold",
        "",
        "| | Predicted stay | Predicted churn |",
        "| --- | --- | --- |",
        f"| Actually stayed | {holdout['tn']} | {holdout['fp']} |",
        f"| Actually churned | {holdout['fn']} | {holdout['tp']} |",
        "",
        "Figures: `reports/roc_pr_curves.png`, `reports/confusion_matrix.png`.",
        "",
        "## 7. Calibration check",
        "",
        f"Hold-out predictions were split into ten equal-sized probability buckets. Expected calibration error: **{results['calibration']['summary']['expected_calibration_error']:.3f}**, "
        f"largest bucket gap: {results['calibration']['summary']['max_bucket_gap']:.3f}, Brier score: {holdout['brier']:.3f}.",
        "",
        md_table(
            results["calibration"]["table"],
            [
                ("customers", "Customers", lambda v: str(int(v))),
                ("mean_predicted", "Mean predicted", num),
                ("observed_rate", "Observed churn rate", num),
                ("gap", "Absolute gap", num),
            ],
        ),
        "",
        "Figure: `reports/calibration.png`. Probabilities are calibrated with Platt scaling (sigmoid) fitted on out-of-fold predictions.",
        "",
        "## 8. Error analysis by customer segment (hold-out)",
        "",
        f"Overall: {pct(results['error_analysis']['overall_fn_rate'])} of churners are missed (false negatives) and "
        f"{pct(results['error_analysis']['overall_fp_rate'])} of loyal customers are flagged (false positives).",
        "",
    ]
    for title, rows in results["error_analysis"]["tables"].items():
        lines += [
            f"**By {title.lower()}**",
            "",
            md_table(
                rows,
                [
                    ("segment", "Segment", None),
                    ("customers", "Customers", str),
                    ("churners", "Churners", str),
                    ("false_negatives", "Missed churners", str),
                    ("fn_rate", "Miss rate", pct),
                    ("false_positives", "False alarms", str),
                    ("fp_rate", "False alarm rate", pct),
                ],
            ),
            "",
        ]
    lines.append("**Findings**")
    lines.append("")
    for finding in results["error_analysis"]["findings"]:
        lines.append(f"- {finding}")
    lines += [
        "",
        "## 9. Interpretability",
        "",
        "Global importance is the mean absolute SHAP contribution, with one-hot columns summed back to their source feature. Figures: `reports/shap_summary.png`.",
        "",
        md_table(
            results["importance"][:12],
            [
                ("feature", "Feature", None),
                ("mean_abs_contribution", "Mean |SHAP| (log-odds)", lambda v: num(v, 4)),
            ],
        ),
        "",
        "Local explanations for four hold-out customers (the same text is returned by `POST /predict` and shown in the Streamlit app):",
        "",
    ]
    for example in results["local_examples"]:
        lines += [
            f"**{example['case']}** (actual churn: {example['actual_churn']}, predicted probability {example['probability']:.1%}, model decision: {example['predicted_label']})",
            "",
        ]
        for driver in example["drivers"]:
            lines.append(f"- {driver['explanation']}")
        lines.append("")
    lines += [
        "## 10. Reproducibility",
        "",
        f"Seed {config.SEED}, library versions: " + ", ".join(f"{key} {value}" for key, value in results["versions"].items()) + ".",
        "",
    ]
    return "\n".join(lines)


def model_card(results):
    lines = [
        "# Model Card: ChurnGuard",
        "",
        "## Problem",
        "",
        "Subscription businesses lose revenue to preventable churn but cannot review every customer by hand. ChurnGuard scores each customer with a calibrated churn probability and explains the three biggest drivers in plain English so a retention team can decide whom to contact first.",
        "",
        "## Model",
        "",
        "- Type: binary classifier on tabular data (churn = Yes/No).",
        "- Pipeline: feature engineering, one-hot encoding, XGBoost with class-imbalance weighting, Platt (sigmoid) probability calibration.",
        "- Candidates compared under identical cross-validation: majority baseline, Logistic Regression, Random Forest, XGBoost.",
        "- Explanations: exact TreeSHAP contributions of the underlying XGBoost model, aggregated per source feature.",
        "",
        "## Data",
        "",
        "IBM Telco Customer Churn sample dataset (7,043 customers, 20 features, binary `Churn` target, about 26.5% positive). It is a fictional telecom company in California in a single quarter.",
        "",
    ]
    if results is None:
        lines += ["## Metrics", "", PENDING, ""]
    else:
        holdout = results["holdout"]
        baseline = results["baseline_holdout"]
        lines += [
            "## Metrics (untouched hold-out)",
            "",
            f"Model version `{results['tag']}`, operating threshold {holdout['threshold']:.2f} (chosen to maximise recall-weighted F2 on training data).",
            "",
            md_table(
                [
                    {"name": "Majority baseline", **baseline},
                    {"name": "ChurnGuard", **holdout},
                ],
                [
                    ("name", "Model", None),
                    ("accuracy", "Accuracy", num),
                    ("precision", "Precision", num),
                    ("recall", "Recall", num),
                    ("f2", "F2", num),
                    ("roc_auc", "ROC-AUC", num),
                    ("pr_auc", "PR-AUC", num),
                    ("brier", "Brier", num),
                ],
            ),
            "",
            f"Calibration: expected calibration error {results['calibration']['summary']['expected_calibration_error']:.3f} across ten probability buckets.",
            "",
            "Full results: `EVALUATION_REPORT.md`.",
            "",
        ]
        if results["error_analysis"]["findings"]:
            lines += ["## Known weaknesses found in error analysis", ""]
            for finding in results["error_analysis"]["findings"]:
                lines.append(f"- {finding}")
            lines.append("")
    lines += [
        "## Known limitations",
        "",
        "- Trained on one fictional company and one quarter. Behaviour on a different telecom, country, price plan or time period is unknown and must be re-validated.",
        "- The model finds customers who look like past churners. It does not tell you why a customer is unhappy, and SHAP drivers describe the model, not a proven cause.",
        "- Customers with very short tenure have little history, so their scores are less reliable.",
        "- The dataset contains gender and senior-citizen fields. They are inputs to the model and must be reviewed for fairness before any real use.",
        "- Unseen category values are encoded as all zeros for that field rather than rejected, so a prediction is still returned but is less trustworthy.",
        "",
        "## Intended use",
        "",
        "- Daily triage list for a retention team, ranking customers for proactive contact together with a plain-English reason.",
        "- Decision support only. A person decides whether and how to contact a customer.",
        "",
        "## Out-of-scope use",
        "",
        "- Automated adverse action against a customer (denying service, raising prices, closing accounts) based on the score.",
        "- Credit, employment, insurance or any decision about eligibility.",
        "- Use on populations or products very different from the training data without re-validation.",
        "",
    ]
    return "\n".join(lines)


def write_docs(results):
    (config.ROOT / "EVALUATION_REPORT.md").write_text(evaluation_report(results), encoding="utf-8")
    (config.ROOT / "MODEL_CARD.md").write_text(model_card(results), encoding="utf-8")
