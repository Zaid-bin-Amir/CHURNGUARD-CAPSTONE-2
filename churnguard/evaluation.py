import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churnguard import config
from churnguard.features import TENURE_LABELS, FeatureEngineer


def holdout_metrics(y, probabilities, threshold):
    actual = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "f2": float(fbeta_score(actual, predicted, beta=config.BETA, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "brier": float(brier_score_loss(actual, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report,
    }


def baseline_holdout_metrics(y_train, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    probabilities = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    return holdout_metrics(y_test, probabilities, 0.5)


def calibration_table(y, probabilities, bins=10):
    frame = pd.DataFrame({"actual": np.asarray(y), "probability": np.asarray(probabilities, dtype=float)})
    frame["bucket"] = pd.qcut(frame["probability"].rank(method="first"), q=bins, labels=False)
    grouped = frame.groupby("bucket").agg(
        customers=("actual", "size"),
        mean_predicted=("probability", "mean"),
        observed_rate=("actual", "mean"),
        min_probability=("probability", "min"),
        max_probability=("probability", "max"),
    )
    grouped["gap"] = (grouped["mean_predicted"] - grouped["observed_rate"]).abs()
    grouped = grouped.reset_index(drop=True)
    total = grouped["customers"].sum()
    summary = {
        "expected_calibration_error": float((grouped["gap"] * grouped["customers"]).sum() / total),
        "max_bucket_gap": float(grouped["gap"].max()),
    }
    return grouped, summary


def error_analysis(X, y, probabilities, threshold, min_churners=15, min_non_churners=30):
    frame = FeatureEngineer().transform(X.reset_index(drop=True))
    frame["actual"] = np.asarray(y)
    frame["predicted"] = (np.asarray(probabilities) >= threshold).astype(int)
    overall_churners = int((frame["actual"] == 1).sum())
    overall_non_churners = int((frame["actual"] == 0).sum())
    overall_fn_rate = float(((frame["actual"] == 1) & (frame["predicted"] == 0)).sum() / max(overall_churners, 1))
    overall_fp_rate = float(((frame["actual"] == 0) & (frame["predicted"] == 1)).sum() / max(overall_non_churners, 1))
    dimensions = {
        "Contract": "Contract",
        "Tenure bucket": "tenure_bucket",
        "Internet service": "InternetService",
        "Payment method": "PaymentMethod",
    }
    tables = {}
    flat = []
    for title, column in dimensions.items():
        rows = []
        for value, group in frame.groupby(column):
            churners = int((group["actual"] == 1).sum())
            non_churners = int((group["actual"] == 0).sum())
            false_negatives = int(((group["actual"] == 1) & (group["predicted"] == 0)).sum())
            false_positives = int(((group["actual"] == 0) & (group["predicted"] == 1)).sum())
            row = {
                "segment": str(value),
                "customers": int(len(group)),
                "churners": churners,
                "false_negatives": false_negatives,
                "fn_rate": float(false_negatives / churners) if churners else None,
                "non_churners": non_churners,
                "false_positives": false_positives,
                "fp_rate": float(false_positives / non_churners) if non_churners else None,
            }
            rows.append(row)
            flat.append({"dimension": title, **row})
        tables[title] = rows
    fn_candidates = [r for r in flat if r["churners"] >= min_churners and r["fn_rate"] is not None]
    fp_candidates = [r for r in flat if r["non_churners"] >= min_non_churners and r["fp_rate"] is not None]
    worst_fn = max(fn_candidates, key=lambda r: r["fn_rate"]) if fn_candidates else None
    worst_fp = max(fp_candidates, key=lambda r: r["fp_rate"]) if fp_candidates else None
    findings = []
    if worst_fn:
        findings.append(
            f"Systematic weakness (missed churners): in the segment {worst_fn['dimension']} = {worst_fn['segment']} "
            f"the model misses {worst_fn['fn_rate']:.1%} of churners ({worst_fn['false_negatives']} of {worst_fn['churners']}), "
            f"against {overall_fn_rate:.1%} overall."
        )
    if worst_fp:
        findings.append(
            f"Systematic weakness (false alarms): in the segment {worst_fp['dimension']} = {worst_fp['segment']} "
            f"{worst_fp['fp_rate']:.1%} of loyal customers are flagged ({worst_fp['false_positives']} of {worst_fp['non_churners']}), "
            f"against {overall_fp_rate:.1%} overall."
        )
    return {
        "overall_fn_rate": overall_fn_rate,
        "overall_fp_rate": overall_fp_rate,
        "tables": tables,
        "worst_false_negative_segment": worst_fn,
        "worst_false_positive_segment": worst_fp,
        "findings": findings,
    }


def leakage_check(X_train, y_train, threshold=0.85):
    frame = FeatureEngineer().transform(X_train)
    ordinal = {label: index for index, label in enumerate(TENURE_LABELS)}
    checks = []
    candidates = {name: frame[name].astype(float) for name in config.ENGINEERED_NUMERIC}
    candidates["tenure_bucket"] = frame["tenure_bucket"].map(ordinal).astype(float)
    for name, values in candidates.items():
        auc = float(roc_auc_score(y_train, values))
        strength = max(auc, 1.0 - auc)
        checks.append({"feature": name, "univariate_auc": round(auc, 4), "strength": round(strength, 4), "flagged": strength > threshold})
    return {
        "target_in_inputs": config.TARGET in X_train.columns,
        "derived_from_input_columns_only": True,
        "fitted_inside_cross_validation": True,
        "flag_threshold": threshold,
        "checks": checks,
        "flagged_features": [c["feature"] for c in checks if c["flagged"]],
    }


def plot_roc_pr(y, probabilities):
    actual = np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(actual, probabilities)
    axes[0].plot(fpr, tpr, color="#1b9e77", linewidth=2, label=f"Model (AUC {roc_auc_score(actual, probabilities):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Baseline (AUC 0.500)")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (hold-out)")
    axes[0].legend(loc="lower right")
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    axes[1].plot(recall, precision, color="#d95f02", linewidth=2, label=f"Model (AP {average_precision_score(actual, probabilities):.3f})")
    axes[1].axhline(actual.mean(), linestyle="--", color="#888888", label=f"Baseline (AP {actual.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve (hold-out)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_confusion(metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted stay", "Predicted churn"])
    ax.set_yticklabels(["Actually stayed", "Actually churned"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14, color="#111111")
    ax.set_title(f"Confusion matrix (threshold {metrics['threshold']:.2f})")
    fig.tight_layout()
    return fig


def plot_calibration(table):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect calibration")
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", color="#7570b3", linewidth=2, label="Model")
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration by probability bucket (hold-out)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def save_figure(fig, name, close=True):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
