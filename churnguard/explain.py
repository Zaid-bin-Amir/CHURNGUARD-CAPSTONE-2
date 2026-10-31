import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from churnguard import config

FEATURE_LABELS = {
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen",
    "total_services": "Number of subscribed services",
    "charges_per_tenure": "Monthly charges relative to tenure",
    "contract_risk": "Month-to-month contract",
    "auto_pay": "Automatic payment",
    "security_support": "Online security plus tech support",
    "tenure_bucket": "Tenure group (months)",
    "gender": "Gender",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


def get_base_estimator(pipeline):
    classifier = pipeline.named_steps["clf"]
    if hasattr(classifier, "calibrated_classifiers_"):
        calibrated = classifier.calibrated_classifiers_[0]
        if hasattr(calibrated, "estimator"):
            return calibrated.estimator
        return calibrated.base_estimator
    return classifier


def encode(pipeline, X):
    engineered = pipeline.named_steps["features"].transform(X)
    encoded = pipeline.named_steps["prep"].transform(engineered)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    names = [str(name) for name in pipeline.named_steps["prep"].get_feature_names_out()]
    return engineered.reset_index(drop=True), np.asarray(encoded, dtype=float), names


def contributions(base, encoded):
    if hasattr(base, "get_booster"):
        import xgboost as xgb

        raw = base.get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)
        return np.asarray(raw)[:, :-1]
    import shap

    values = shap.TreeExplainer(base).shap_values(encoded)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def source_of(name):
    if name in config.NUMERIC_FEATURES:
        return name
    for feature in config.CATEGORICAL_FEATURES:
        if name.startswith(feature + "_"):
            return feature
    return name


def aggregate(values, names):
    sources = [source_of(name) for name in names]
    frame = pd.DataFrame(np.asarray(values), columns=sources)
    return frame.T.groupby(level=0).sum().T


def format_value(feature, value):
    if feature == "tenure":
        return f"{int(round(float(value)))} months"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return f"${float(value):,.2f}"
    if feature in ("SeniorCitizen", "contract_risk", "auto_pay", "security_support"):
        return "yes" if int(value) == 1 else "no"
    if feature == "total_services":
        return str(int(round(float(value))))
    if feature == "charges_per_tenure":
        return f"{float(value):.2f}"
    return str(value)


def describe_driver(source, impact, row):
    label = FEATURE_LABELS.get(source, source)
    value_text = format_value(source, row[source])
    verb = "raises" if impact > 0 else "lowers"
    return {
        "feature": label,
        "value": value_text,
        "impact": round(float(impact), 4),
        "direction": "increases risk" if impact > 0 else "decreases risk",
        "explanation": f"{label} ({value_text}) {verb} the predicted churn risk.",
    }


def explain(pipeline, X, top_k=3):
    engineered, encoded, names = encode(pipeline, X)
    grouped = aggregate(contributions(get_base_estimator(pipeline), encoded), names)
    results = []
    for position in range(len(grouped)):
        values = grouped.iloc[position]
        order = values.abs().sort_values(ascending=False).index[:top_k]
        row = engineered.iloc[position]
        results.append([describe_driver(source, float(values[source]), row) for source in order])
    return results


def global_importance(pipeline, X):
    engineered, encoded, names = encode(pipeline, X)
    grouped = aggregate(contributions(get_base_estimator(pipeline), encoded), names)
    importance = grouped.abs().mean().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": [FEATURE_LABELS.get(name, name) for name in importance.index],
            "source": list(importance.index),
            "mean_abs_contribution": importance.values,
        }
    )


def importance_figure(importance, top=15):
    subset = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["feature"], subset["mean_abs_contribution"], color="#1b9e77")
    ax.set_xlabel("Mean absolute SHAP contribution (log-odds)")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    return fig


def summary_figure(pipeline, X, importance):
    engineered, encoded, names = encode(pipeline, X)
    values = contributions(get_base_estimator(pipeline), encoded)
    try:
        import shap

        plt.figure(figsize=(9, 7))
        shap.summary_plot(values, encoded, feature_names=names, max_display=15, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig
    except Exception:
        plt.close("all")
        return importance_figure(importance)


def local_examples(pipeline, X, y, probabilities, threshold, top_k=3):
    actual = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    flagged = probabilities >= threshold
    positions = np.arange(len(actual))
    picks = []
    true_positive = positions[(actual == 1) & flagged]
    true_negative = positions[(actual == 0) & ~flagged]
    false_negative = positions[(actual == 1) & ~flagged]
    if len(true_positive):
        picks.append(("High-risk churner correctly flagged", int(true_positive[np.argmax(probabilities[true_positive])])))
    if len(true_negative):
        picks.append(("Loyal customer correctly cleared", int(true_negative[np.argmin(probabilities[true_negative])])))
    picks.append(("Borderline customer close to the threshold", int(np.argmin(np.abs(probabilities - threshold)))))
    if len(false_negative):
        picks.append(("Hard case: churner the model missed", int(false_negative[np.argmin(probabilities[false_negative])])))
    examples = []
    for title, position in picks:
        frame = X.iloc[[position]]
        drivers = explain(pipeline, frame, top_k=top_k)[0]
        examples.append(
            {
                "case": title,
                "actual_churn": "Yes" if int(actual[position]) == 1 else "No",
                "probability": round(float(probabilities[position]), 4),
                "predicted_label": "Yes" if flagged[position] else "No",
                "customer": json.loads(frame.to_json(orient="records"))[0],
                "drivers": drivers,
                "summary": " ".join(driver["explanation"] for driver in drivers),
            }
        )
    return examples
