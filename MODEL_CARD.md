# Model Card: ChurnGuard

## Problem

Subscription businesses lose revenue to preventable churn but cannot review every customer by hand. ChurnGuard scores each customer with a calibrated churn probability and explains the three biggest drivers in plain English so a retention team can decide whom to contact first.

## Model

- Type: binary classifier on tabular data (churn = Yes/No).
- Pipeline: feature engineering, one-hot encoding, XGBoost with class-imbalance weighting, Platt (sigmoid) probability calibration.
- Candidates compared under identical cross-validation: majority baseline, Logistic Regression, Random Forest, XGBoost.
- Explanations: exact TreeSHAP contributions of the underlying XGBoost model, aggregated per source feature.

## Data

IBM Telco Customer Churn sample dataset (7,043 customers, 20 features, binary `Churn` target, about 26.5% positive). It is a fictional telecom company in California in a single quarter.

## Metrics (untouched hold-out)

Model version `20261031_161824_retrain`, operating threshold 0.12 (chosen to maximise recall-weighted F2 on training data).

| Model | Accuracy | Precision | Recall | F2 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.734 | 0.000 | 0.000 | 0.000 | 0.500 | 0.266 | 0.266 |
| ChurnGuard | 0.683 | 0.451 | 0.901 | 0.751 | 0.845 | 0.644 | 0.138 |

Calibration: expected calibration error 0.039 across ten probability buckets.

Full results: `EVALUATION_REPORT.md`.

## Known weaknesses found in error analysis

- Systematic weakness (missed churners): in the segment Internet service = No the model misses 48.1% of churners (13 of 27), against 9.9% overall.
- Systematic weakness (false alarms): in the segment Contract = Month-to-month 78.6% of loyal customers are flagged (363 of 462), against 39.7% overall.

## Known limitations

- Trained on one fictional company and one quarter. Behaviour on a different telecom, country, price plan or time period is unknown and must be re-validated.
- The model finds customers who look like past churners. It does not tell you why a customer is unhappy, and SHAP drivers describe the model, not a proven cause.
- Customers with very short tenure have little history, so their scores are less reliable.
- The dataset contains gender and senior-citizen fields. They are inputs to the model and must be reviewed for fairness before any real use.
- Unseen category values are encoded as all zeros for that field rather than rejected, so a prediction is still returned but is less trustworthy.

## Intended use

- Daily triage list for a retention team, ranking customers for proactive contact together with a plain-English reason.
- Decision support only. A person decides whether and how to contact a customer.

## Out-of-scope use

- Automated adverse action against a customer (denying service, raising prices, closing accounts) based on the score.
- Credit, employment, insurance or any decision about eligibility.
- Use on populations or products very different from the training data without re-validation.
