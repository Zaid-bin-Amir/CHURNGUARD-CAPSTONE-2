# Monitoring Notes

This is a conceptual monitoring plan. Nothing here is automated.

## What is already collected

The API appends one JSON line per prediction to `logs/predictions.jsonl` with the timestamp, request ID, model version, all input fields, probability, label, risk level, top drivers and latency. This history is the raw material for every check below.

## Feature distributions to check for drift

| Feature | What to compare | Why it matters |
| --- | --- | --- |
| `tenure` | mean, median, share in each tenure bucket | new customer waves shift the whole risk profile |
| `MonthlyCharges` | mean, 5th and 95th percentile | a price change moves this directly |
| `Contract` | share of month-to-month | the strongest churn driver, so a shift changes the base rate |
| `InternetService` | share of Fiber optic | fibre customers churn more, so product mix matters |
| `PaymentMethod` | share of Electronic check | strongly linked to churn in this data |
| `TotalCharges` | mean and share of zeros | zeros mark brand new customers |
| New category values | count of values never seen in training | the model ignores them silently |

Also compare the predicted probability itself: average score, and the share of customers above the threshold. A sudden jump or fall is often the first visible sign of drift.

## How to compare

Take the last 30 days of logged inputs and compare each feature against the training data in `data/`. Use the population stability index (PSI) for categorical and bucketed numeric features, and the Kolmogorov-Smirnov test for continuous ones. Reasonable alert levels: PSI above 0.10 means investigate, above 0.25 means act.

## How often

- Weekly: feature distribution and score distribution check, count of unseen categories.
- Monthly: once true churn outcomes are known for the customers scored 30 to 60 days earlier, recompute recall, precision, PR-AUC and the calibration table, and compare with `EVALUATION_REPORT.md`.
- After any pricing change, new plan launch or marketing campaign: an extra check within one week.

## Retraining trigger

Retrain with `python -m churnguard.retrain` when any of these holds:

1. Recall on recent customers falls more than 5 percentage points below the hold-out recall in the model card.
2. PSI above 0.25 on two or more of the key features above, or on the predicted score.
3. Calibration gets worse: expected calibration error above 0.05, or the largest bucket gap above 0.10.
4. More than 5% of requests contain a category value never seen in training.
5. Six months have passed since the last training run, even if nothing else triggered.

After retraining, the new model should only replace the old one if it is at least as good on the new hold-out, and the change should be recorded in the evaluation report.

## Operational checks

- Latency: the typical response should stay under 2 seconds, excluding free-tier cold starts. Alert if the 95th percentile exceeds 2 seconds for a day.
- Error rate: alert on a rise in 401 or 422 responses, which can mean a broken client.
- Log growth: rotate or archive `predictions.jsonl` when it becomes large, and keep access restricted because it contains customer attributes.
