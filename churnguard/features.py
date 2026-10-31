import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from churnguard import config

TENURE_BINS = [-1, 12, 24, 48, 100000]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]

FEATURE_JUSTIFICATION = {
    "tenure_bucket": "Hypothesis: churn risk is highest early in the customer lifetime and falls with loyalty. Buckets capture that non-linear shape for linear models and give retention staff readable segments.",
    "total_services": "Hypothesis: customers subscribed to more services are more embedded in the product and switch less. One count summarises nine service columns.",
    "charges_per_tenure": "Hypothesis: a high monthly bill relative to time with the company marks newer customers who feel they overpay. Computed as MonthlyCharges / (tenure + 1).",
    "contract_risk": "Hypothesis: month-to-month contracts carry no commitment and are the easiest to leave. The flag makes this segment explicit.",
    "auto_pay": "Hypothesis: automatic payment reduces forgotten or failed payments and signals commitment.",
    "security_support": "Hypothesis: customers who have both online security and tech support receive more service touchpoints and tend to stay.",
}


def count_services(frame):
    subscribed = (frame[config.SERVICE_COLUMNS] == "Yes").sum(axis=1)
    internet = (frame["InternetService"] != "No").astype(int)
    return subscribed + internet


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame["tenure_bucket"] = pd.cut(
            frame["tenure"].astype(float),
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
        ).astype(str)
        frame["total_services"] = count_services(frame).astype(float)
        frame["charges_per_tenure"] = frame["MonthlyCharges"].astype(float) / (frame["tenure"].astype(float) + 1.0)
        frame["contract_risk"] = (frame["Contract"] == "Month-to-month").astype(int)
        frame["auto_pay"] = frame["PaymentMethod"].astype(str).str.contains("automatic", case=False, na=False).astype(int)
        frame["security_support"] = ((frame["OnlineSecurity"] == "Yes") & (frame["TechSupport"] == "Yes")).astype(int)
        return frame
