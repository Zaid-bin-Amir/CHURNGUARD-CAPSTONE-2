import pandas as pd
import pytest

from churnguard import config, data
from churnguard.features import FeatureEngineer, count_services


def frame_with(**changes):
    record = dict(config.SAMPLE_CUSTOMER)
    record.update(changes)
    return pd.DataFrame([record])[config.RAW_FEATURES]


def test_engineered_columns_exist():
    result = FeatureEngineer().transform(frame_with())
    for column in config.ENGINEERED_NUMERIC + config.ENGINEERED_CATEGORICAL:
        assert column in result.columns


def test_input_frame_is_not_modified():
    original = frame_with()
    snapshot = original.copy()
    FeatureEngineer().transform(original)
    pd.testing.assert_frame_equal(original, snapshot)


@pytest.mark.parametrize(
    "tenure, expected",
    [(0, "0-12"), (12, "0-12"), (13, "13-24"), (24, "13-24"), (25, "25-48"), (48, "25-48"), (49, "49+"), (72, "49+")],
)
def test_tenure_bucket(tenure, expected):
    result = FeatureEngineer().transform(frame_with(tenure=tenure))
    assert result["tenure_bucket"].iloc[0] == expected


def test_total_services_counts_subscriptions():
    base = frame_with(
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="No",
        OnlineSecurity="No internet service",
        OnlineBackup="No internet service",
        DeviceProtection="No internet service",
        TechSupport="No internet service",
        StreamingTV="No internet service",
        StreamingMovies="No internet service",
    )
    assert count_services(base).iloc[0] == 0
    full = frame_with(
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="Fiber optic",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="Yes",
        StreamingMovies="Yes",
    )
    assert count_services(full).iloc[0] == 9


def test_flags_and_ratio():
    result = FeatureEngineer().transform(
        frame_with(Contract="Two year", PaymentMethod="Credit card (automatic)", OnlineSecurity="Yes", TechSupport="Yes", tenure=9, MonthlyCharges=50.0)
    )
    assert result["contract_risk"].iloc[0] == 0
    assert result["auto_pay"].iloc[0] == 1
    assert result["security_support"].iloc[0] == 1
    assert result["charges_per_tenure"].iloc[0] == pytest.approx(5.0)
    month_to_month = FeatureEngineer().transform(frame_with(Contract="Month-to-month", PaymentMethod="Mailed check"))
    assert month_to_month["contract_risk"].iloc[0] == 1
    assert month_to_month["auto_pay"].iloc[0] == 0


def test_cleaning_handles_blank_total_charges():
    raw = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "tenure": [0, 5],
            "MonthlyCharges": [20.0, 30.0],
            "TotalCharges": [" ", "150.5"],
            "Churn": ["No", "Yes"],
        }
    )
    cleaned = data.clean(raw)
    assert cleaned["TotalCharges"].tolist() == [0.0, 150.5]
    assert cleaned["Churn"].tolist() == [0, 1]


def test_real_dataset_shape_and_target():
    frame = data.load_clean()
    assert frame.shape[0] == 7043
    assert set(frame["Churn"].unique()) == {0, 1}
    assert 0.25 <= frame["Churn"].mean() <= 0.30
    assert frame["TotalCharges"].isna().sum() == 0
