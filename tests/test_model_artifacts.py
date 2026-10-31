import numpy as np
import pandas as pd
import pytest

from churnguard import config, registry


@pytest.fixture(scope="module")
def loaded():
    return registry.load_or_train()


def test_saved_pipeline_predicts_valid_probability(loaded):
    pipeline, meta = loaded
    frame = pd.DataFrame([config.SAMPLE_CUSTOMER])[config.RAW_FEATURES]
    probability = pipeline.predict_proba(frame)[0, 1]
    assert 0.0 <= probability <= 1.0
    assert 0.0 < meta["threshold"] < 1.0


def test_version_tag_reflects_date_and_description(loaded):
    _, meta = loaded
    assert meta["tag"].count("_") >= 1
    assert meta["tag"].split("_")[0].isdigit()


def test_skops_round_trip_matches_joblib(loaded):
    pytest.importorskip("skops")
    pipeline, meta = loaded
    if not meta.get("skops_file"):
        pytest.skip("skops file was not written for this model")
    restored = registry.load_skops(config.MODELS_DIR / meta["skops_file"])
    frame = pd.DataFrame([config.SAMPLE_CUSTOMER])[config.RAW_FEATURES]
    np.testing.assert_allclose(pipeline.predict_proba(frame), restored.predict_proba(frame), atol=1e-9)


def test_unseen_category_does_not_break_pipeline(loaded):
    pipeline, _ = loaded
    record = dict(config.SAMPLE_CUSTOMER)
    record["InternetService"] = "Satellite"
    frame = pd.DataFrame([record])[config.RAW_FEATURES]
    probability = pipeline.predict_proba(frame)[0, 1]
    assert 0.0 <= probability <= 1.0
