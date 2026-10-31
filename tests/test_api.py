import json
import os
from pathlib import Path

from churnguard import config


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1
    assert body["model_version"]


def test_predict_happy_path(client, headers, customer):
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Yes", "No")
    assert body["risk_level"] in ("low", "medium", "high")
    assert len(body["top_drivers"]) == 3
    for driver in body["top_drivers"]:
        assert driver["explanation"].endswith("churn risk.")
        assert driver["direction"] in ("increases risk", "decreases risk")
    assert body["summary"]
    assert "X-Request-ID" in response.headers


def test_missing_field_is_rejected(client, headers, customer):
    del customer["tenure"]
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 422


def test_invalid_value_is_rejected(client, headers, customer):
    customer["tenure"] = -5
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 422
    customer["tenure"] = 5
    customer["MonthlyCharges"] = "not a number"
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 422


def test_unknown_field_is_rejected(client, headers, customer):
    customer["favourite_colour"] = "green"
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 422


def test_unseen_category_is_handled(client, headers, customer):
    customer["Contract"] = "Quarterly"
    customer["PaymentMethod"] = "Cryptocurrency"
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_missing_api_key_fails(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 401


def test_wrong_api_key_fails(client, customer):
    response = client.post("/predict", json=customer, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_regression_known_input_output_pair(client, headers):
    case = json.loads((config.MODELS_DIR / "regression_case.json").read_text(encoding="utf-8"))
    response = client.post("/predict", json=case["input"], headers=headers)
    assert response.status_code == 200
    assert abs(response.json()["probability"] - case["expected_probability"]) < 1e-3


def test_batch_endpoint(client, headers, customer):
    other = dict(customer)
    other["Contract"] = "Two year"
    other["tenure"] = 60
    response = client.post("/predict/batch", json={"customers": [customer, other]}, headers=headers)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [item["index"] for item in predictions] == [0, 1]
    assert predictions[0]["probability"] > predictions[1]["probability"]


def test_batch_requires_api_key(client, customer):
    response = client.post("/predict/batch", json={"customers": [customer]})
    assert response.status_code == 401


def test_predictions_are_logged(client, headers, customer):
    client.post("/predict", json=customer, headers=headers)
    log_path = Path(os.environ["CHURNGUARD_LOG_DIR"]) / "predictions.jsonl"
    assert log_path.exists()
    last = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["endpoint"] in ("/predict", "/predict/batch")
    assert "probability" in last
    assert "inputs" in last
    assert "X-API-Key" not in json.dumps(last)
