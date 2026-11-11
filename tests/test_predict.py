"""Tests for the prediction service (run from the project root: pytest)."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import app  # noqa: E402
from scoring import FEATURES, validate_record  # noqa: E402

client = TestClient(app)
SAMPLE = json.loads((ROOT / "sample_request.json").read_text())

HIGH_RISK = {**SAMPLE, "bmi": 45.0, "genhlth": 5, "highbp": 1, "highchol": 1,
             "diffwalk": 1, "age": 12, "physhlth": 20, "heartdiseaseorattack": 1}
LOW_RISK = {**SAMPLE, "bmi": 21.0, "genhlth": 1, "highbp": 0, "highchol": 0,
            "diffwalk": 0, "age": 3, "physactivity": 1, "income": 8}


def test_sample_has_every_feature():
    assert list(SAMPLE) == FEATURES


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_reports_model_and_threshold():
    body = client.get("/").json()
    assert body["service"] == "diabetes-health-indicators-predictor"
    assert body["feature_count"] == 21
    assert 0 < body["default_threshold"] < 1
    assert "disclaimer" in body


def test_predict_shape_and_ranges():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"diabetes_probability", "diabetes_prediction",
                         "threshold", "model"}
    assert 0.0 <= body["diabetes_probability"] <= 1.0
    assert body["diabetes_prediction"] in (0, 1)
    assert body["diabetes_prediction"] == int(
        body["diabetes_probability"] >= body["threshold"])


def test_higher_risk_profile_gets_higher_probability():
    low = client.post("/predict", json=LOW_RISK).json()
    high = client.post("/predict", json=HIGH_RISK).json()
    assert high["diabetes_probability"] > low["diabetes_probability"]
    assert high["diabetes_prediction"] == 1
    assert low["diabetes_prediction"] == 0


def test_threshold_override():
    base = client.post("/predict", json=SAMPLE).json()
    p = base["diabetes_probability"]
    always = client.post("/predict?threshold=0", json=SAMPLE).json()
    assert always["diabetes_prediction"] == 1 and always["threshold"] == 0.0
    if p < 1:
        strict = client.post(f"/predict?threshold={min(p + 1e-6, 1.0)}",
                             json=SAMPLE).json()
        assert strict["diabetes_prediction"] == 0


def test_threshold_out_of_range_is_422():
    assert client.post("/predict?threshold=1.5", json=SAMPLE).status_code == 422


def test_missing_field_is_422():
    bad = dict(SAMPLE)
    bad.pop("bmi")
    assert client.post("/predict", json=bad).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("genhlth", 99), ("age", 0), ("age", 14), ("bmi", 5), ("bmi", 500),
    ("highbp", 2), ("menthlth", 31), ("income", 9), ("sex", -1),
])
def test_out_of_range_value_is_422(field, value):
    bad = {**SAMPLE, field: value}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_field_is_422():
    assert client.post("/predict", json={**SAMPLE, "foo": 1}).status_code == 422


def test_cors_header_present():
    r = client.get("/health", headers={"Origin": "https://example.pages.dev"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.pages.dev")


def test_cors_preflight():
    r = client.options("/predict", headers={
        "Origin": "https://example.pages.dev",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200


def test_validate_record_helper():
    assert validate_record(SAMPLE)["bmi"] == 28.0
    with pytest.raises(ValueError):
        validate_record({**SAMPLE, "genhlth": 9})
    with pytest.raises(ValueError):
        validate_record({k: v for k, v in SAMPLE.items() if k != "age"})
    with pytest.raises(ValueError):
        validate_record({**SAMPLE, "extra": 1})
