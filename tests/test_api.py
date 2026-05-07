"""Smoke + behaviour tests untuk endpoint utama FaultSense API."""

from __future__ import annotations

from typing import Any, Dict

VALID_PAYLOAD: Dict[str, Any] = {
    "type"               : "L",
    "air_temperature"    : 298.0,
    "process_temperature": 308.0,
    "rotational_speed"   : 1500,
    "torque"             : 40.0,
    "tool_wear"          : 50,
}


# ----------------------------------------------------------------------
# Health & root
# ----------------------------------------------------------------------
def test_root_returns_metadata(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("FaultSense API")
    assert "endpoints" in data


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"


# ----------------------------------------------------------------------
# /predict — happy path
# ----------------------------------------------------------------------
def test_predict_valid_input_returns_normal_or_failure(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"Normal", "Failure"}
    assert 0.0 <= data["probability_failure"] <= 1.0
    assert "checked_at" in data


def test_predict_lowercase_type_is_normalized(client):
    payload = dict(VALID_PAYLOAD, type="l")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] in {"Normal", "Failure"}


def test_predict_logs_to_history(client):
    pre = client.get("/history").json()
    pre_count = len(pre)

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    post = client.get("/history").json()
    assert len(post) == pre_count + 1


# ----------------------------------------------------------------------
# /predict — validation
# ----------------------------------------------------------------------
def test_predict_invalid_type_returns_422(client):
    payload = dict(VALID_PAYLOAD, type="X")
    response = client.post("/predict", json=payload)
    # Validasi `Literal['H','L','M']` di Pydantic harus menolak input ini
    # dengan HTTP 422, bukan 500.
    assert response.status_code == 422


def test_predict_negative_torque_returns_422(client):
    payload = dict(VALID_PAYLOAD, torque=-1.0)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    payload = {"type": "L"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_failure_case_returns_known_label(client):
    # Skenario heavy load yang biasanya menghasilkan klasifikasi Failure.
    payload = dict(
        VALID_PAYLOAD,
        air_temperature=305.0,
        process_temperature=320.0,
        rotational_speed=2800,
        torque=75.0,
        tool_wear=240,
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    if data["status"] == "Failure":
        from app.prediction import label_map

        # Hanya kelas yang benar-benar dipelajari model boleh muncul.
        valid_labels = set(label_map.values())
        assert data["failure_type"] in valid_labels


# ----------------------------------------------------------------------
# /history & /analytics
# ----------------------------------------------------------------------
def test_history_empty_initially(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_analytics_empty_state(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 0
    assert data["failure_rate"] == 0.0
    assert data["avg_probability"] == 0.0


def test_analytics_after_prediction(client):
    client.post("/predict", json=VALID_PAYLOAD)
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 1
    assert data["total_normal"] + data["total_failure"] == 1


def test_history_limit_validation(client):
    # limit 0 should fail validation (we set ge=1)
    response = client.get("/history?limit=0")
    assert response.status_code == 422
