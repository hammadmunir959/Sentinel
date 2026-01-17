"""
Tests for the FastAPI application.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Sentinel Fraud Detection API"
    assert data["status"] == "running"


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_no_model():
    """Test prediction when model is not loaded."""
    # This test may pass or fail depending on whether model is loaded
    sample = {
        "Time": 0.0,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
        "Amount": 149.62
    }
    response = client.post("/predict", json=sample)
    # Either 200 (model loaded) or 503 (model not loaded)
    assert response.status_code in [200, 503]


def test_predict_invalid_input():
    """Test prediction with invalid input."""
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422  # Validation error


def test_batch_empty():
    """Test batch prediction with empty list."""
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 400
