"""
Unit tests for API endpoints.
"""

import pytest
from fastapi import status


def test_root_endpoint(api_client, mock_model):
    """Test root endpoint returns API information."""
    response = api_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint(api_client, mock_model):
    """Test health check endpoint."""
    response = api_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data


def test_ready_endpoint(api_client, mock_model):
    """Test readiness check endpoint."""
    response = api_client.get("/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"


def test_model_info_endpoint(api_client, mock_model):
    """Test model info endpoint."""
    response = api_client.get("/model/info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["name"] == "churn_prediction"


def test_predict_endpoint(api_client, mock_model, sample_customer_features):
    """Test prediction endpoint with valid data."""
    request_data = {
        "customer_id": "test-123",
        "features": sample_customer_features
    }
    
    response = api_client.post("/predict", json=request_data)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "prediction_label" in data
    assert data["customer_id"] == "test-123"
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_endpoint_invalid_data(api_client, mock_model):
    """Test prediction endpoint with invalid data."""
    invalid_request = {
        "customer_id": "test-123",
        "features": {
            "gender": "Invalid",  # Invalid gender
            "senior_citizen": 0,
            "partner": "Yes",
            # Missing required fields...
        }
    }
    
    response = api_client.post("/predict", json=invalid_request)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_metrics_endpoint(api_client, mock_model):
    """Test Prometheus metrics endpoint."""
    response = api_client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "text/plain" in response.headers["content-type"]