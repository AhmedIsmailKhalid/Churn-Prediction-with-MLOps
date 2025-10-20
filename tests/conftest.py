"""
Pytest configuration and shared fixtures.

This module provides fixtures used across all test modules.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def sample_customer_data() -> pd.DataFrame:
    """
    Create sample customer data for testing.
    
    Returns:
        DataFrame with sample customer records
    """
    data = {
        'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK'],
        'gender': ['Female', 'Male', 'Male'],
        'SeniorCitizen': [0, 0, 0],
        'Partner': ['Yes', 'No', 'No'],
        'Dependents': ['No', 'No', 'No'],
        'tenure': [1, 34, 2],
        'PhoneService': ['No', 'Yes', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'No'],
        'InternetService': ['DSL', 'DSL', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'Yes'],
        'OnlineBackup': ['Yes', 'No', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No'],
        'StreamingTV': ['No', 'No', 'No'],
        'StreamingMovies': ['No', 'No', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check'],
        'MonthlyCharges': [29.85, 56.95, 53.85],
        'TotalCharges': [29.85, 1889.5, 108.15],
        'Churn': ['No', 'No', 'Yes']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_customer_features() -> dict:
    """
    Sample customer features for API testing.
    
    Returns:
        Dictionary with customer features
    """
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 65.50,
        "total_charges": 786.00,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No"
    }


@pytest.fixture
def mock_model_instance():
    """
    Create a mock model instance with predict capabilities.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
    return mock_model


@pytest.fixture
def mock_model_loader(mock_model_instance):
    """
    Create a mock ModelLoader instance.
    """
    mock_loader = MagicMock()
    mock_loader.get_model.return_value = mock_model_instance
    mock_loader.load_model.return_value = mock_model_instance
    
    # Mock preprocessing to return a simple DataFrame
    def mock_preprocess(df):
        # Return a DataFrame with 10 dummy features
        return pd.DataFrame([[1] * 10], columns=[f'feature_{i}' for i in range(10)])
    
    mock_loader.preprocess_features.side_effect = mock_preprocess
    
    mock_loader.get_model_info.return_value = {
        "name": "churn_prediction",
        "version": "test-v1.0.0",
        "stage": "Test",
        "loaded_at": "2024-01-01T00:00:00",
        "mlflow_run_id": "test-run-id-12345",
        "metrics": {
            "accuracy": 0.85,
            "roc_auc": 0.88,
            "f1_score": 0.82
        }
    }
    
    return mock_loader


@pytest.fixture
def mock_model(mock_model_loader):
    """
    Mock the model loader to avoid requiring MLflow during tests.
    This fixture patches the get_model_loader function before the app starts.
    """
    with patch("src.api.dependencies.get_model_loader") as mock_get_loader:
        # Set the mock to return our mock loader
        mock_get_loader.return_value = mock_model_loader
        
        # Also patch the global _model_loader_instance
        with patch("src.api.dependencies._model_loader_instance", mock_model_loader):
            yield mock_model_loader


@pytest.fixture
def api_client(mock_model):
    """
    Create a test client for the FastAPI application.
    
    The mock_model fixture is applied before creating the client,
    ensuring the app starts up with mocked dependencies.
    """
    # Need to import after patching to ensure patches are active during import
    from src.api.main import app, ModelState
    
    # Manually set the model state for tests
    ModelState.model_loader = mock_model
    ModelState.model_info = mock_model.get_model_info()
    ModelState.startup_time = "2024-01-01T00:00:00"
    
    # Create test client
    client = TestClient(app)
    
    yield client
    
    # Cleanup
    ModelState.model_loader = None
    ModelState.model_info = None
    ModelState.startup_time = None