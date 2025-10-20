"""
Pydantic schemas for API request/response validation.

This module defines all data models used in the FastAPI application,
ensuring type safety and automatic validation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ServiceStatus(BaseModel):
    """Status of a dependent service."""
    name: str = Field(..., description="Service name")
    status: HealthStatus = Field(..., description="Service health status")
    latency_ms: Optional[float] = Field(None, description="Service response latency in milliseconds")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: HealthStatus = Field(..., description="Overall service health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, ServiceStatus] = Field(default_factory=dict, description="Status of dependent services")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00",
            "version": "1.0.0",
            "services": {
                "mlflow": {
                    "name": "mlflow",
                    "status": "healthy",
                    "latency_ms": 45.2
                },
                "redis": {
                    "name": "redis",
                    "status": "healthy",
                    "latency_ms": 2.1
                }
            }
        }
    })


class CustomerFeatures(BaseModel):
    """
    Customer features for churn prediction.
    
    Based on Telco Customer Churn dataset schema.
    """
    # Demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0/1)", alias="senior_citizen")
    Partner: str = Field(..., description="Whether customer has partner (Yes/No)", alias="partner")
    Dependents: str = Field(..., description="Whether customer has dependents (Yes/No)", alias="dependents")
    
    # Account information
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with company")
    Contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)", alias="contract")
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing (Yes/No)", alias="paperless_billing")
    PaymentMethod: str = Field(..., description="Payment method", alias="payment_method")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges amount", alias="monthly_charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges amount", alias="total_charges")
    
    # Services
    PhoneService: str = Field(..., description="Whether customer has phone service (Yes/No)", alias="phone_service")
    MultipleLines: str = Field(..., description="Whether customer has multiple lines (Yes/No/No phone service)", alias="multiple_lines")
    InternetService: str = Field(..., description="Internet service type (DSL/Fiber optic/No)", alias="internet_service")
    OnlineSecurity: str = Field(..., description="Whether customer has online security (Yes/No/No internet service)", alias="online_security")
    OnlineBackup: str = Field(..., description="Whether customer has online backup (Yes/No/No internet service)", alias="online_backup")
    DeviceProtection: str = Field(..., description="Whether customer has device protection (Yes/No/No internet service)", alias="device_protection")
    TechSupport: str = Field(..., description="Whether customer has tech support (Yes/No/No internet service)", alias="tech_support")
    StreamingTV: str = Field(..., description="Whether customer has streaming TV (Yes/No/No internet service)", alias="streaming_tv")
    StreamingMovies: str = Field(..., description="Whether customer has streaming movies (Yes/No/No internet service)", alias="streaming_movies")
    
    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field name and alias
        json_schema_extra={
            "example": {
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
        }
    )
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validate gender field."""
        allowed = ['Male', 'Female']
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v
    
    @field_validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Validate Yes/No fields."""
        allowed = ['Yes', 'No']
        if v not in allowed:
            raise ValueError(f"Value must be one of {allowed}")
        return v
    
    @field_validator('Contract')
    @classmethod
    def validate_contract(cls, v: str) -> str:
        """Validate contract type."""
        allowed = ['Month-to-month', 'One year', 'Two year']
        if v not in allowed:
            raise ValueError(f"Contract must be one of {allowed}")
        return v
    
    @field_validator('InternetService')
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        """Validate internet service type."""
        allowed = ['DSL', 'Fiber optic', 'No']
        if v not in allowed:
            raise ValueError(f"Internet service must be one of {allowed}")
        return v


class PredictionRequest(BaseModel):
    """Request model for churn prediction."""
    customer_id: Optional[str] = Field(None, description="Optional customer ID for tracking")
    features: CustomerFeatures = Field(..., description="Customer features")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "7590-VHVEG",
            "features": {
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
        }
    })


class PredictionResponse(BaseModel):
    """Response model for churn prediction."""
    customer_id: Optional[str] = Field(None, description="Customer ID if provided")
    prediction: int = Field(..., ge=0, le=1, description="Churn prediction (0=No Churn, 1=Churn)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "7590-VHVEG",
            "prediction": 1,
            "prediction_label": "Churn",
            "probability": 0.75,
            "confidence": 0.75,
            "timestamp": "2024-01-15T10:30:00",
            "model_version": "v1.2.0"
        }
    })


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
    
    @field_validator('predictions')
    @classmethod
    def validate_batch_size(cls, v: List[PredictionRequest]) -> List[PredictionRequest]:
        """Validate batch size."""
        if len(v) == 0:
            raise ValueError("Batch must contain at least one prediction request")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "predictions": [
                {
                    "customer_id": "7590-VHVEG",
                    "features": {
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
                }
            ]
        }
    })


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(default=0, description="Number of failed predictions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch processing timestamp")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "predictions": [
                {
                    "customer_id": "7590-VHVEG",
                    "prediction": 1,
                    "prediction_label": "Churn",
                    "probability": 0.75,
                    "confidence": 0.75,
                    "timestamp": "2024-01-15T10:30:00",
                    "model_version": "v1.2.0"
                }
            ],
            "total_predictions": 1,
            "successful_predictions": 1,
            "failed_predictions": 0,
            "timestamp": "2024-01-15T10:30:00"
        }
    })


class ModelInfo(BaseModel):
    """Information about the loaded model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage (Production/Staging/etc)")
    loaded_at: datetime = Field(..., description="When model was loaded")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "churn_prediction",
            "version": "v1.2.0",
            "stage": "Production",
            "loaded_at": "2024-01-15T09:00:00",
            "mlflow_run_id": "abc123def456",
            "metrics": {
                "accuracy": 0.85,
                "roc_auc": 0.88,
                "f1_score": 0.82
            }
        }
    })


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "ValidationError",
            "message": "Invalid input data",
            "detail": {
                "field": "tenure",
                "error": "must be non-negative"
            },
            "timestamp": "2024-01-15T10:30:00"
        }
    })


class MetricsResponse(BaseModel):
    """Response model for Prometheus metrics endpoint."""
    message: str = Field(default="Metrics available at /metrics", description="Information message")
    metrics_format: str = Field(default="prometheus", description="Metrics format")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "message": "Metrics available at /metrics",
            "metrics_format": "prometheus"
        }
    })