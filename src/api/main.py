"""
FastAPI application for churn prediction service.

This module implements a production-ready REST API for serving churn predictions
with comprehensive error handling, monitoring, and health checks.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from src.api.dependencies import ModelLoader, get_model_loader
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    ServiceStatus,
)
from src.monitoring.drift_detection import DriftDetector

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'churn_predictions_total',
    'Total number of predictions made',
    ['prediction_label']
)

PREDICTION_LATENCY = Histogram(
    'churn_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTION_PROBABILITY = Histogram(
    'churn_prediction_probability',
    'Distribution of prediction probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

API_REQUEST_COUNTER = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

API_REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['method', 'endpoint']
)

MODEL_LOAD_TIME = Gauge(
    'model_load_timestamp',
    'Timestamp when model was loaded'
)

HEALTH_STATUS = Gauge(
    'service_health_status',
    'Service health status (1=healthy, 0=unhealthy)'
)


class ModelState:
    """Global state for model and dependencies."""
    model_loader: Optional[ModelLoader] = None
    model_info: Optional[Dict[str, Any]] = None
    startup_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Handles model loading on startup and cleanup on shutdown.
    """
    # Startup
    logger.info("Starting Churn Prediction API...")
    ModelState.startup_time = datetime.now()
    
    try:
        # Initialize model loader
        ModelState.model_loader = get_model_loader()
        
        # Load model
        logger.info("Loading ML model...")
        load_start = time.time()
        model = ModelState.model_loader.load_model() # noqa
        load_time = time.time() - load_start
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        MODEL_LOAD_TIME.set(time.time())
        
        # Get model info
        ModelState.model_info = ModelState.model_loader.get_model_info()
        logger.info(f"Model Info: {ModelState.model_info}")
        
        # Set health status to healthy
        HEALTH_STATUS.set(1)
        
        logger.info("API startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        HEALTH_STATUS.set(0)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Churn Prediction API...")
    HEALTH_STATUS.set(0)
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Production-ready API for customer churn prediction with MLOps best practices",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all requests and track metrics.
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} latency={latency:.3f}s"
    )
    
    # Track metrics
    API_REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    API_REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(latency)
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
        detail={"exception": str(exc)}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its dependencies"
)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks:
    - API status
    - Model loaded
    - MLflow connectivity
    - Redis connectivity (if using Feast)
    """
    services = {}
    overall_status = HealthStatus.HEALTHY
    
    # Check if model is loaded
    if ModelState.model_loader is None or ModelState.model_info is None:
        services["model"] = ServiceStatus(
            name="model",
            status=HealthStatus.UNHEALTHY,
            error="Model not loaded"
        )
        overall_status = HealthStatus.UNHEALTHY
    else:
        services["model"] = ServiceStatus(
            name="model",
            status=HealthStatus.HEALTHY
        )
    
    # Check MLflow connectivity
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        start_time = time.time()
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        # Simple check - try to list experiments
        client.search_experiments(max_results=1)
        latency = (time.time() - start_time) * 1000
        
        services["mlflow"] = ServiceStatus(
            name="mlflow",
            status=HealthStatus.HEALTHY,
            latency_ms=latency
        )
    except Exception as e:
        logger.warning(f"MLflow health check failed: {str(e)}")
        services["mlflow"] = ServiceStatus(
            name="mlflow",
            status=HealthStatus.DEGRADED,
            error=str(e)
        )
        if overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.DEGRADED
    
    # Check Redis (for Feast online store)
    try:
        import redis
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        start_time = time.time()
        r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=2)
        r.ping()
        latency = (time.time() - start_time) * 1000
        
        services["redis"] = ServiceStatus(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=latency
        )
    except Exception as e:
        logger.warning(f"Redis health check failed: {str(e)}")
        services["redis"] = ServiceStatus(
            name="redis",
            status=HealthStatus.DEGRADED,
            error=str(e)
        )
        # Redis is optional for now, so don't mark as unhealthy
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        services=services
    )


@app.get(
    "/ready",
    summary="Readiness check",
    description="Check if the service is ready to handle requests"
)
async def readiness_check():
    """
    Readiness check for Kubernetes/orchestration platforms.
    
    Returns 200 if ready, 503 if not ready.
    """
    if ModelState.model_loader is None or ModelState.model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: Model not loaded"
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat()
    }


@app.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Get model information",
    description="Retrieve information about the currently loaded model"
)
async def get_model_info():
    """
    Get information about the currently loaded model.
    """
    if ModelState.model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(**ModelState.model_info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make churn prediction",
    description="Predict customer churn probability based on customer features",
    status_code=status.HTTP_200_OK
)
async def predict(request: PredictionRequest):
    """
    Make a churn prediction for a single customer.
    
    Args:
        request: Prediction request with customer features
        
    Returns:
        Prediction response with churn probability and label
    """
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if ModelState.model_loader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Get customer_id (use provided or generate a temporary one)
        customer_id = request.customer_id or f"temp_{int(time.time())}"
        
        # Convert request to dictionary
        features_dict = request.features.model_dump()
        
        # Get features (from Feast online store if available, otherwise use provided)
        features_df = ModelState.model_loader.get_online_features(
            customer_id=customer_id,
            features_dict=features_dict
        )
        
        # Preprocess features
        processed_features = ModelState.model_loader.preprocess_features(features_df)
        
        # Make prediction
        model = ModelState.model_loader.get_model()
        prediction = model.predict(processed_features)[0]
        probability = model.predict_proba(processed_features)[0][1]
        
        # Create response
        prediction_label = "Churn" if prediction == 1 else "No Churn"
        
        response = PredictionResponse(
            customer_id=request.customer_id,
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=float(probability),
            confidence=float(max(probability, 1 - probability)),
            timestamp=datetime.now(),
            model_version=ModelState.model_info.get("version") if ModelState.model_info else None
        )
        
        # Track metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(prediction_label=prediction_label).inc()
        PREDICTION_PROBABILITY.observe(probability)
        
        logger.info(
            f"Prediction made: customer_id={request.customer_id}, "
            f"prediction={prediction_label}, probability={probability:.3f}, "
            f"latency={latency:.3f}s"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Make batch predictions",
    description="Predict churn for multiple customers in a single request",
    status_code=status.HTTP_200_OK
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple customers.
    
    Args:
        request: Batch prediction request with multiple customer features
        
    Returns:
        Batch prediction response with results for all customers
    """
    start_time = time.time()
    
    try:
        if ModelState.model_loader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        predictions = []
        successful = 0
        failed = 0
        
        for pred_request in request.predictions:
            try:
                # Make individual prediction
                pred_response = await predict(pred_request)
                predictions.append(pred_response)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to predict for customer {pred_request.customer_id}: {str(e)}")
                failed += 1
                # Continue with other predictions
        
        latency = time.time() - start_time
        
        logger.info(
            f"Batch prediction completed: total={len(request.predictions)}, "
            f"successful={successful}, failed={failed}, latency={latency:.3f}s"
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(request.predictions),
            successful_predictions=successful,
            failed_predictions=failed,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Expose Prometheus metrics for monitoring"
)
async def metrics():
    """
    Expose Prometheus metrics.
    
    Returns metrics in Prometheus text format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get(
    "/drift/check",
    summary="Check for feature drift",
    description="Compare current production data against training data to detect drift"
)
async def check_drift():
    """
    Check for feature drift between training and production data.
    """
    try:
        # Load training data (reference)
        from pathlib import Path

        import pandas as pd
        
        data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = pd.read_csv(data_path)
        
        # Handle TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        
        # Split: training (70%) vs production (30%)
        split_idx = int(len(df) * 0.7)
        training_data = df.iloc[:split_idx]
        production_data = df.iloc[split_idx:]
        
        # Initialize detector
        detector = DriftDetector(training_data, threshold=0.2)
        
        # Check drift
        results = detector.check_drift(production_data)
        
        # Convert report to list of dicts with proper types
        report_list = []
        for _, row in results['report'].iterrows():
            report_list.append({
                'feature': str(row['feature']),
                'psi': float(row['psi']),
                'status': str(row['status']),
                'drifted': bool(row['drifted']),
                'description': str(row['description'])
            })
        
        # Format response with type conversions
        return {
            "status": "success",
            "drift_check": {
                "overall_status": str(results['overall_status']),
                "drifted_features": int(results['drifted_features']),
                "total_features": int(results['total_features']),
                "max_psi": float(results['max_psi']),
                "mean_psi": float(results['mean_psi']),
                "threshold": 0.2
            },
            "features": report_list,
            "timestamp": datetime.now().isoformat()  # Use .isoformat() for datetime
        }
        
    except Exception as e:
        logger.error(f"Drift check error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )