"""
Prefect workflow for model training pipeline.

This flow orchestrates:
1. Data loading and validation
2. Feature engineering
3. Model training with MLflow
4. Model evaluation
5. Model registration
"""

# ruff : noqa : E402


from dotenv import load_dotenv

load_dotenv()

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import flow, task

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train import ChurnModelTrainer

logger = logging.getLogger(__name__)


@task(name="validate_data", retries=2, retry_delay_seconds=10)
def validate_data_task(data_path: str) -> Dict[str, Any]:
    """
    Validate input data before training.
    
    Args:
        data_path: Path to raw data CSV
        
    Returns:
        Validation results
    """
    import pandas as pd
    
    logger.info(f"Validating data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Basic validation checks
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty data
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    # Check for target column
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found")
    
    # Check target values
    valid_churn_values = {'Yes', 'No'}
    actual_churn_values = set(df['Churn'].dropna().unique())
    if not actual_churn_values.issubset(valid_churn_values):
        raise ValueError(f"Invalid churn values: {actual_churn_values - valid_churn_values}")
    
    logger.info("Data validation passed")
    
    return {
        "is_valid": True,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "validation_report": "All checks passed"
    }


@task(name="train_model", retries=1, retry_delay_seconds=30)
def train_model_task(
    data_path: str,
    model_name: str,
    experiment_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    use_feast: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost model with MLflow tracking.
    
    Args:
        data_path: Path to raw data
        model_name: Model name for MLflow registry
        experiment_name: MLflow experiment name
        hyperparameters: Model hyperparameters
        use_feast: Whether to use Feast for features
        
    Returns:
        Training results
    """
    logger.info(f"Starting model training: {model_name}")
    
    # Initialize trainer
    trainer = ChurnModelTrainer(
        data_path=data_path,
        model_name=model_name,
        experiment_name=experiment_name,
        use_feast=use_feast
    )
    
    # Run training pipeline
    results = trainer.run_training_pipeline(
        hyperparameters=hyperparameters,
        register_model=True
    )
    
    logger.info(f"Training completed: Run ID {results['run_id']}")
    
    return {
        "run_id": results['run_id'],
        "model_name": model_name,
        "test_accuracy": results['test_metrics']['accuracy'],
        "test_roc_auc": results['test_metrics']['roc_auc'],
        "test_f1_score": results['test_metrics']['f1_score'],
        "num_features": len(results['feature_names'])
    }


@task(name="promote_model", retries=2, retry_delay_seconds=10)
def promote_model_task(
    model_name: str,
    version: Optional[int] = None,
    stage: str = "Production"
) -> Dict[str, Any]:
    """
    Promote model to specified stage.
    
    Args:
        model_name: Model name in registry
        version: Model version (if None, promotes latest)
        stage: Target stage (Production/Staging)
        
    Returns:
        Promotion results
    """
    import os

    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    client = MlflowClient()
    
    logger.info(f"Promoting model {model_name} to {stage}")
    
    # Get latest version if not specified
    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        # Get latest version number
        version = max([int(v.version) for v in versions])
    
    # Transition to target stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    logger.info(f"Model {model_name} version {version} promoted to {stage}")
    
    return {
        "model_name": model_name,
        "version": version,
        "stage": stage
    }


@task(name="check_drift")
def check_drift_task(reference_data_path: str, current_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check for feature drift.
    
    Args:
        reference_data_path: Path to reference (training) data
        current_data_path: Path to current data (optional)
        
    Returns:
        Drift detection results
    """
    import pandas as pd

    from src.monitoring.drift_detection import DriftDetector
    
    logger.info("Checking for feature drift")
    
    # Load reference data
    df_ref = pd.read_csv(reference_data_path)
    df_ref['TotalCharges'] = pd.to_numeric(df_ref['TotalCharges'], errors='coerce')
    df_ref['TotalCharges'].fillna(0, inplace=True)
    
    # If no current data provided, use last 30% of reference as simulation
    if current_data_path is None:
        split_idx = int(len(df_ref) * 0.7)
        df_train = df_ref.iloc[:split_idx]
        df_current = df_ref.iloc[split_idx:]
    else:
        df_train = df_ref
        df_current = pd.read_csv(current_data_path)
        df_current['TotalCharges'] = pd.to_numeric(df_current['TotalCharges'], errors='coerce')
        df_current['TotalCharges'].fillna(0, inplace=True)
    
    # Check drift
    detector = DriftDetector(df_train, threshold=0.2)
    results = detector.check_drift(df_current)
    
    logger.info(f"Drift check: {results['overall_status']}, {results['drifted_features']}/{results['total_features']} features drifted")
    
    return {
        "overall_status": results['overall_status'],
        "drifted_features": results['drifted_features'],
        "total_features": results['total_features'],
        "max_psi": results['max_psi'],
        "mean_psi": results['mean_psi']
    }


@flow(
    name="churn-prediction-training",
    description="End-to-end training pipeline for churn prediction model",
    log_prints=True
)
def training_flow(
    data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    model_name: str = "churn_prediction",
    experiment_name: str = "churn_prediction_experiments",
    hyperparameters: Optional[Dict[str, Any]] = None,
    use_feast: bool = True,
    auto_promote: bool = True,
    check_drift: bool = True
) -> Dict[str, Any]:
    """
    Main training flow for churn prediction.
    
    Args:
        data_path: Path to training data
        model_name: Model name for registry
        experiment_name: MLflow experiment name
        hyperparameters: Model hyperparameters
        use_feast: Use Feast for features
        auto_promote: Automatically promote to Production
        check_drift: Check for feature drift
        
    Returns:
        Flow execution results
    """
    print("=" * 80)
    print("FLOW START - DEBUG INFO")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data path parameter: {data_path}")
    print(f"Data path exists: {Path(data_path).exists()}")
    print(f"Absolute data path: {Path(data_path).absolute()}")
    print(f"Environment variables: MLFLOW_TRACKING_URI={os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"Environment variables: USE_FEAST={os.getenv('USE_FEAST')}")
    print("=" * 80)
    
    
    logger.info("=" * 60)
    logger.info("Starting Churn Prediction Training Flow")
    logger.info("=" * 60)
    
    # Step 1: Validate data
    validation_results = validate_data_task(data_path)
    logger.info(f"Data validation: {validation_results['num_rows']} rows validated")
    
    # Step 2: Check drift (optional)
    drift_results = None
    if check_drift:
        drift_results = check_drift_task(data_path)
        logger.info(f"Drift status: {drift_results['overall_status']}")
    
    # Step 3: Train model
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    training_results = train_model_task(
        data_path=data_path,
        model_name=model_name,
        experiment_name=experiment_name,
        hyperparameters=hyperparameters,
        use_feast=use_feast
    )
    
    logger.info(f"Training complete: Accuracy={training_results['test_accuracy']:.4f}, ROC-AUC={training_results['test_roc_auc']:.4f}")
    
    # Step 4: Promote model (optional)
    promotion_results = None
    if auto_promote:
        promotion_results = promote_model_task(
            model_name=model_name,
            stage="Production"
        )
        logger.info(f"Model promoted to {promotion_results['stage']}")
    
    # Compile results
    results = {
        "status": "success",
        "validation": validation_results,
        "drift": drift_results,
        "training": training_results,
        "promotion": promotion_results
    }
    
    logger.info("=" * 60)
    logger.info("Training Flow Complete!")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run flow locally
    results = training_flow()
    print("\n" + "=" * 60)
    print("FLOW EXECUTION RESULTS")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Test Accuracy: {results['training']['test_accuracy']:.4f}")
    print(f"Test ROC-AUC: {results['training']['test_roc_auc']:.4f}")
    print(f"MLflow Run ID: {results['training']['run_id']}")