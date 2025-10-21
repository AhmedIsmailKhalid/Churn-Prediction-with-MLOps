"""
Dependency injection for FastAPI application.

This module provides dependencies for model loading and management,
following FastAPI's dependency injection pattern.
"""

# ruff : noqa : F401
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from feast import FeatureStore
from feast.errors import FeatureViewNotFoundException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Model loader and manager for the API.
    
    Handles loading models from MLflow and preprocessing features for inference.
    """
    
    def __init__(
        self,
        mlflow_uri: str,
        model_name: str,
        model_stage: str = "Production",
        use_feast: bool = True
    ):
        """
        Initialize the model loader.
        
        Args:
            mlflow_uri: MLflow tracking URI
            model_name: Name of the registered model
            model_stage: Model stage to load (Production/Staging/etc)
            use_feast: Whether to use Feast for feature serving
        """
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_stage = model_stage
        self.use_feast = use_feast
        
        self._model = None
        self._model_info = None
        self._feature_columns = None
        
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)
        
        # Initialize Feast if enabled
        if self.use_feast:
            try:
                feast_repo_path = Path(__file__).parent.parent / "features" / "feature_repo"
                self.feast_store = FeatureStore(repo_path=str(feast_repo_path))
                logger.info(f"Initialized Feast feature store from {feast_repo_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize Feast: {str(e)}")
                logger.warning("Falling back to non-Feast feature serving")
                self.feast_store = None
                self.use_feast = False
        else:
            self.feast_store = None
        
        logger.info(f"ModelLoader initialized: {model_name} @ {model_stage}, Feast: {self.use_feast}")
    
    def load_model(self):
        """
        Load the model from MLflow registry.
        
        Returns:
            Loaded model
            
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name} ({self.model_stage})")
            
            # Get model version from registry
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            
            # Try to load as XGBoost first, then sklearn
            try:
                import mlflow.xgboost
                self._model = mlflow.xgboost.load_model(model_uri)
                logger.info("Loaded as XGBoost model")
            except Exception:
                import mlflow.sklearn
                self._model = mlflow.sklearn.load_model(model_uri)
                logger.info("Loaded as sklearn model")
            
            # Get the feature names from the model
            if hasattr(self._model, 'feature_names_in_'):
                self._feature_columns = list(self._model.feature_names_in_)
                logger.info(f"Loaded {len(self._feature_columns)} feature names from model")
            elif hasattr(self._model, 'get_booster'):
                # XGBoost specific
                self._feature_columns = self._model.get_booster().feature_names
                logger.info(f"Loaded {len(self._feature_columns)} feature names from XGBoost model")
            else:
                logger.warning("Could not extract feature names from model")
            
            # Get model metadata
            model_versions = self.client.search_model_versions(
                f"name='{self.model_name}'"
            )
            
            # Find the version in the specified stage
            current_version = None
            for mv in model_versions:
                if mv.current_stage == self.model_stage:
                    current_version = mv
                    break
            
            if current_version:
                # Get run info for metrics
                run = self.client.get_run(current_version.run_id)
                
                self._model_info = {
                    "name": self.model_name,
                    "version": current_version.version,
                    "stage": self.model_stage,
                    "loaded_at": datetime.now(),
                    "mlflow_run_id": current_version.run_id,
                    "metrics": run.data.metrics
                }
            else:
                logger.warning(f"No model version found in stage '{self.model_stage}'")
                self._model_info = {
                    "name": self.model_name,
                    "version": "unknown",
                    "stage": self.model_stage,
                    "loaded_at": datetime.now(),
                    "mlflow_run_id": None,
                    "metrics": None
                }
            
            logger.info(f"Model loaded successfully: {self._model_info}")
            if self._feature_columns:
                logger.info(f"Expected features: {self._feature_columns[:5]}... ({len(self._feature_columns)} total)")
            
            return self._model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_model(self):
        """
        Get the loaded model.
        
        Returns:
            Loaded model
            
        Raises:
            ValueError: If model not loaded
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        if self._model_info is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self._model_info
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for model inference.
        
        This should match the preprocessing done during training.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            Preprocessed features ready for prediction
        """
        # Create a copy
        df = df.copy()
        
        # Handle TotalCharges - convert to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
        
        # Convert categorical variables to dummy variables (same as training)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
        
        # If we have stored feature columns from training, align them
        if self._feature_columns is not None:
            # Add missing columns with 0
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Keep only the columns from training, in the same order
            df = df[self._feature_columns]
        
        return df
    
    def set_feature_columns(self, columns: list):
        """
        Set the expected feature columns (from training).
        
        Args:
            columns: List of feature column names
        """
        self._feature_columns = columns
        logger.info(f"Feature columns set: {len(columns)} columns")


    def get_online_features(self, customer_id: str, features_dict: dict) -> pd.DataFrame:
        """
        Get features from Feast online store.
        
        Args:
            customer_id: Customer ID
            features_dict: Dictionary with feature values (fallback if Feast unavailable)
            
        Returns:
            DataFrame with features
        """
        if not self.use_feast or self.feast_store is None:
            # Fallback: use provided features directly
            logger.debug("Using provided features (Feast not available)")
            return pd.DataFrame([features_dict])
        
        try:
            # Fetch from Feast online store
            feature_vector = self.feast_store.get_online_features(
                entity_rows=[{"customer_id": customer_id}],
                features=[
                    "customer_demographics:gender",
                    "customer_demographics:SeniorCitizen",
                    "customer_demographics:Partner",
                    "customer_demographics:Dependents",
                    "customer_account:tenure",
                    "customer_account:Contract",
                    "customer_account:PaperlessBilling",
                    "customer_account:PaymentMethod",
                    "customer_account:MonthlyCharges",
                    "customer_account:TotalCharges",
                    "customer_services:PhoneService",
                    "customer_services:MultipleLines",
                    "customer_services:InternetService",
                    "customer_services:OnlineSecurity",
                    "customer_services:OnlineBackup",
                    "customer_services:DeviceProtection",
                    "customer_services:TechSupport",
                    "customer_services:StreamingTV",
                    "customer_services:StreamingMovies",
                ]
            ).to_df()
            
            # Drop customer_id column
            feature_vector = feature_vector.drop(columns=['customer_id'])
            
            logger.debug(f"Retrieved {len(feature_vector.columns)} features from Feast online store")
            return feature_vector
            
        except Exception as e:
            logger.warning(f"Failed to get features from Feast online store: {str(e)}")
            logger.warning("Falling back to provided features")
            return pd.DataFrame([features_dict])
    
    
# Singleton instance
_model_loader_instance: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """
    Get or create the model loader instance.
    
    This implements a singleton pattern to ensure only one model
    is loaded in memory.
    
    Returns:
        ModelLoader instance
    """
    global _model_loader_instance
    
    if _model_loader_instance is None:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.getenv("MODEL_NAME", "churn_prediction")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        use_feast = os.getenv("USE_FEAST", "true").lower() == "true"
        
        _model_loader_instance = ModelLoader(
            mlflow_uri=mlflow_uri,
            model_name=model_name,
            model_stage=model_stage,
            use_feast=use_feast
        )
    
    return _model_loader_instance