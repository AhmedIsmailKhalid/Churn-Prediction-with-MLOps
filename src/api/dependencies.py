"""
Dependency injection for FastAPI application.

This module provides dependencies for model loading and management,
following FastAPI's dependency injection pattern.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import joblib
import mlflow
import pandas as pd
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
        model_stage: str = "Production"
    ):
        """
        Initialize the model loader.
        
        Args:
            mlflow_uri: MLflow tracking URI
            model_name: Name of the registered model
            model_stage: Model stage to load (Production/Staging/etc)
        """
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.model_stage = model_stage
        
        self._model = None
        self._model_info = None
        self._feature_columns = None
        
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)
        
        logger.info(f"ModelLoader initialized: {model_name} @ {model_stage}")
    
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
            
            # Load model
            self._model = mlflow.sklearn.load_model(model_uri)
            
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
        
        # Convert categorical variables to dummy variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        
        # If we have stored feature columns from training, align them
        if self._feature_columns is not None:
            # Add missing columns with 0
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Remove extra columns and reorder
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
        
        _model_loader_instance = ModelLoader(
            mlflow_uri=mlflow_uri,
            model_name=model_name,
            model_stage=model_stage
        )
    
    return _model_loader_instance