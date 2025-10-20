 
"""
Model training module for churn prediction.

This module handles the complete training pipeline including data loading,
preprocessing, model training, and MLflow experiment tracking.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Trainer class for customer churn prediction model.
    
    This class encapsulates the entire training workflow including data loading,
    preprocessing, model training, evaluation, and MLflow tracking.
    """
    
    def __init__(
        self,
        data_path: str,
        model_name: str = "churn_prediction",
        experiment_name: str = "churn_prediction_experiments",
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to the training data CSV file
            model_name: Name for the model in MLflow registry
            experiment_name: Name for the MLflow experiment
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.random_state = random_state
        
        # Set MLflow tracking URI from environment or default
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Set or create experiment
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"Initialized ChurnModelTrainer with MLflow URI: {self.mlflow_uri}")
        logger.info(f"Experiment: {self.experiment_name}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load training data from CSV file.
        
        Returns:
            DataFrame containing the training data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data file is empty or malformed
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            
            if df.empty:
                raise ValueError("Loaded data is empty")
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for training.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Starting data preprocessing")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        # Assuming 'Churn' is the target column
        if 'Churn' not in df.columns:
            raise ValueError("Target column 'Churn' not found in data")
        
        # Convert target to binary (Yes=1, No=0)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Drop target and customerID from features
        columns_to_drop = ['Churn', 'customerID'] if 'customerID' in df.columns else ['Churn']
        X = df.drop(columns=columns_to_drop)
        
        # Convert categorical variables to dummy variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/val/test sets")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameters: Model hyperparameters (optional)
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest model")
        
        # Default hyperparameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Update with provided hyperparameters
        if hyperparameters:
            default_params.update(hyperparameters)
        
        logger.info(f"Model hyperparameters: {default_params}")
        
        # Train model
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return model
    
    def evaluate_model(
        self,
        model: RandomForestClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set")
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        # Log metrics
        logger.info(f"{dataset_name.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Classification report
        logger.info(f"Classification Report:\n{classification_report(y, y_pred)}")
        
        return metrics
    
    def get_feature_importance(
        self,
        model: RandomForestClassifier,
        feature_names: list
    ) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances sorted by importance
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def run_training_pipeline(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        register_model: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline with MLflow tracking.
        
        Args:
            hyperparameters: Model hyperparameters (optional)
            register_model: Whether to register the model in MLflow
            
        Returns:
            Dictionary containing model, metrics, and run information
        """
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            try:
                # Log parameters
                mlflow.log_param("data_path", str(self.data_path))
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
                # Load data
                df = self.load_data()
                mlflow.log_param("total_samples", len(df))
                
                # Preprocess
                X, y = self.preprocess_data(df)
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("feature_names", list(X.columns))
                
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("test_samples", len(X_test))
                
                # Train model
                if hyperparameters:
                    for key, value in hyperparameters.items():
                        mlflow.log_param(f"model_{key}", value)
                
                model = self.train_model(X_train, y_train, hyperparameters)
                
                # Evaluate on all sets
                train_metrics = self.evaluate_model(model, X_train, y_train, "train")
                val_metrics = self.evaluate_model(model, X_val, y_val, "validation")
                test_metrics = self.evaluate_model(model, X_test, y_test, "test")
                
                # Log metrics to MLflow
                for metric_name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", value)
                
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)
                
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                
                # Feature importance
                feature_importance = self.get_feature_importance(model, list(X.columns))
                
                # Log feature importance as artifact
                importance_path = "feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=self.model_name if register_model else None
                )
                
                logger.info(f"Model logged successfully. Run ID: {run_id}")
                
                # Return results
                results = {
                    'model': model,
                    'run_id': run_id,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': feature_importance,
                    'feature_names': list(X.columns)
                }
                
                return results
                
            except Exception as e:
                logger.error(f"Training pipeline failed: {str(e)}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise


def main():
    """Main function to run training pipeline."""
    # Configuration
    data_path = os.getenv("RAW_DATA_PATH", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Initialize trainer
    trainer = ChurnModelTrainer(
        data_path=data_path,
        model_name="churn_prediction",
        experiment_name="churn_prediction_experiments"
    )
    
    # Run training
    results = trainer.run_training_pipeline(
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4
        },
        register_model=True
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    logger.info(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()