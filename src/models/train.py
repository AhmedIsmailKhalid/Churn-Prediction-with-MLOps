"""
XGBoost model training for churn prediction with better imbalance handling.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage  # noqa
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """XGBoost trainer for customer churn prediction with class balancing."""
    
    def __init__(
        self,
        data_path: str,
        model_name: str = "churn_prediction",
        experiment_name: str = "churn_prediction_experiments",
        random_state: int = 42,
        use_feast: bool = True
    ):
        """Initialize the XGBoost trainer."""
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.use_feast = use_feast
        
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # Initialize Feast if enabled
        if self.use_feast:
            feast_repo_path = Path(__file__).parent.parent / "features" / "feature_repo"
            self.feast_store = FeatureStore(repo_path=str(feast_repo_path))
            logger.info(f"Initialized Feast feature store from {feast_repo_path}")
        else:
            self.feast_store = None
        
        logger.info(f"Initialized XGBoost Trainer with MLflow URI: {self.mlflow_uri}")
    
    def load_data(self) -> pd.DataFrame:
        """Load training data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        if df.empty:
            raise ValueError("Loaded data is empty")
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data for training."""
        logger.info("Starting data preprocessing")
        df = df.copy()
        
        # Handle TotalCharges - convert to numeric (it might have spaces causing string type)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)  # Fill NaN with 0
        
        # Drop rows with missing target
        df = df.dropna(subset=['Churn'])
        
        if 'Churn' not in df.columns:
            raise ValueError("Target column 'Churn' not found in data")
        
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        columns_to_drop = ['Churn', 'customerID'] if 'customerID' in df.columns else ['Churn']
        X = df.drop(columns=columns_to_drop)
        
        # Convert categorical variables to dummy variables (ONLY object types)
        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        logger.info(f"Number of features: {X.shape[1]}")
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data into train/val/test sets")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_features_from_feast(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features from Feast offline store.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Loading features from Feast offline store...")
        
        # Load raw data to get customer IDs and target
        df_raw = pd.read_csv(self.data_path)
        df_raw = df_raw.rename(columns={'customerID': 'customer_id'})
        
        # Get target variable
        y_raw = df_raw[['customer_id', 'Churn']].copy()
        y_raw['Churn'] = y_raw['Churn'].map({'Yes': 1, 'No': 0})
        
        # Create entity dataframe for Feast
        entity_df = pd.DataFrame({
            'customer_id': df_raw['customer_id'],
            'event_timestamp': pd.to_datetime('now')
        })
        
        # Get features from Feast
        feature_vector = self.feast_store.get_historical_features(
            entity_df=entity_df,
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
        
        # Merge features with target to ensure alignment
        feature_vector = feature_vector.merge(y_raw, on='customer_id', how='inner')
        
        # Separate features and target
        y = feature_vector['Churn']
        X = feature_vector.drop(columns=['event_timestamp', 'customer_id', 'Churn'])
        
        logger.info(f"Loaded {len(X)} records with {len(X.columns)} features from Feast")
        logger.info(f"Features and labels aligned: X shape={X.shape}, y shape={y.shape}")
        
        return X, y
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> xgb.XGBClassifier:
        """Train an XGBoost model with class balancing."""
        logger.info("Training XGBoost model with class balancing")
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        logger.info(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Default hyperparameters
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,  # Handle imbalance
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if hyperparameters:
            default_params.update(hyperparameters)
        
        logger.info(f"Model hyperparameters: {default_params}")
        
        # Train model
        model = xgb.XGBClassifier(**default_params)
        
        # Use early stopping with validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info("Model training completed")
        return model
    
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info(f"Evaluating model on {dataset_name} set")
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        logger.info(f"{dataset_name.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        cm = confusion_matrix(y, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{classification_report(y, y_pred)}")
        
        return metrics
    
    def preprocess_feast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features loaded from Feast.
        
        Args:
            df: DataFrame with features from Feast
            
        Returns:
            Preprocessed features
        """
        logger.info("Preprocessing Feast features")
        df = df.copy()
        
        # Handle TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
        
        # One-hot encode categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Convert boolean to int
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
        
        logger.info(f"Preprocessed features shape: {df.shape}")
        return df
    
    def run_training_pipeline(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        register_model: bool = True
    ) -> Dict[str, Any]:
        """Run the complete training pipeline with MLflow tracking."""
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            try:
                mlflow.log_param("data_path", str(self.data_path))
                mlflow.log_param("model_type", "XGBoost")
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("use_feast", self.use_feast)
                
                df = self.load_data()
                mlflow.log_param("total_samples", len(df))
                
                # Load features - use Feast if enabled
                if self.use_feast:
                    X, y = self.load_features_from_feast()
                    # Preprocess Feast features
                    X = self.preprocess_feast_features(X)
                else:
                    df = self.load_data()
                    X, y = self.preprocess_data(df)
                    
                mlflow.log_param("n_features", X.shape[1])
                
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("test_samples", len(X_test))
                
                if hyperparameters:
                    for key, value in hyperparameters.items():
                        mlflow.log_param(f"model_{key}", value)
                
                model = self.train_model(X_train, y_train, X_val, y_val, hyperparameters)
                
                train_metrics = self.evaluate_model(model, X_train, y_train, "train")
                val_metrics = self.evaluate_model(model, X_val, y_val, "validation")
                test_metrics = self.evaluate_model(model, X_test, y_test, "test")
                
                for metric_name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", value)
                
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)
                
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                
                # Log feature importance
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                
                # Log model
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    registered_model_name=self.model_name if register_model else None
                )
                
                logger.info(f"Model logged successfully. Run ID: {run_id}")
                
                return {
                    'model': model,
                    'run_id': run_id,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': importance_df,
                    'feature_names': list(X.columns)
                }
                
            except Exception as e:
                logger.error(f"Training pipeline failed: {str(e)}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise


def main():
    """Main function to run XGBoost training pipeline."""
    data_path = os.getenv("RAW_DATA_PATH", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Initialize trainer
    trainer = ChurnModelTrainer(
        data_path=data_path,
        model_name="churn_prediction",
        experiment_name="churn_prediction_experiments"
    )
    
    results = trainer.run_training_pipeline(
        hyperparameters={
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        register_model=True
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    logger.info(f"Test Precision: {results['test_metrics']['precision']:.4f}")
    logger.info(f"Test Recall: {results['test_metrics']['recall']:.4f}")
    logger.info(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")
    logger.info(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()