"""
Custom Prometheus exporter for MLflow metrics.

Exposes MLflow experiment tracking and model registry metrics to Prometheus.
"""

# ruff : noqa

import logging
import os
import time
from typing import Dict

import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import CollectorRegistry, Gauge, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowExporter:
    """Prometheus exporter for MLflow metrics."""
    
    def __init__(self, mlflow_uri: str, port: int = 8001):
        """
        Initialize MLflow exporter.
        
        Args:
            mlflow_uri: MLflow tracking server URI
            port: Port to expose metrics on
        """
        self.mlflow_uri = mlflow_uri
        self.port = port
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.experiments_total = Gauge(
            'mlflow_experiments_total',
            'Total number of experiments',
            registry=self.registry
        )
        
        self.runs_total = Gauge(
            'mlflow_runs_total',
            'Total number of runs',
            ['status'],
            registry=self.registry
        )
        
        self.registered_models_total = Gauge(
            'mlflow_registered_models_total',
            'Total number of registered models',
            registry=self.registry
        )
        
        self.model_versions_total = Gauge(
            'mlflow_model_versions_total',
            'Total number of model versions',
            ['model_name', 'stage'],
            registry=self.registry
        )
        
        self.latest_run_timestamp = Gauge(
            'mlflow_latest_run_timestamp',
            'Timestamp of the latest run',
            ['experiment_id'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'mlflow_model_accuracy',
            'Model accuracy score',
            ['model_name', 'version', 'dataset'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'mlflow_model_precision',
            'Model precision score',
            ['model_name', 'version', 'dataset'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'mlflow_model_recall',
            'Model recall score',
            ['model_name', 'version', 'dataset'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'mlflow_model_f1_score',
            'Model F1 score',
            ['model_name', 'version', 'dataset'],
            registry=self.registry
        )
        
        self.model_roc_auc = Gauge(
            'mlflow_model_roc_auc',
            'Model ROC-AUC score',
            ['model_name', 'version', 'dataset'],
            registry=self.registry
        )
        
        logger.info(f"MLflow Exporter initialized for {mlflow_uri}")
    
    def collect_metrics(self) -> Dict[str, int]:
        """
        Collect metrics from MLflow.
        
        Returns:
            Dictionary of collected metrics
        """
        try:
            # Count experiments
            experiments = self.client.search_experiments()
            self.experiments_total.set(len(experiments))
            
            # Count runs by status
            all_runs = self.client.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
            
            status_counts = {'FINISHED': 0, 'RUNNING': 0, 'FAILED': 0, 'SCHEDULED': 0}
            for run in all_runs:
                status = run.info.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            for status, count in status_counts.items():
                self.runs_total.labels(status=status).set(count)
            
            # Get latest run timestamp per experiment
            for exp in experiments:
                exp_runs = self.client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"]
                )
                if exp_runs:
                    latest_timestamp = exp_runs[0].info.start_time / 1000  # Convert to seconds
                    self.latest_run_timestamp.labels(experiment_id=exp.experiment_id).set(latest_timestamp)
            
            # Count registered models (FIXED VERSION)
            try:
                # Try to get known models directly (workaround for search bug)
                known_model_names = ['churn_prediction'] 
                registered_models = []
                
                for model_name in known_model_names:
                    try:
                        model = self.client.get_registered_model(model_name)
                        registered_models.append(model)
                    except Exception:
                        pass  # Model doesn't exist
                
                self.registered_models_total.set(len(registered_models))
                
                # Count model versions by stage
                for model in registered_models:
                    # Use correct filter syntax
                    model_versions = self.client.search_model_versions(
                        filter_string=f"name='{model.name}'"
                    )
                    
                    # Count by stage
                    stage_counts = {}
                    for version in model_versions:
                        stage = version.current_stage
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    
                    for stage, count in stage_counts.items():
                        self.model_versions_total.labels(
                            model_name=model.name,
                            stage=stage
                        ).set(count)
                        
                # Collect model performance metrics
                for model in registered_models:
                    # Get Production model version
                    production_versions = [v for v in model_versions if v.current_stage == 'Production']
                    
                    if production_versions:
                        prod_version = production_versions[0]
                        
                        # Get run metrics
                        try:
                            run = self.client.get_run(prod_version.run_id)
                            metrics = run.data.metrics
                            
                            # Extract metrics for each dataset (train/val/test)
                            for dataset in ['train', 'val', 'test']:
                                # Accuracy
                                if f'{dataset}_accuracy' in metrics:
                                    self.model_accuracy.labels(
                                        model_name=model.name,
                                        version=prod_version.version,
                                        dataset=dataset
                                    ).set(metrics[f'{dataset}_accuracy'])
                                
                                # Precision
                                if f'{dataset}_precision' in metrics:
                                    self.model_precision.labels(
                                        model_name=model.name,
                                        version=prod_version.version,
                                        dataset=dataset
                                    ).set(metrics[f'{dataset}_precision'])
                                
                                # Recall
                                if f'{dataset}_recall' in metrics:
                                    self.model_recall.labels(
                                        model_name=model.name,
                                        version=prod_version.version,
                                        dataset=dataset
                                    ).set(metrics[f'{dataset}_recall'])
                                
                                # F1 Score
                                if f'{dataset}_f1_score' in metrics:
                                    self.model_f1_score.labels(
                                        model_name=model.name,
                                        version=prod_version.version,
                                        dataset=dataset
                                    ).set(metrics[f'{dataset}_f1_score'])
                                
                                # ROC-AUC
                                if f'{dataset}_roc_auc' in metrics:
                                    self.model_roc_auc.labels(
                                        model_name=model.name,
                                        version=prod_version.version,
                                        dataset=dataset
                                    ).set(metrics[f'{dataset}_roc_auc'])
                            
                            logger.info(f"Collected performance metrics for {model.name} v{prod_version.version}")
                        
                        except Exception as e:
                            logger.error(f"Error collecting performance metrics for {model.name}: {str(e)}")
                
                logger.info(
                    f"Collected metrics: {len(experiments)} experiments, "
                    f"{len(all_runs)} runs, {len(registered_models)} models"
                )
            except Exception as e:
                logger.error(f"Error collecting model metrics: {str(e)}")
                self.registered_models_total.set(0)
            
            return {
                'experiments': len(experiments),
                'runs': len(all_runs),
                'models': len(registered_models) if registered_models else 0
            }
            
        except Exception as e:
            logger.error(f"Error collecting MLflow metrics: {str(e)}")
            return {}
    
    def start(self, interval: int = 30):
        """
        Start the exporter.
        
        Args:
            interval: Metrics collection interval in seconds
        """
        # Start HTTP server
        start_http_server(self.port, registry=self.registry)
        logger.info(f"MLflow Exporter started on port {self.port}")
        
        # Collect metrics in loop
        while True:
            self.collect_metrics()
            time.sleep(interval)


def main():
    """Main entry point."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    port = int(os.getenv("MLFLOW_EXPORTER_PORT", "8001"))
    interval = int(os.getenv("MLFLOW_EXPORTER_INTERVAL", "30"))
    
    exporter = MLflowExporter(mlflow_uri=mlflow_uri, port=port)
    exporter.start(interval=interval)


if __name__ == "__main__":
    main()