"""
Deploy Prefect flows for scheduled execution.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

project_root = Path(__file__).parent.parent

from src.pipelines.training import training_flow # noqa


def deploy_training_flow():
    """Deploy the training flow with a schedule."""
    
    # Create deployment
    training_flow.serve(
        name="churn-training-scheduled",
        cron="0 2 * * 0",  # Every Sunday at 2 AM
        parameters={
            "data_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "model_name": "churn_prediction",
            "use_feast": True,
            "auto_promote": True,
            "check_drift": True
        },
        tags=["training", "production", "scheduled"],
        description="Weekly automated retraining of churn prediction model"
    )

if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    print("Deploying Prefect training flow...")
    training_flow.serve(
        name="churn-training-scheduled",
        cron="0 2 * * *",
        tags=["training", "churn-prediction"],
        parameters={
            "data_path": str(project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        }
    )