"""
Test drift detection on training vs recent data.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.drift_detection import DriftDetector, detect_drift_report  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test drift detection."""
    logger.info("=" * 60)
    logger.info("Drift Detection Test")
    logger.info("=" * 60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Split into "training" (first 70%) and "production" (last 30%)
    split_idx = int(len(df) * 0.7)
    training_data = df.iloc[:split_idx].copy()
    production_data = df.iloc[split_idx:].copy()
    
    logger.info(f"Training data: {len(training_data)} samples")
    logger.info(f"Production data: {len(production_data)} samples")
    
    # Initialize drift detector
    detector = DriftDetector(training_data, threshold=0.2)
    
    # Check drift
    results = detector.check_drift(production_data)
    
    # Print results
    print("\n" + "=" * 60)
    print("DRIFT DETECTION RESULTS")
    print("=" * 60)
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Drifted Features: {results['drifted_features']}/{results['total_features']}")
    print(f"Max PSI: {results['max_psi']:.4f}")
    print(f"Mean PSI: {results['mean_psi']:.4f}")
    print("\nDetailed Report:")
    print(results['report'].to_string(index=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())