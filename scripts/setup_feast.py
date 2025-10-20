"""
Setup Feast feature store for churn prediction.

This script:
1. Prepares historical feature data from raw CSV
2. Creates Parquet files for Feast offline store
3. Applies Feast feature definitions
4. Materializes features to online store (Redis)
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "features"
FEATURE_REPO_PATH = PROJECT_ROOT / "src" / "features" / "feature_repo"


def prepare_feature_data() -> pd.DataFrame:
    """
    Prepare feature data from raw CSV for Feast.
    
    Returns:
        DataFrame with features ready for Feast
    """
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {RAW_DATA_PATH}")
    
    # Load data
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df)} rows")
    
    # Rename customerID to customer_id (Feast convention)
    df = df.rename(columns={'customerID': 'customer_id'})
    
    # Handle TotalCharges - convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Add event_timestamp (required by Feast)
    # For historical data, we'll simulate timestamps going back in time
    # Most recent data = now, oldest data = 2 years ago
    base_time = datetime.now()
    time_deltas = np.linspace(0, 730, len(df))  # 730 days = 2 years
    df['event_timestamp'] = [base_time - timedelta(days=float(delta)) for delta in time_deltas]
    
    # Add created_timestamp (optional - when feature was computed)
    df['created_timestamp'] = datetime.now()
    
    # Remove Churn column (target, not a feature)
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])
    
    logger.info(f"Prepared {len(df)} feature records")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def save_to_parquet(df: pd.DataFrame):
    """
    Save feature data to Parquet format for Feast offline store.
    
    Args:
        df: DataFrame with features
    """
    # Create directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = PROCESSED_DIR / "customer_features.parquet"
    
    logger.info(f"Saving features to {output_path}")
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    # Verify saved file
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Saved {file_size:.2f} MB")


def apply_feast_definitions():
    """
    Apply Feast feature definitions.
    
    This registers the feature views with Feast.
    """
    import subprocess
    
    logger.info("Applying Feast feature definitions...")
    
    try:
        # Change to feature repo directory
        original_dir = Path.cwd()
        feature_repo_dir = PROJECT_ROOT / "src" / "features" / "feature_repo"
        
        logger.info(f"Changing to directory: {feature_repo_dir}")
        
        # Run feast apply
        result = subprocess.run(
            ["feast", "apply"],
            cwd=str(feature_repo_dir),
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Feast apply output:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning(f"Feast warnings: {result.stderr}")
        
        logger.info("✅ Feast feature definitions applied successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to apply Feast definitions: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error applying Feast: {str(e)}")
        raise


def materialize_features(start_date: datetime = None, end_date: datetime = None):
    """
    Materialize features from offline to online store.
    
    Args:
        start_date: Start date for materialization (default: 2 years ago)
        end_date: End date for materialization (default: now)
    """
    import subprocess
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=730)  # 2 years ago
    
    if end_date is None:
        end_date = datetime.now()
    
    logger.info(f"Materializing features from {start_date} to {end_date}")
    
    try:
        feature_repo_dir = PROJECT_ROOT / "src" / "features" / "feature_repo"
        
        # Run feast materialize
        result = subprocess.run(
            [
                "feast",
                "materialize",
                start_date.isoformat(),
                end_date.isoformat()
            ],
            cwd=str(feature_repo_dir),
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Feast materialize output:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning(f"Feast warnings: {result.stderr}")
        
        logger.info("✅ Features materialized to online store successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to materialize features: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        logger.warning("Note: Materialization requires Redis to be running")
        logger.warning("Skip materialization for now if Redis is not available")
    except Exception as e:
        logger.error(f"Error materializing features: {str(e)}")


def main():
    """Main function to setup Feast feature store."""
    logger.info("=" * 60)
    logger.info("Feast Feature Store Setup")
    logger.info("=" * 60)
    
    try:
        # Step 1: Prepare feature data
        logger.info("\n Step 1: Preparing feature data...")
        df = prepare_feature_data()
        
        # Step 2: Save to Parquet
        logger.info("\n Step 2: Saving to Parquet...")
        save_to_parquet(df)
        
        # Step 3: Apply Feast definitions
        logger.info("\n Step 3: Applying Feast feature definitions...")
        apply_feast_definitions()
        
        # Step 4: Materialize to online store (optional - requires Redis)
        logger.info("\n Step 4: Materializing features to online store...")
        logger.info("⚠️  This step requires Redis to be running")
        
        try:
            materialize_features()
        except Exception as e:
            logger.warning(f"Materialization failed (Redis may not be running): {str(e)}")
            logger.info("You can materialize later when Redis is available")
        
        # Success
        logger.info("\n" + "=" * 60)
        logger.info("✅ Feast setup completed successfully!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. Start Redis: docker-compose up -d redis")
        logger.info("  2. Materialize features: poetry run python scripts/setup_feast.py")
        logger.info("  3. Update training to use Feast offline store")
        logger.info("  4. Update API to use Feast online store")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())