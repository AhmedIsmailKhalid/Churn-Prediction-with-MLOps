"""
Prepare Telco Churn data for Feast feature store.
Adds required timestamp column for Feast offline store.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_feast_data(
    input_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_path: str = "data/processed/features/customer_features.parquet"
):
    """
    Prepare customer churn data for Feast.
    
    Args:
        input_path: Path to raw CSV data
        output_path: Path to save processed Parquet file
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Add required timestamp column for Feast
    # Use tenure to simulate historical timestamps
    # Customers with tenure=1 month joined recently, tenure=72 joined 6 years ago
    base_date = datetime.now()
    df['event_timestamp'] = df['tenure'].apply(
        lambda x: base_date - timedelta(days=int(x) * 30)
    )
    
    # Rename customerID to match Feast entity name
    if 'customerID' in df.columns:
        df = df.rename(columns={'customerID': 'customer_id'})
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert SeniorCitizen to int if it's not already
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet for Feast
    logger.info(f"Saving processed data to {output_path}")
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Processed {len(df)} rows")
    logger.info(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    prepare_feast_data()