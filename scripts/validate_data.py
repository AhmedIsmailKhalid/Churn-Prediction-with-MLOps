"""
Validate the raw data quality.

This script performs comprehensive data validation checks.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def create_schema() -> DataFrameSchema:
    """
    Create Pandera schema for Telco Churn dataset.
    
    Returns:
        DataFrameSchema for validation
    """
    schema = DataFrameSchema(
        {
            "customerID": Column(str, nullable=False, unique=True),
            "gender": Column(str, checks=Check.isin(["Male", "Female"])),
            "SeniorCitizen": Column(int, checks=Check.isin([0, 1])),
            "Partner": Column(str, checks=Check.isin(["Yes", "No"])),
            "Dependents": Column(str, checks=Check.isin(["Yes", "No"])),
            "tenure": Column(int, checks=Check.greater_than_or_equal_to(0)),
            "PhoneService": Column(str, checks=Check.isin(["Yes", "No"])),
            "MultipleLines": Column(str, checks=Check.isin(["Yes", "No", "No phone service"])),
            "InternetService": Column(str, checks=Check.isin(["DSL", "Fiber optic", "No"])),
            "OnlineSecurity": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "OnlineBackup": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "DeviceProtection": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "TechSupport": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "StreamingTV": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "StreamingMovies": Column(str, checks=Check.isin(["Yes", "No", "No internet service"])),
            "Contract": Column(str, checks=Check.isin(["Month-to-month", "One year", "Two year"])),
            "PaperlessBilling": Column(str, checks=Check.isin(["Yes", "No"])),
            "PaymentMethod": Column(str),
            "MonthlyCharges": Column(float, checks=Check.greater_than(0)),
            "TotalCharges": Column(nullable=True),  # Can have missing values
            "Churn": Column(str, checks=Check.isin(["Yes", "No"])),
        },
        strict=False  # Allow extra columns
    )
    return schema


def validate_data(file_path: Path) -> bool:
    """
    Validate the dataset using Pandera schema.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Create schema
        schema = create_schema()
        
        # Validate
        logger.info("Running validation checks...")
        _ = schema.validate(df, lazy=True)
        
        logger.info("✅ All validation checks passed!")
        
        # Additional statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total records: {len(df):,}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        logger.info(f"  Duplicate customer IDs: {df['customerID'].duplicated().sum()}")
        logger.info("\nChurn Distribution:")
        logger.info(f"{df['Churn'].value_counts()}")
        logger.info(f"\nChurn Rate: {(df['Churn'] == 'Yes').mean():.2%}")
        
        return True
        
    except pa.errors.SchemaErrors as e:
        logger.error("❌ Validation failed!")
        logger.error(f"Schema errors:\n{e}")
        return False
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}")
        return False


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Data Validation")
    logger.info("=" * 60)
    
    # Check if file exists
    if not RAW_DATA_PATH.exists():
        logger.error(f"Dataset not found: {RAW_DATA_PATH}")
        logger.info("Run: poetry run python scripts/download_data.py")
        return 1
    
    # Validate
    if validate_data(RAW_DATA_PATH):
        logger.info("\n✅ Data validation successful!")
        return 0
    else:
        logger.error("\n❌ Data validation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())