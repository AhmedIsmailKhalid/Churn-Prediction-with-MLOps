 
"""
Download Telco Customer Churn dataset from Kaggle.

This script downloads the dataset and performs initial validation.
You'll need Kaggle API credentials configured.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Alternative: Direct download URL (IBM dataset hosted publicly)
DIRECT_DOWNLOAD_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)
        
        logger.info(f"Downloaded successfully to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False


def download_from_kaggle(dataset_name: str, destination_dir: Path) -> bool:
    """
    Download dataset from Kaggle using Kaggle API.
    
    Args:
        dataset_name: Kaggle dataset identifier
        destination_dir: Directory to save the dataset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import kaggle
        
        logger.info(f"Downloading from Kaggle: {dataset_name}")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=destination_dir,
            unzip=True
        )
        
        logger.info("Kaggle download successful")
        return True
        
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Kaggle download failed: {str(e)}")
        logger.info("Make sure you have Kaggle API credentials configured")
        logger.info("See: https://www.kaggle.com/docs/api")
        return False


def validate_dataset(file_path: Path) -> bool:
    """
    Validate the downloaded dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        logger.info("Validating dataset...")
        
        # Read the dataset
        df = pd.read_csv(file_path)
        
        # Expected columns
        expected_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        
        # Check columns
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        
        # Check data size
        if len(df) < 1000:
            logger.error(f"Dataset too small: {len(df)} rows")
            return False
        
        # Check for target column
        if 'Churn' not in df.columns:
            logger.error("Target column 'Churn' not found")
            return False
        
        # Log statistics
        logger.info("Dataset validation successful!")
        logger.info(f"  Rows: {len(df):,}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info("  Churn distribution:")
        logger.info(f"    {df['Churn'].value_counts().to_dict()}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False


def main():
    """Main function to download the dataset."""
    logger.info("=" * 60)
    logger.info("Telco Customer Churn Dataset Downloader")
    logger.info("=" * 60)
    
    # Ensure data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {RAW_DATA_DIR}")
    
    # Check if dataset already exists
    dataset_path = RAW_DATA_DIR / DATASET_FILENAME
    if dataset_path.exists():
        logger.info(f"Dataset already exists: {dataset_path}")
        
        # Validate existing dataset
        if validate_dataset(dataset_path):
            logger.info("✅ Existing dataset is valid!")
            return 0
        else:
            logger.warning("Existing dataset is invalid. Re-downloading...")
            dataset_path.unlink()
    
    # Try multiple download methods
    download_success = False
    
    # Method 1: Direct download from IBM GitHub
    logger.info("\n Method 1: Downloading from IBM GitHub repository...")
    if download_file(DIRECT_DOWNLOAD_URL, dataset_path):
        if validate_dataset(dataset_path):
            download_success = True
            logger.info("✅ Download and validation successful!")
        else:
            logger.error("Downloaded file is invalid")
            dataset_path.unlink()
    
    # Method 2: Kaggle API (if Method 1 fails)
    if not download_success:
        logger.info("\n Method 2: Trying Kaggle API...")
        logger.info("Note: Requires Kaggle API credentials")
        logger.info("Setup: https://www.kaggle.com/docs/api")
        
        if download_from_kaggle("blastchar/telco-customer-churn", RAW_DATA_DIR):
            # Kaggle might save with different name, try to find it
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            if csv_files:
                source_file = csv_files[0]
                if source_file.name != DATASET_FILENAME:
                    source_file.rename(dataset_path)
                
                if validate_dataset(dataset_path):
                    download_success = True
                    logger.info("✅ Download and validation successful!")
    
    # Method 3: Manual download instructions
    if not download_success:
        logger.error("\n❌ Automatic download failed!")
        logger.info("\n Manual Download Instructions:")
        logger.info("=" * 60)
        logger.info("1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        logger.info("2. Download the dataset CSV file")
        logger.info(f"3. Save it as: {dataset_path}")
        logger.info("4. Run this script again to validate")
        logger.info("=" * 60)
        return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset ready for training!")
    logger.info(f" Location: {dataset_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())