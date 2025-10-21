"""
Feature drift detection using Population Stability Index (PSI).

PSI measures the shift in distribution of a feature between two datasets
(e.g., training data vs production data).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    bin_type: str = "quantile"
) -> float:
    """
    Calculate Population Stability Index (PSI) for a single feature.
    
    PSI Formula:
    PSI = Σ (actual% - expected%) * ln(actual% / expected%)
    
    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.2: Moderate change
    - PSI ≥ 0.2: Significant change (requires investigation)
    
    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (production data)
        bins: Number of bins for discretization
        bin_type: 'quantile' or 'uniform' binning
        
    Returns:
        PSI value
    """
    # Handle missing values
    expected = expected.dropna()
    actual = actual.dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        logger.warning("Empty series provided for PSI calculation")
        return 0.0
    
    # Create bins based on expected distribution
    if bin_type == "quantile":
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    else:  # uniform
        breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    
    # Ensure unique breakpoints
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        logger.warning("Not enough unique values for binning")
        return 0.0
    
    # Bin the data
    expected_binned = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates='drop')
    actual_binned = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates='drop')
    
    # Calculate distributions
    expected_counts = expected_binned.value_counts(sort=False)
    actual_counts = actual_binned.value_counts(sort=False)
    
    # Calculate percentages
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Align indices
    expected_percents = expected_percents.reindex(actual_percents.index, fill_value=0.0001)
    actual_percents = actual_percents.fillna(0.0001)
    
    # Replace zeros with small value to avoid log(0)
    expected_percents = expected_percents.replace(0, 0.0001)
    actual_percents = actual_percents.replace(0, 0.0001)
    
    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = psi_values.sum()
    
    return float(psi)


def calculate_psi_for_dataframe(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    features: List[str] = None,
    bins: int = 10
) -> Dict[str, float]:
    """
    Calculate PSI for multiple features in a DataFrame.
    
    Args:
        expected_df: Reference DataFrame (training data)
        actual_df: Current DataFrame (production data)
        features: List of features to check (if None, check all numeric)
        bins: Number of bins for discretization
        
    Returns:
        Dictionary mapping feature names to PSI values
    """
    if features is None:
        # Only numeric features
        features = expected_df.select_dtypes(include=[np.number]).columns.tolist()
    
    psi_results = {}
    
    for feature in features:
        if feature not in expected_df.columns or feature not in actual_df.columns:
            logger.warning(f"Feature {feature} not found in both DataFrames")
            continue
        
        try:
            psi = calculate_psi(expected_df[feature], actual_df[feature], bins=bins)
            psi_results[feature] = psi
            
            # Log warning for high drift
            if psi >= 0.2:
                logger.warning(f"Significant drift detected in {feature}: PSI={psi:.4f}")
            elif psi >= 0.1:
                logger.info(f"Moderate drift detected in {feature}: PSI={psi:.4f}")
                
        except Exception as e:
            logger.error(f"Error calculating PSI for {feature}: {str(e)}")
            psi_results[feature] = None
    
    return psi_results


def interpret_psi(psi: float) -> Tuple[str, str]:
    """
    Interpret PSI value.
    
    Args:
        psi: PSI value
        
    Returns:
        Tuple of (status, description)
    """
    if psi < 0.1:
        return "stable", "No significant distribution change"
    elif psi < 0.2:
        return "moderate", "Moderate distribution change - monitor closely"
    else:
        return "unstable", "Significant distribution change - investigate and consider retraining"


def detect_drift_report(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    features: List[str] = None,
    threshold: float = 0.2
) -> pd.DataFrame:
    """
    Generate a drift detection report.
    
    Args:
        expected_df: Reference DataFrame (training data)
        actual_df: Current DataFrame (production data)
        features: List of features to check
        threshold: PSI threshold for flagging drift
        
    Returns:
        DataFrame with drift report
    """
    psi_results = calculate_psi_for_dataframe(expected_df, actual_df, features)
    
    report_data = []
    for feature, psi in psi_results.items():
        if psi is None:
            continue
            
        status, description = interpret_psi(psi)
        drifted = psi >= threshold
        
        report_data.append({
            'feature': feature,
            'psi': psi,
            'status': status,
            'drifted': drifted,
            'description': description
        })
    
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values('psi', ascending=False)
    
    return report_df


class DriftDetector:
    """
    Drift detector for monitoring feature distributions over time.
    """
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.2):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Reference DataFrame (training data)
            threshold: PSI threshold for flagging drift
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.numeric_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"DriftDetector initialized with {len(self.numeric_features)} numeric features")
    
    def check_drift(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """
        Check for drift in current data.
        
        Args:
            current_data: Current DataFrame to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Checking for feature drift...")
        
        # Calculate PSI
        psi_results = calculate_psi_for_dataframe(
            self.reference_data,
            current_data,
            features=self.numeric_features
        )
        
        # Generate report
        report = detect_drift_report(
            self.reference_data,
            current_data,
            features=self.numeric_features,
            threshold=self.threshold
        )
        
        # Count drifted features
        drifted_count = report['drifted'].sum()
        total_features = len(report)
        
        # Overall status
        if drifted_count == 0:
            overall_status = "stable"
        elif drifted_count / total_features < 0.3:
            overall_status = "moderate"
        else:
            overall_status = "unstable"
        
        results = {
            'overall_status': overall_status,
            'drifted_features': drifted_count,
            'total_features': total_features,
            'psi_scores': psi_results,
            'report': report,
            'max_psi': report['psi'].max() if len(report) > 0 else 0.0,
            'mean_psi': report['psi'].mean() if len(report) > 0 else 0.0
        }
        
        logger.info(f"Drift check complete: {drifted_count}/{total_features} features drifted")
        logger.info(f"Overall status: {overall_status}")
        
        return results