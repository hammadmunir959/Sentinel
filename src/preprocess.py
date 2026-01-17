"""
Data Preprocessing Pipeline for Credit Card Fraud Detection.

This module handles:
- Loading raw data
- Feature scaling (Amount, Time)
- Stratified train/validation/test splits
- Saving processed datasets
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_RAW, DATA_PROCESSED, data_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw credit card fraud dataset.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame with raw data.
    """
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} transactions")
    return df


def analyze_class_distribution(df: pd.DataFrame, target_col: str) -> dict:
    """Analyze class distribution for imbalanced dataset.
    
    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        
    Returns:
        Dictionary with class distribution statistics.
    """
    class_counts = df[target_col].value_counts()
    total = len(df)
    
    stats = {
        "total_samples": total,
        "fraud_count": int(class_counts.get(1, 0)),
        "normal_count": int(class_counts.get(0, 0)),
        "fraud_ratio": float(class_counts.get(1, 0) / total),
        "imbalance_ratio": float(class_counts.get(0, 0) / class_counts.get(1, 1)),
    }
    
    logger.info(f"Class distribution:")
    logger.info(f"  Normal (0): {stats['normal_count']:,} ({100 - stats['fraud_ratio']*100:.2f}%)")
    logger.info(f"  Fraud (1): {stats['fraud_count']:,} ({stats['fraud_ratio']*100:.4f}%)")
    logger.info(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}:1")
    
    return stats


def scale_features(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale specified features using StandardScaler.
    
    Args:
        df: Input DataFrame.
        columns: List of column names to scale.
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    logger.info(f"Scaling features: {columns}")
    
    df = df.copy()
    scaler = StandardScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train/validation/test split.
    
    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    logger.info(f"Splitting data: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df[target_col],
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[target_col],
        random_state=random_state
    )
    
    logger.info(f"Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    
    # Verify stratification
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fraud_ratio = split_df[target_col].mean()
        logger.info(f"  {name} fraud ratio: {fraud_ratio*100:.4f}%")
    
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save processed datasets to CSV files.
    
    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        output_dir: Directory to save files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved processed data to {output_dir}")
    logger.info(f"  train.csv: {len(train_df):,} rows")
    logger.info(f"  val.csv: {len(val_df):,} rows")
    logger.info(f"  test.csv: {len(test_df):,} rows")


def main():
    """Run the preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Credit Card Fraud Detection Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Load raw data
    raw_data_path = DATA_RAW / data_config.raw_data_file
    df = load_raw_data(raw_data_path)
    
    # Analyze class distribution
    stats = analyze_class_distribution(df, data_config.target_column)
    
    # Scale features
    df_scaled, scaler = scale_features(df, data_config.scale_columns)
    
    # Stratified split
    train_df, val_df, test_df = stratified_split(
        df_scaled,
        target_col=data_config.target_column,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio,
        random_state=data_config.random_state
    )
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, DATA_PROCESSED)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Scale pos weight for XGBoost: {stats['imbalance_ratio']:.1f}")
    logger.info("=" * 60)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
