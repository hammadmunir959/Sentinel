"""
Lightweight Training Script for CI/CD and Local Testing.

This script trains a model on a subsampled dataset for quick validation.
For full training, use the Colab notebook.
"""

import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, model_config, mlflow_config
from evaluate import compute_metrics, generate_full_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path, sample_frac: float = None) -> tuple:
    """Load train/val/test data.
    
    Args:
        data_dir: Directory containing processed data.
        sample_frac: Optional fraction to sample (for quick testing).
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    if sample_frac and sample_frac < 1.0:
        logger.info(f"Sampling {sample_frac*100:.0f}% of data for quick training")
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        val_df = val_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
    
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame, target_col: str = "Class") -> tuple:
    """Prepare features and target from DataFrame.
    
    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        
    Returns:
        Tuple of (X, y).
    """
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict = None
) -> xgb.XGBClassifier:
    """Train XGBoost classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: Model configuration (uses default if None).
        
    Returns:
        Trained XGBoost model.
    """
    if config is None:
        # Calculate scale_pos_weight from data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model_config.scale_pos_weight = scale_pos_weight
        config = model_config.to_xgb_params()
    
    logger.info("Training XGBoost model...")
    logger.info(f"  Scale pos weight: {config.get('scale_pos_weight', 'N/A'):.1f}")
    
    model = xgb.XGBClassifier(**config)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    return model


def main(sample_frac: float = None, use_mlflow: bool = True):
    """Run training pipeline.
    
    Args:
        sample_frac: Optional fraction to sample data (for quick CI runs).
        use_mlflow: Whether to log to MLflow.
    """
    logger.info("=" * 60)
    logger.info("Starting Sentinel Training Pipeline")
    logger.info("=" * 60)
    
    # Load data
    train_df, val_df, test_df = load_data(DATA_PROCESSED, sample_frac)
    
    logger.info(f"Train: {len(train_df):,} rows")
    logger.info(f"Val: {len(val_df):,} rows")
    logger.info(f"Test: {len(test_df):,} rows")
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    feature_names = list(X_train.columns)
    
    # Setup MLflow (optional)
    if use_mlflow and mlflow_config.tracking_uri:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name)
        logger.info(f"MLflow tracking: {mlflow_config.tracking_uri}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Generate report
    logger.info("Generating evaluation report...")
    metrics = generate_full_report(
        y_test, y_test_pred, y_test_prob,
        feature_names=feature_names,
        feature_importances=model.feature_importances_,
        output_dir=REPORTS_DIR
    )
    
    logger.info("Test Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Log to MLflow
    if use_mlflow and mlflow_config.tracking_uri:
        with mlflow.start_run(run_name="local_training"):
            # Log params
            mlflow.log_params(model_config.to_xgb_params())
            mlflow.log_param("sample_frac", sample_frac or 1.0)
            
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)
            
            # Log artifacts
            for artifact in REPORTS_DIR.glob("*"):
                mlflow.log_artifact(str(artifact))
            mlflow.log_artifact(str(model_path))
            
            logger.info("Logged to MLflow")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--sample", type=float, default=None,
        help="Fraction of data to sample (e.g., 0.1 for 10%%)"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging"
    )
    
    args = parser.parse_args()
    
    main(sample_frac=args.sample, use_mlflow=not args.no_mlflow)
