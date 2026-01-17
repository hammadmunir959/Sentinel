"""
Sentinel Configuration Module.

Centralized configuration for paths, hyperparameters, and environment settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    raw_data_file: str = "creditcard.csv"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    
    # Feature columns (V1-V28 are PCA transformed, Amount and Time are raw)
    target_column: str = "Class"
    scale_columns: List[str] = field(default_factory=lambda: ["Amount", "Time"])
    
    @property
    def raw_data_path(self) -> Path:
        return DATA_RAW / self.raw_data_file


@dataclass
class ModelConfig:
    """Configuration for XGBoost model hyperparameters."""
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0
    reg_alpha: float = 0
    reg_lambda: float = 1
    
    # Class imbalance handling
    scale_pos_weight: float = None  # Will be computed from data
    
    # Training
    early_stopping_rounds: int = 10
    eval_metric: str = "aucpr"
    random_state: int = 42
    
    # Model artifacts
    model_filename: str = "model.json"
    
    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_filename
    
    def to_xgb_params(self) -> dict:
        """Convert to XGBoost parameter dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "early_stopping_rounds": self.early_stopping_rounds,
            "use_label_encoder": False,
        }


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    
    experiment_name: str = "sentinel-fraud-detection"
    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "")
    )
    
    # DagsHub credentials (for remote tracking)
    dagshub_username: str = field(
        default_factory=lambda: os.getenv("DAGSHUB_USERNAME", "")
    )
    dagshub_token: str = field(
        default_factory=lambda: os.getenv("DAGSHUB_TOKEN", "")
    )


@dataclass
class APIConfig:
    """Configuration for FastAPI service."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: Path = MODELS_DIR / "model.json"
    
    # Prediction logging
    log_predictions: bool = True
    predictions_log_file: Path = LOGS_DIR / "predictions.jsonl"


@dataclass
class MonitoringConfig:
    """Configuration for drift monitoring."""
    
    reference_data_path: Path = DATA_PROCESSED / "train.csv"
    drift_threshold: float = 0.2  # Alert if >20% features show drift
    report_output_dir: Path = REPORTS_DIR


# Default configurations
data_config = DataConfig()
model_config = ModelConfig()
mlflow_config = MLflowConfig()
api_config = APIConfig()
monitoring_config = MonitoringConfig()
