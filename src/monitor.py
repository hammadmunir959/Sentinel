"""
Drift Monitoring Module using Evidently AI.

This module detects data drift between reference (training) data
and production (incoming) data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor for detecting data drift."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: list,
        drift_threshold: float = 0.2
    ):
        """Initialize drift monitor.
        
        Args:
            reference_data: Training/reference dataset.
            feature_columns: List of feature column names.
            drift_threshold: Threshold for drift detection (0-1).
        """
        self.reference_data = reference_data[feature_columns]
        self.feature_columns = feature_columns
        self.drift_threshold = drift_threshold
        
    def analyze_drift(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> dict:
        """Analyze drift between reference and current data.
        
        Args:
            current_data: Production/current dataset.
            output_path: Optional path to save HTML report.
            
        Returns:
            Dictionary with drift analysis results.
        """
        logger.info("Analyzing data drift...")
        
        # Ensure we only use feature columns
        current_df = current_data[self.feature_columns]
        
        # Create Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_df
        )
        
        # Extract results
        result = report.as_dict()
        
        # Parse drift metrics
        drift_summary = self._parse_drift_results(result)
        
        # Save HTML report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report.save_html(str(output_path))
            logger.info(f"Drift report saved to {output_path}")
            drift_summary["report_path"] = str(output_path)
        
        return drift_summary
    
    def _parse_drift_results(self, result: dict) -> dict:
        """Parse Evidently results into summary dict.
        
        Args:
            result: Raw Evidently report dict.
            
        Returns:
            Parsed drift summary.
        """
        metrics = result.get("metrics", [])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(self.reference_data),
            "dataset_drift_detected": False,
            "drift_share": 0.0,
            "drifted_features": [],
            "total_features": len(self.feature_columns),
        }
        
        # Find dataset drift metric
        for metric in metrics:
            metric_result = metric.get("result", {})
            
            if "drift_share" in metric_result:
                summary["drift_share"] = metric_result["drift_share"]
                summary["dataset_drift_detected"] = (
                    summary["drift_share"] > self.drift_threshold
                )
            
            if "drift_by_columns" in metric_result:
                for col, col_data in metric_result["drift_by_columns"].items():
                    if col_data.get("drift_detected", False):
                        summary["drifted_features"].append({
                            "feature": col,
                            "drift_score": col_data.get("drift_score", 0),
                            "stattest": col_data.get("stattest_name", "unknown")
                        })
        
        summary["drifted_count"] = len(summary["drifted_features"])
        
        return summary
    
    def check_drift(self, current_data: pd.DataFrame) -> bool:
        """Quick check if drift is detected.
        
        Args:
            current_data: Production/current dataset.
            
        Returns:
            True if drift detected above threshold.
        """
        result = self.analyze_drift(current_data)
        return result["dataset_drift_detected"]


def create_drift_report(
    reference_path: Path,
    current_path: Path,
    output_path: Path,
    target_column: str = "Class"
) -> dict:
    """Create drift report from CSV files.
    
    Args:
        reference_path: Path to reference dataset CSV.
        current_path: Path to current dataset CSV.
        output_path: Path to save HTML report.
        target_column: Name of target column to exclude.
        
    Returns:
        Drift analysis summary.
    """
    logger.info(f"Loading reference data from {reference_path}")
    reference_df = pd.read_csv(reference_path)
    
    logger.info(f"Loading current data from {current_path}")
    current_df = pd.read_csv(current_path)
    
    # Get feature columns (exclude target)
    feature_columns = [col for col in reference_df.columns if col != target_column]
    
    # Create monitor and analyze
    monitor = DriftMonitor(
        reference_data=reference_df,
        feature_columns=feature_columns
    )
    
    return monitor.analyze_drift(current_df, output_path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from config import DATA_PROCESSED, REPORTS_DIR
    
    # Example: Check drift between train and test sets
    result = create_drift_report(
        reference_path=DATA_PROCESSED / "train.csv",
        current_path=DATA_PROCESSED / "test.csv",
        output_path=REPORTS_DIR / "drift_report.html"
    )
    
    print("\nDrift Analysis Summary:")
    print(json.dumps(result, indent=2, default=str))
    
    if result["dataset_drift_detected"]:
        print("\n[WARNING] Data drift detected!")
    else:
        print("\n[OK] No significant drift detected.")
