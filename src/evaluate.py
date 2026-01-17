"""
Evaluation Module for Credit Card Fraud Detection.

This module computes metrics and generates visualizations for model evaluation.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.
        
    Returns:
        Dictionary of computed metrics.
    """
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = None
) -> str:
    """Generate classification report.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_path: Optional path to save report.
        
    Returns:
        Classification report string.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Fraud"],
        digits=4
    )
    
    if output_path:
        output_path.write_text(report)
    
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = None,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_path: Optional path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path = None,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        output_path: Optional path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, "b-", linewidth=2, label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.3)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path = None,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        output_path: Optional path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 15,
    output_path: Path = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """Plot feature importances.
    
    Args:
        feature_names: List of feature names.
        importances: Feature importance values.
        top_n: Number of top features to show.
        output_path: Optional path to save figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics.
        output_path: Path to save JSON.
    """
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def generate_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_names: list = None,
    feature_importances: np.ndarray = None,
    output_dir: Path = None
) -> dict:
    """Generate full evaluation report with all metrics and plots.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        feature_names: Optional list of feature names.
        feature_importances: Optional feature importance values.
        output_dir: Optional directory to save outputs.
        
    Returns:
        Dictionary with all metrics.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Generate plots
    plot_confusion_matrix(
        y_true, y_pred,
        output_path=output_dir / "confusion_matrix.png" if output_dir else None
    )
    plt.close()
    
    plot_precision_recall_curve(
        y_true, y_prob,
        output_path=output_dir / "pr_curve.png" if output_dir else None
    )
    plt.close()
    
    plot_roc_curve(
        y_true, y_prob,
        output_path=output_dir / "roc_curve.png" if output_dir else None
    )
    plt.close()
    
    if feature_names and feature_importances is not None:
        plot_feature_importance(
            feature_names, feature_importances,
            output_path=output_dir / "feature_importance.png" if output_dir else None
        )
        plt.close()
    
    # Save metrics
    if output_dir:
        save_metrics(metrics, output_dir / "metrics.json")
        
        # Save classification report
        report = generate_classification_report(y_true, y_pred)
        (output_dir / "classification_report.txt").write_text(report)
    
    return metrics
