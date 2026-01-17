"""
Benchmark the API against the test dataset to generate rigorous performance metrics.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict/batch"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "test.csv"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_graphs(y_true, y_pred, y_prob):
    """Generate professional evaluation plots."""
    sns.set_style("whitegrid")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Custom annotations with TP/FP/TN/FN
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Greys', cbar=False)
    plt.title('Confusion Matrix (API Benchmark)', fontsize=14)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'confusion_matrix_api.png', dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'roc_curve_api.png', dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#2ca02c', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'precision_recall_api.png', dpi=300)
    plt.close()

def benchmark():
    """Run benchmark."""
    if not DATA_PATH.exists():
        logger.error(f"Test data not found at {DATA_PATH}")
        return

    logger.info("Loading test data...")
    df = pd.read_csv(DATA_PATH)
    
    # Take a large sample if dataset is huge, otherwise use all
    # Using 10k samples for speed if needed, but here we use full test set for "rigorous"
    sample_size = min(len(df), 10000) 
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > 10000 else df
    
    y_true = df_sample['Class'].tolist()
    feature_cols = [c for c in df.columns if c != 'Class']
    
    transactions = df_sample[feature_cols].to_dict(orient='records')
    
    # Process in batches
    batch_size = 1000
    y_pred = []
    y_prob = []
    
    logger.info(f"Benchmarking API with {len(transactions)} transactions...")
    start_time = time.time()
    
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        try:
            response = requests.post(API_URL, json={'transactions': batch})
            if response.status_code == 200:
                results = response.json()['predictions']
                for res in results:
                    y_pred.append(res['prediction'])
                    y_prob.append(res['fraud_probability'])
            else:
                logger.error(f"Batch failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return

    duration = time.time() - start_time
    logger.info(f"Benchmark complete in {duration:.2f} seconds ({len(transactions)/duration:.1f} req/s)")

    # Compute Metrics
    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_samples": len(y_true),
        "accuracy": float(sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "latency_per_req_ms": float((duration / len(transactions)) * 1000)
    }
    
    logger.info("Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # Save metrics
    with open(REPORTS_DIR / 'api_benchmark_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate Graphs
    import numpy as np # Need numpy for the heatmap labels
    generate_graphs(y_true, y_pred, y_prob)
    logger.info(f"Graphs saved to {REPORTS_DIR}")

if __name__ == "__main__":
    benchmark()
