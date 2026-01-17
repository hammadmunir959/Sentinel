# Sentinel: End-to-End MLOps Credit Card Fraud Detection

A production-grade MLOps pipeline demonstrating data versioning, experiment tracking, model serving, and drift monitoring.

## Quick Start

```bash
# Clone and setup
cd ML/Sentinel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run preprocessing
cd src && python preprocess.py

# Start API (requires model.json in models/)
uvicorn src.api.main:app --reload
```

**API Docs:** http://localhost:8000/docs

## Project Structure

```
Sentinel/
├── data/               # Datasets (DVC tracked)
├── models/             # Trained model artifacts
├── notebooks/          # Training notebooks (Colab)
├── reports/            # Metrics, plots, drift reports
├── src/
│   ├── api/            # FastAPI application
│   ├── config.py       # Configuration
│   ├── preprocess.py   # Data preprocessing
│   ├── train.py        # Local training script
│   ├── evaluate.py     # Metrics & visualization
│   └── monitor.py      # Drift detection
├── tests/              # Test suite
├── .github/workflows/  # CI/CD
├── Dockerfile          # Container config
└── requirements.txt    # Dependencies
```

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Kaggle    │───>│   DVC       │───>│   Colab     │
│   Dataset   │    │  Versioning │    │   Training  │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Evidently  │<───│  FastAPI    │<───│   MLflow    │
│  Monitoring │    │    API      │    │   DagsHub   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Model Performance

| Metric | Value |
|--------|-------|
| Precision | 0.35 |
| Recall | **0.84** |
| F1-Score | 0.49 |
| ROC-AUC | **0.97** |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |

## Experiments

View experiments: [DagsHub MLflow](https://dagshub.com/hammadmunir959/my-first-repo.mlflow)

## Tech Stack

- **Training:** XGBoost, Google Colab
- **Tracking:** MLflow + DagsHub
- **Versioning:** DVC
- **Serving:** FastAPI
- **Monitoring:** Evidently AI
- **CI/CD:** GitHub Actions + CML

## License

MIT
