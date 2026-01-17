# Sentinel: End-to-End MLOps Credit Card Fraud Detection

> A production-grade MLOps pipeline demonstrating data versioning, automated retraining, model serving, and real-time drift monitoring.

---

## Project Overview

**Problem:** Credit card fraud costs billions annually. Models degrade over time as fraud patterns evolve (concept drift) and transaction distributions shift (data drift).

**Solution:** Build a complete MLOps system that:
1. Trains a fraud detection model with reproducible experiments
2. Versions data and models for auditability
3. Automates retraining via CI/CD
4. Serves predictions via REST API
5. Monitors for drift and alerts when intervention is needed

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Data Versioning | DVC + DagsHub | Track dataset versions, reproducibility |
| Experiment Tracking | MLflow (DagsHub hosted) | Log params, metrics, artifacts |
| Training | XGBoost / LightGBM | Fraud classification |
| CI/CD | GitHub Actions + CML | Automated training on code changes |
| Serving | FastAPI + Uvicorn | REST API for predictions |
| Monitoring | Evidently AI | Data drift detection |
| Deployment | Render (free tier) | Host API |

---

## Project Structure

```
Sentinel/
├── .github/
│   └── workflows/
│       └── train.yaml          # CI/CD pipeline
├── data/
│   ├── raw/                    # Original dataset (DVC tracked)
│   └── processed/              # Train/test splits
├── models/                     # Saved model artifacts
├── reports/                    # Drift reports, metrics
├── src/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters, paths
│   ├── preprocess.py           # Data cleaning, feature engineering
│   ├── train.py                # Model training with MLflow
│   ├── evaluate.py             # Metrics computation
│   ├── predict.py              # Inference logic
│   ├── monitor.py              # Drift detection
│   └── api/
│       ├── __init__.py
│       ├── main.py             # FastAPI app
│       └── schemas.py          # Pydantic models
├── tests/
│   ├── test_preprocess.py
│   └── test_api.py
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory analysis
├── .dvc/                       # DVC config
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── Makefile                    # Common commands
└── README.md
```

---

## Phase 1: Foundation Setup
**Objective:** Establish reproducible project infrastructure.

### 1.1 Repository Initialization
- [ ] Initialize git repository
- [ ] Create `.gitignore` (Python, venv, data files, models)
- [ ] Create project directory structure
- [ ] Set up virtual environment

### 1.2 Dependency Management
- [ ] Create `requirements.txt`:
  ```
  pandas>=2.0.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  xgboost>=2.0.0
  mlflow>=2.9.0
  dvc>=3.30.0
  dagshub>=0.3.0
  fastapi>=0.104.0
  uvicorn>=0.24.0
  evidently>=0.4.0
  pydantic>=2.5.0
  ```
- [ ] Create `requirements-dev.txt`:
  ```
  pytest>=7.4.0
  httpx>=0.25.0
  ruff>=0.1.0
  pre-commit>=3.6.0
  ```
- [ ] Install all dependencies

### 1.3 Configuration Management
- [ ] Create `src/config.py`:
  - Define paths (DATA_RAW, DATA_PROCESSED, MODELS)
  - Define hyperparameters as dataclass
  - Define feature columns
  - Add environment variable support

---

## Phase 2: Data Pipeline
**Objective:** Acquire, version, and preprocess data reproducibly.

### 2.1 Data Acquisition
- [ ] Download Kaggle dataset: `creditcard.csv` (284,807 transactions, 492 frauds)
- [ ] Place in `data/raw/`
- [ ] Document dataset source and license

### 2.2 DVC Setup
- [ ] Initialize DVC: `dvc init`
- [ ] Create DagsHub repository
- [ ] Configure remote: `dvc remote add -d dagshub <url>`
- [ ] Add data: `dvc add data/raw/creditcard.csv`
- [ ] Commit `.dvc` files to git
- [ ] Push data: `dvc push`

### 2.3 Preprocessing Pipeline
- [ ] Create `src/preprocess.py`:
  - Load raw data
  - Handle class imbalance analysis
  - Scale `Amount` and `Time` features
  - Stratified train/validation/test split (70/15/15)
  - Save processed splits to `data/processed/`
- [ ] Add DVC pipeline step:
  ```yaml
  # dvc.yaml
  stages:
    preprocess:
      cmd: python src/preprocess.py
      deps:
        - data/raw/creditcard.csv
        - src/preprocess.py
      outs:
        - data/processed/train.csv
        - data/processed/val.csv
        - data/processed/test.csv
  ```

---

## Phase 3: Model Training & Experiment Tracking (Google Colab)
**Objective:** Train models using Google Colab GPU, with full reproducibility and experiment logging.

> **Note:** Training is done in Google Colab (free GPU) via the Colab extension in VSCode. Local machine is used only for preprocessing, API serving, and monitoring.

### 3.1 Colab Notebook Setup
- [ ] Create `notebooks/train_colab.ipynb`:
  - Mount Google Drive for data/model persistence
  - Install dependencies: `!pip install xgboost mlflow dagshub scikit-learn`
  - Clone repo or pull data from DagsHub/DVC
  - Load processed data from Drive or DVC remote

### 3.2 MLflow Integration (in Colab)
- [ ] Configure DagsHub MLflow tracking URI in Colab:
  ```python
  import dagshub
  dagshub.init(repo_owner="<username>", repo_name="Sentinel", mlflow=True)
  ```
- [ ] Training script in notebook:
  - Define XGBoost classifier with configurable hyperparameters
  - Handle class imbalance (scale_pos_weight)
  - Log to MLflow:
    - Parameters (all hyperparameters)
    - Metrics (Precision, Recall, F1, AUPRC, ROC-AUC)
    - Artifacts (model file, feature importance plot)
    - Model signature
  - Save model to Google Drive / download locally

### 3.3 Local Training Script (Lightweight)
- [ ] Create `src/train.py` (for CI/CD and local testing):
  - Same logic as Colab notebook but can run on subsampled data
  - Used by GitHub Actions for automated validation
  - Full training still happens in Colab

### 3.4 Evaluation Module
- [ ] Create `src/evaluate.py`:
  - Compute classification report
  - Generate confusion matrix visualization
  - Compute precision-recall curve
  - Compute ROC curve
  - Save all plots to `reports/`

### 3.5 Model Download Workflow
- [ ] After Colab training:
  - Download `model.json` from Colab/Drive
  - Place in `models/` directory
  - Commit model artifact (or track via DVC)
- [ ] Create helper script `scripts/download_model.py`:
  - Fetch latest model from MLflow registry
  - Save to local `models/` directory

### 3.6 Baseline Experiment
- [ ] Run training in Colab
- [ ] Verify experiments appear on DagsHub MLflow UI
- [ ] Download trained model to local `models/`
- [ ] Document baseline metrics

---

## Phase 4: CI/CD Automation
**Objective:** Automate training and model evaluation on every code change.

### 4.1 GitHub Actions Workflow
- [ ] Create `.github/workflows/train.yaml`:
  ```yaml
  name: Train Model
  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]

  jobs:
    train:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.11'
        - uses: iterative/setup-cml@v2
        - uses: iterative/setup-dvc@v1
        
        - name: Install dependencies
          run: pip install -r requirements.txt
        
        - name: Pull data
          run: dvc pull
          env:
            DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
        
        - name: Run training pipeline
          run: dvc repro
          env:
            MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
            MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
            MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
        
        - name: Generate CML Report
          if: github.event_name == 'pull_request'
          run: |
            echo "## Model Training Report" >> report.md
            echo "### Metrics" >> report.md
            cat reports/metrics.json >> report.md
            echo "" >> report.md
            echo "### Confusion Matrix" >> report.md
            cml asset publish reports/confusion_matrix.png --md >> report.md
            cml comment create report.md
          env:
            REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  ```

### 4.2 Secrets Configuration
- [ ] Add to GitHub repository secrets:
  - `DAGSHUB_USERNAME`
  - `DAGSHUB_TOKEN`
  - `MLFLOW_TRACKING_URI`

### 4.3 Verification
- [ ] Create test branch with hyperparameter change
- [ ] Open PR and verify CML comment appears
- [ ] Merge and verify main branch training succeeds

---

## Phase 5: Model Serving API
**Objective:** Deploy model as production-ready REST API.

### 5.1 API Development
- [ ] Create `src/api/schemas.py`:
  ```python
  from pydantic import BaseModel
  
  class TransactionInput(BaseModel):
      V1: float
      V2: float
      # ... V28
      Amount: float
      Time: float
  
  class PredictionOutput(BaseModel):
      is_fraud: bool
      fraud_probability: float
      model_version: str
  ```

- [ ] Create `src/api/main.py`:
  - Load model on startup
  - Health check endpoint: `GET /health`
  - Prediction endpoint: `POST /predict`
  - Batch prediction: `POST /predict/batch`
  - Model info: `GET /model/info`

### 5.2 API Testing
- [ ] Create `tests/test_api.py`:
  - Test health endpoint
  - Test valid prediction
  - Test invalid input handling
  - Test batch predictions
- [ ] Run tests: `pytest tests/`

### 5.3 Containerization
- [ ] Create `Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY src/ ./src/
  COPY models/ ./models/
  EXPOSE 8000
  CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Build and test locally: `docker build -t sentinel . && docker run -p 8000:8000 sentinel`

### 5.4 Deployment to Render
- [ ] Create Render account (if needed)
- [ ] Create new Web Service
- [ ] Connect GitHub repository
- [ ] Configure:
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- [ ] Set environment variables
- [ ] Deploy and test live endpoint

---

## Phase 6: Drift Monitoring
**Objective:** Detect when production data diverges from training data.

### 6.1 Reference Dataset
- [ ] Save training data statistics as reference baseline
- [ ] Create `src/monitor.py`:
  - Load reference data
  - Compare against production data window
  - Generate Evidently DataDriftPreset report

### 6.2 Monitoring Integration
- [ ] Add logging middleware to FastAPI:
  - Log each prediction request to `logs/predictions.jsonl`
  - Include timestamp, features, prediction
- [ ] Create drift analysis function:
  ```python
  from evidently.report import Report
  from evidently.metric_preset import DataDriftPreset
  
  def analyze_drift(reference_df, current_df):
      report = Report(metrics=[DataDriftPreset()])
      report.run(reference_data=reference_df, current_data=current_df)
      return report
  ```

### 6.3 Drift Dashboard
- [ ] Add dashboard endpoint: `GET /monitoring/drift`
  - Load last N predictions
  - Generate and return HTML report
- [ ] Create scheduled drift check (GitHub Action cron):
  - Run daily
  - Generate report
  - Alert if drift detected (>20% features drifted)

### 6.4 Alerting
- [ ] Implement drift threshold logic
- [ ] Log warnings when drift detected
- [ ] (Optional) Send email/Slack notification

---

## Phase 7: Documentation & Polish
**Objective:** Make project portfolio-ready.

### 7.1 README.md
- [ ] Project title and badges (CI status, license)
- [ ] One-paragraph description
- [ ] Architecture diagram (Mermaid)
- [ ] Quick start guide
- [ ] API documentation link
- [ ] Screenshots of:
  - MLflow experiment tracking
  - CML PR comment
  - Drift dashboard

### 7.2 Code Quality
- [ ] Configure `ruff` for linting
- [ ] Add pre-commit hooks
- [ ] Ensure all functions have docstrings
- [ ] Type hints on all public functions

### 7.3 Makefile
- [ ] Create convenience commands:
  ```makefile
  install:
      pip install -r requirements.txt -r requirements-dev.txt
  
  train:
      dvc repro
  
  serve:
      uvicorn src.api.main:app --reload
  
  test:
      pytest tests/ -v
  
  lint:
      ruff check src/ tests/
  
  docker-build:
      docker build -t sentinel .
  ```

### 7.4 Final Verification
- [ ] Clone to fresh directory
- [ ] Follow README setup instructions
- [ ] Verify all commands work
- [ ] Test live deployment

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Model AUPRC | > 0.80 |
| API Response Time (p95) | < 100ms |
| CI Pipeline Duration | < 10 min |
| Drift Detection Latency | < 24 hours |
| Test Coverage | > 80% |

---

## Timeline Estimate

| Phase | Duration |
|-------|----------|
| Phase 1: Foundation | 1 day |
| Phase 2: Data Pipeline | 1 day |
| Phase 3: Training | 2 days |
| Phase 4: CI/CD | 1 day |
| Phase 5: API | 2 days |
| Phase 6: Monitoring | 2 days |
| Phase 7: Polish | 1 day |
| **Total** | **~10 days** |

---

## References

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [CML by Iterative](https://cml.dev/)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
