"""
FastAPI application for Credit Card Fraud Detection.

Endpoints:
- GET /health - Health check
- GET /model/info - Model information
- POST /predict - Single transaction prediction
- POST /predict/batch - Batch prediction
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    BatchPredictionOutput,
    BatchTransactionInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionOutput,
    TransactionInput,
)
from config import MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model variable
model = None
MODEL_PATH = MODELS_DIR / "model.json"

# Feature columns in order
FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]


def load_model():
    """Load the XGBoost model."""
    global model
    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Sentinel - Fraud Detection API",
    description="Credit Card Fraud Detection API using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def transaction_to_features(transaction: TransactionInput) -> np.ndarray:
    """Convert transaction input to feature array."""
    features = [getattr(transaction, col) for col in FEATURE_COLUMNS]
    return np.array(features).reshape(1, -1)


def transactions_to_dataframe(transactions: List[TransactionInput]) -> pd.DataFrame:
    """Convert list of transactions to DataFrame."""
    data = []
    for t in transactions:
        row = {col: getattr(t, col) for col in FEATURE_COLUMNS}
        data.append(row)
    return pd.DataFrame(data)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "Sentinel Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type="XGBoost",
        model_path=str(MODEL_PATH),
        features_count=len(FEATURE_COLUMNS),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    """Predict fraud for a single transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = transaction_to_features(transaction)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return PredictionOutput(
            is_fraud=bool(prediction == 1),
            fraud_probability=float(probability),
            prediction=int(prediction)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchTransactionInput):
    """Predict fraud for a batch of transactions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch.transactions) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")
    
    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (1000)")
    
    try:
        df = transactions_to_dataframe(batch.transactions)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionOutput(
                is_fraud=bool(pred == 1),
                fraud_probability=float(prob),
                prediction=int(pred)
            ))
        
        fraud_count = int(sum(predictions))
        
        return BatchPredictionOutput(
            predictions=results,
            total_count=len(results),
            fraud_count=fraud_count
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
