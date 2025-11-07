"""
Simplified FastAPI application for spam detection.
Unified, minimal API surface with core endpoints only.
"""

import os
import sys
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator

# Load environment
load_dotenv()

# Add src directory for model code
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'src'))

# Import ML helpers
from models.predict import (
    load_all_models,
    load_clusterer,
    predict_message,
    predict_with_all_models,
    preprocess_message
)
from utils.helpers import load_config

# Local modules
from middleware import RateLimiter, log_request_middleware
from error_handlers import http_exception_handler, value_error_handler, general_exception_handler
from db_simple import SimpleDatabase
from seed_database import seed_database

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("spam_api")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
API_TITLE = os.getenv("API_TITLE", "Spam Detection API")
API_VERSION = os.getenv("API_VERSION", "3.0.0")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5174,http://localhost:3000").split(",")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
AUTO_SEED = os.getenv("AUTO_SEED", "true").lower() == "true"
SEED_COUNT = int(os.getenv("SEED_COUNT", "400"))
ENABLE_MULTI_MODEL = os.getenv("ENABLE_MULTI_MODEL", "true").lower() == "true"

# ----------------------------------------------------------------------------
# Pydantic Schemas (inline to keep API lean)
# ----------------------------------------------------------------------------
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    @field_validator('message')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Message cannot be blank')
        return v

class BatchMessageRequest(BaseModel):
    messages: List[str] = Field(..., min_items=1, max_items=100)
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[str]) -> List[str]:
        for msg in v:
            if not msg.strip():
                raise ValueError('Messages cannot be blank')
        return v

class PredictionResponse(BaseModel):
    message: str
    prediction: str
    confidence: float
    is_spam: bool
    processed_message: str
    timestamp: str

class ModelPrediction(BaseModel):
    model: str
    prediction: str
    confidence: float
    is_spam: bool

class MultiModelResponse(BaseModel):
    message: str
    models: List[ModelPrediction]
    ensemble: Dict[str, Any]
    timestamp: str

# ----------------------------------------------------------------------------
# Global State
# ----------------------------------------------------------------------------
model = None          # Best/default model
models = None         # Dict of all models
clusterer = None      # Optional clusterer
model_meta: Dict[str, Any] = {"status": "booting"}
db: SimpleDatabase | None = None

# ----------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, models, clusterer, model_meta, db
    try:
        logger.info("🚀 Starting Spam Detection API ...")

        # Initialize database
        db = SimpleDatabase()

        if AUTO_SEED and db.is_empty():
            logger.info(f"Seeding empty database with {SEED_COUNT} synthetic predictions ...")
            seed_database(num_records=SEED_COUNT, quiet=True)

        # Load models
        models = load_all_models()
        if not models:
            raise RuntimeError("No models could be loaded")

        # Choose best model (metadata file logic) or fallback first
        config = load_config()
        models_dir = os.path.join(config['data']['output_dir'], 'models')
        # Attempt best model discovery; if fails, pick first
        try:
            from models.predict import get_best_model_info
            best_name, best_score, _ = get_best_model_info(models_dir)
            if best_name and best_name in models:
                model = models[best_name]
                logger.info(f"Using best model: {best_name} (score={best_score:.4f})")
            else:
                model = next(iter(models.values()))
                logger.warning("Best model metadata not found; using first loaded model")
        except Exception as e:
            model = next(iter(models.values()))
            logger.warning(f"Best model resolution failed ({e}); using first model")

        # Load clusterer (optional)
        try:
            clusterer = load_clusterer()
            if clusterer:
                logger.info(f"Clusterer loaded (k={clusterer.best_k})")
        except Exception as e:
            logger.info(f"Clusterer not available: {e}")
            clusterer = None

        model_meta = {
            "status": "ready",
            "loaded_at": datetime.utcnow().isoformat(),
            "primary_model": next(k for k, v in models.items() if v is model),
            "available_models": list(models.keys()),
            "multi_model_enabled": ENABLE_MULTI_MODEL,
            "clusterer": bool(clusterer),
            "clusters": getattr(clusterer, 'best_k', None)
        }
        logger.info("✅ API startup complete")
    except Exception as e:
        model_meta = {"status": "error", "error": str(e)}
        logger.exception("Startup failed")
    yield
    logger.info("👋 Shutting down Spam Detection API")

# ----------------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Simplified spam detection service",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# app.add_middleware(GZipMiddleware, minimum_size=1000)
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
app.middleware("http")(rate_limiter)
app.middleware("http")(log_request_middleware)

# Error handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def _ensure_model_loaded():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": model_meta.get("status"),
        "model": model_meta.get("primary_model"),
        "models": model_meta.get("available_models", []),
        "clusterer": model_meta.get("clusterer"),
        "version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: MessageRequest):
    _ensure_model_loaded()
    try:
        prediction, confidence = predict_message(model, req.message)
        is_spam = prediction == 'spam'
        processed = preprocess_message(req.message)
        # Persist
        db.save_prediction(
            message=req.message,
            prediction=prediction,
            confidence=confidence,
            is_spam=is_spam,
            processed_message=processed
        )
        return PredictionResponse(
            message=req.message,
            prediction=prediction,
            confidence=round(confidence, 4),
            is_spam=is_spam,
            processed_message=processed,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.exception("Single prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/batch")
async def predict_batch(req: BatchMessageRequest):
    _ensure_model_loaded()
    results: List[PredictionResponse] = []
    spam_count = 0
    try:
        for msg in req.messages:
            pred, conf = predict_message(model, msg)
            is_spam = pred == 'spam'
            processed = preprocess_message(msg)
            db.save_prediction(
                message=msg,
                prediction=pred,
                confidence=conf,
                is_spam=is_spam,
                processed_message=processed
            )
            if is_spam:
                spam_count += 1
            results.append(PredictionResponse(
                message=msg,
                prediction=pred,
                confidence=round(conf, 4),
                is_spam=is_spam,
                processed_message=processed,
                timestamp=datetime.utcnow().isoformat()
            ))
        return {
            "predictions": results,
            "total": len(results),
            "spam": spam_count,
            "ham": len(results) - spam_count,
            "spam_rate": round(spam_count / len(results), 4) if results else 0
        }
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

@app.post("/predict/multi-model", response_model=MultiModelResponse)
async def predict_multi_model(req: MessageRequest):
    if not ENABLE_MULTI_MODEL:
        raise HTTPException(status_code=404, detail="Multi-model endpoint disabled")
    _ensure_model_loaded()
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        raw_results = predict_with_all_models(models, req.message)
        model_preds: List[ModelPrediction] = []
        spam_votes = 0
        for name, res in raw_results.items():
            mp = ModelPrediction(
                model=name,
                prediction=res['prediction'],
                confidence=round(res['confidence'], 4),
                is_spam=res['prediction'] == 'spam'
            )
            if mp.is_spam:
                spam_votes += 1
            model_preds.append(mp)
        ensemble_pred = 'spam' if spam_votes >= len(model_preds) / 2 else 'not spam'
        ensemble_conf = spam_votes / max(1, len(model_preds))
        return MultiModelResponse(
            message=req.message,
            models=model_preds,
            ensemble={
                "prediction": ensemble_pred,
                "confidence": round(ensemble_conf, 4),
                "is_spam": ensemble_pred == 'spam',
                "votes": spam_votes,
                "total_models": len(model_preds)
            },
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.exception("Multi-model prediction failed")
        raise HTTPException(status_code=500, detail=f"Multi-model prediction failed: {e}")

@app.get("/stats")
async def stats():
    return db.get_stats()

@app.get("/trends")
async def trends(period: str = 'day', limit: int = 30):
    if period not in {"hour", "day", "week", "month"}:
        raise HTTPException(status_code=400, detail="Invalid period. Use hour|day|week|month")
    return db.get_trends(period=period, limit=limit)

@app.get("/predictions/recent")
async def recent(limit: int = 100):
    return {"predictions": db.get_recent(limit=limit), "count": min(limit, 1000)}

@app.delete("/predictions")
async def reset():
    db.delete_all()
    return {"message": "All predictions deleted", "timestamp": datetime.utcnow().isoformat()}

# ----------------------------------------------------------------------------
# Entry point for manual execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
