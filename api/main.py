"""
Simplified FastAPI application for spam detection.
Unified, minimal API surface with core endpoints only.
Includes cluster naming for spam classification.
"""

import os
import sys
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Load environment
load_dotenv()

# Add src directory for model code
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'src'))

# ML model imports
from models.predict import (
    load_all_models,
    load_clusterer,
    predict_message,
    predict_with_all_models,
    predict_cluster,
    preprocess_message
)
from utils.helpers import load_config

# Local API modules
from middleware import RateLimiter, log_request_middleware
from error_handlers import http_exception_handler, value_error_handler, general_exception_handler
from sql_db import SimpleDatabase
from seed_database import seed_database
from cluster_names import get_cluster_info
from schemas import (
    MessageRequest,
    BatchMessageRequest,
    PredictionResponse,
    ModelPrediction,
    MultiModelResponse,
    BatchPredictionResponse,
    StatsResponse,
    RecentPredictionsResponse,
    ClusterDistributionResponse,
    ClusterInfoResponse,
)

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
        logger.info("Starting Spam Detection API ...")

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
        logger.info("API startup complete")
    except Exception as e:
        model_meta = {"status": "error", "error": str(e)}
        logger.exception("Startup failed")
    yield
    logger.info("Shutting down Spam Detection API")

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
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)
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


def _get_cluster_details(message: str) -> Dict[str, Any] | None:
    """Return enriched cluster metadata for a message if available."""
    if not message or clusterer is None:
        return None

    try:
        cluster_data = predict_cluster(clusterer, message, enrich_with_names=False)
        if not cluster_data:
            return None

        cluster_id_raw = cluster_data.get("cluster_id")
        try:
            cluster_id = int(cluster_id_raw)
        except (TypeError, ValueError):
            return None

        cluster_meta = get_cluster_info(cluster_id)

        top_terms_raw = cluster_data.get("top_terms", []) or []
        top_terms: List[Dict[str, Any]] = []
        for item in top_terms_raw:
            if isinstance(item, dict):
                term = str(item.get("term", ""))
                score = float(item.get("score", 0.0))
            elif isinstance(item, (list, tuple)) and item:
                term = str(item[0])
                score = float(item[1]) if len(item) > 1 else 0.0
            else:
                continue
            top_terms.append({"term": term, "score": score})

        return {
            "cluster_id": cluster_id,
            "name": str(cluster_meta["name"]),
            "short_name": str(cluster_meta["short_name"]),
            "description": str(cluster_meta["description"]),
            "icon": str(cluster_meta["icon"]),
            "color": str(cluster_meta["color"]),
            "confidence": float(cluster_data.get("confidence", 0.0)),
            "total_clusters": int(cluster_data.get("total_clusters", getattr(clusterer, "best_k", 0) or 0)),
            "top_terms": top_terms,
        }
    except Exception as cluster_error:
        logger.warning(f"Failed to determine cluster info: {cluster_error}")
        return None

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint with model status information."""
    return {
        "status": model_meta.get("status"),
        "model": model_meta.get("primary_model"),
        "models": model_meta.get("available_models", []),
        "clusterer": model_meta.get("clusterer"),
        "version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: MessageRequest) -> PredictionResponse:
    """Predict if a single message is spam using the primary model."""
    _ensure_model_loaded()
    try:
        prediction, confidence = predict_message(model, req.message)
        is_spam = prediction == 'spam'
        processed = preprocess_message(req.message)
        cluster_info = _get_cluster_details(req.message) if is_spam else None
        cluster_id = cluster_info["cluster_id"] if cluster_info else None
        # Persist
        db.save_prediction(
            message=req.message,
            prediction=prediction,
            confidence=confidence,
            is_spam=is_spam,
            processed_message=processed,
            cluster_id=cluster_id
        )
        return PredictionResponse(
            message=req.message,
            prediction=prediction,
            confidence=round(confidence, 4),
            is_spam=is_spam,
            processed_message=processed,
            timestamp=datetime.utcnow().isoformat(),
            cluster=cluster_info
        )
    except Exception as e:
        logger.exception("Single prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchMessageRequest) -> BatchPredictionResponse:
    """Predict if multiple messages are spam (batch processing)."""
    _ensure_model_loaded()
    results: List[PredictionResponse] = []
    spam_count = 0
    try:
        for msg in req.messages:
            pred, conf = predict_message(model, msg)
            is_spam = pred == 'spam'
            processed = preprocess_message(msg)
            cluster_info = _get_cluster_details(msg) if is_spam else None
            cluster_id = cluster_info["cluster_id"] if cluster_info else None
            db.save_prediction(
                message=msg,
                prediction=pred,
                confidence=conf,
                is_spam=is_spam,
                processed_message=processed,
                cluster_id=cluster_id
            )
            if is_spam:
                spam_count += 1
            results.append(PredictionResponse(
                message=msg,
                prediction=pred,
                confidence=round(conf, 4),
                is_spam=is_spam,
                processed_message=processed,
                timestamp=datetime.utcnow().isoformat(),
                cluster=cluster_info
            ))
        return BatchPredictionResponse(
            predictions=results,
            total=len(results),
            spam=spam_count,
            ham=len(results) - spam_count,
            spam_rate=round(spam_count / len(results), 4) if results else 0
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

@app.post("/predict/multi-model", response_model=MultiModelResponse)
async def predict_multi_model(req: MessageRequest) -> MultiModelResponse:
    """Predict using all available models and return ensemble result."""
    if not ENABLE_MULTI_MODEL:
        raise HTTPException(status_code=404, detail="Multi-model endpoint disabled")
    _ensure_model_loaded()
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        raw_results = predict_with_all_models(models, req.message)
        model_outputs = raw_results.get('models', {})

        model_preds: List[ModelPrediction] = []
        spam_votes = 0

        for name, res in model_outputs.items():
            prediction_value = res.get('prediction', 'not_spam')
            confidence_value = float(res.get('confidence', 0))
            is_spam_value = res.get('is_spam', prediction_value == 'spam')

            mp = ModelPrediction(
                model=name,
                prediction=prediction_value,
                confidence=round(confidence_value, 4),
                is_spam=is_spam_value
            )
            if mp.is_spam:
                spam_votes += 1
            model_preds.append(mp)

        ensemble_data = raw_results.get('ensemble', {})
        ensemble_pred = ensemble_data.get('prediction', 'not_spam')
        ensemble_conf = float(ensemble_data.get('confidence', 0))
        if ensemble_conf == 0 and model_preds:
            ensemble_conf = spam_votes / max(1, len(model_preds))

        spam_votes = ensemble_data.get('spam_votes', ensemble_data.get('votes', spam_votes))
        total_models = ensemble_data.get('total_votes', ensemble_data.get('total_models', len(model_preds)))

        processed = preprocess_message(req.message)
        cluster_info = _get_cluster_details(req.message) if ensemble_pred == 'spam' else None
        cluster_id = cluster_info["cluster_id"] if cluster_info else None

        try:
            db.save_prediction(
                message=req.message,
                prediction=ensemble_pred,
                confidence=ensemble_conf,
                is_spam=ensemble_pred == 'spam',
                processed_message=processed,
                model_results={
                    **model_outputs,
                    "ensemble": {
                        "prediction": ensemble_pred,
                        "confidence": ensemble_conf,
                        "is_spam": ensemble_pred == 'spam',
                        "votes": spam_votes,
                        "total_models": total_models,
                    },
                },
                cluster_id=cluster_id,
            )
        except Exception as db_error:
            logger.warning(f"Failed to persist multi-model prediction: {db_error}")

        return MultiModelResponse(
            message=req.message,
            models=model_preds,
            ensemble={
                "prediction": ensemble_pred,
                "confidence": round(ensemble_conf, 4),
                "is_spam": ensemble_pred == 'spam',
                "votes": spam_votes,
                "total_models": total_models,
            },
            processed_message=processed,
            cluster=cluster_info,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.exception("Multi-model prediction failed")
        raise HTTPException(status_code=500, detail=f"Multi-model prediction failed: {e}")

@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Get prediction statistics."""
    return db.get_stats()

@app.get("/trends")
async def trends(period: str = 'day', limit: int = 30) -> Dict[str, Any]:
    """Get trend data over time."""
    if period not in {"hour", "day", "week", "month"}:
        raise HTTPException(status_code=400, detail="Invalid period. Use hour|day|week|month")
    return db.get_trends(period=period, limit=limit)

@app.get("/predictions/recent", response_model=RecentPredictionsResponse)
async def recent(limit: int = 100) -> RecentPredictionsResponse:
    """Get recent predictions."""
    return RecentPredictionsResponse(
        predictions=db.get_recent(limit=limit),
        count=min(limit, 1000)
    )

@app.delete("/predictions")
async def reset() -> Dict[str, str]:
    """Delete all predictions from the database."""
    db.delete_all()
    return {"message": "All predictions deleted", "timestamp": datetime.utcnow().isoformat()}

@app.get("/analytics/clusters", response_model=ClusterDistributionResponse)
async def cluster_distribution() -> ClusterDistributionResponse:
    """Distribution of spam messages across clusters."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialised")

    clusters = db.get_cluster_distribution()
    
    # Enrich clusters with names and metadata
    enriched_clusters = []
    for item in clusters:
        cluster_id = item["cluster_id"]
        cluster_meta = get_cluster_info(cluster_id)
        enriched_clusters.append({
            "cluster_id": cluster_id,
            "name": cluster_meta["name"],
            "short_name": cluster_meta["short_name"],
            "icon": cluster_meta["icon"],
            "color": cluster_meta["color"],
            "count": item["count"]
        })
    
    total = sum(item["count"] for item in enriched_clusters) if enriched_clusters else 0
    return ClusterDistributionResponse(
        total_spam_with_clusters=total,
        clusters=enriched_clusters,
    )


@app.get("/cluster/info", response_model=ClusterInfoResponse)
async def cluster_info() -> ClusterInfoResponse:
    """Detailed metadata about the spam clustering model."""
    if clusterer is None:
        raise HTTPException(status_code=503, detail="Clusterer not available")

    best_k = getattr(clusterer, "best_k", None)
    cluster_results = getattr(clusterer, "cluster_results", {})

    if not best_k or best_k not in cluster_results:
        raise HTTPException(status_code=503, detail="Cluster information not prepared")

    top_terms_map = cluster_results[best_k].get("top_terms", {})

    cluster_details = []
    for cluster_id, terms in top_terms_map.items():
        formatted_terms = [
            {"term": str(term), "score": float(score)}
            for term, score in terms[:10]
        ]
        cluster_id_int = int(cluster_id)
        cluster_meta = get_cluster_info(cluster_id_int)
        
        cluster_detail = {
            "cluster_id": cluster_id_int,
            "name": str(cluster_meta["name"]),
            "short_name": str(cluster_meta["short_name"]),
            "description": str(cluster_meta["description"]),
            "icon": str(cluster_meta["icon"]),
            "color": str(cluster_meta["color"]),
            "num_terms": len(terms),
            "top_terms": formatted_terms,
        }
        logger.debug(f"Created cluster detail: {cluster_detail}")
        cluster_details.append(cluster_detail)

    cluster_details.sort(key=lambda item: item["cluster_id"])

    description = (
        "K-means clustering over the spam corpus with TF-IDF features. "
        "Each cluster represents a distinct type of spam message."
    )

    return ClusterInfoResponse(
        total_clusters=int(best_k),
        silhouette_score=float(getattr(clusterer, "best_silhouette_score", 0.0)),
        description=description,
        clusters=cluster_details,
    )

# ----------------------------------------------------------------------------
# Entry point for manual execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
