"""
FastAPI application for spam detection web service.
Integrates the AI spam detection model with a RESTful API.
"""

from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
import os
import sys
from datetime import datetime
import time

# Add src directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'src'))

from models.predict import (
    load_trained_model,
    load_all_models,
    load_clusterer,
    predict_message,
    predict_with_all_models,
    predict_cluster,
    get_available_models,
    get_best_model_info,
    preprocess_message
)
from utils.helpers import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="AI-powered spam detection service for cybersecurity applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log request processing time (from Week 10 Lecture 02)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log each request with processing time.
    Demonstrates middleware concept from the lecture.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.url} - Duration: {process_time:.4f} seconds")
    # Add custom header with processing time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global model variables
model = None
models = None  # Dictionary of all models
clusterer = None  # Clustering model
model_info = {
    "name": None,
    "loaded_at": None,
    "status": "not_loaded"
}

# Request/Response Models
class MessageRequest(BaseModel):
    """Request model for single message classification"""
    message: str = Field(..., min_length=1, max_length=5000, description="Message text to classify")

    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v

class BatchMessageRequest(BaseModel):
    """Request model for batch message classification"""
    messages: List[str] = Field(..., min_items=1, max_items=100, description="List of messages to classify")

    @validator('messages')
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError('Messages list cannot be empty')
        for msg in v:
            if not msg.strip():
                raise ValueError('Messages cannot be empty or whitespace only')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    message: str
    prediction: str
    confidence: float
    is_spam: bool
    processed_message: str
    timestamp: str

class ModelPrediction(BaseModel):
    """Individual model prediction"""
    prediction: str
    confidence: float
    is_spam: bool

class EnsemblePrediction(BaseModel):
    """Ensemble prediction result"""
    prediction: str
    confidence: float
    is_spam: bool
    spam_votes: int
    total_votes: int

class ClusterTerm(BaseModel):
    """Top term in a cluster"""
    term: str
    score: float

class ClusterPrediction(BaseModel):
    """Cluster prediction result"""
    cluster_id: int
    confidence: float
    top_terms: List[ClusterTerm]
    total_clusters: int

class MultiModelPredictionResponse(BaseModel):
    """Response model for multi-model prediction results"""
    message: str
    processed_message: str
    multinomial_nb: ModelPrediction
    logistic_regression: ModelPrediction
    linear_svc: ModelPrediction
    ensemble: EnsemblePrediction
    cluster: Optional[ClusterPrediction] = None
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int
    spam_count: int
    not_spam_count: int

class BatchMultiModelPredictionResponse(BaseModel):
    """Response model for batch multi-model predictions"""
    predictions: List[MultiModelPredictionResponse]
    total_processed: int
    spam_count: int
    not_spam_count: int

class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    loaded_at: Optional[str]
    status: str
    available_models: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

class StatsResponse(BaseModel):
    """Statistics response"""
    total_predictions: int
    spam_detected: int
    not_spam_detected: int
    average_confidence: float

# Statistics tracking
stats = {
    "total_predictions": 0,
    "spam_detected": 0,
    "not_spam_detected": 0,
    "confidence_scores": []
}

# Dependency Injection Examples (from Week 10 Lecture 02)
def get_db():
    """
    Dependency function to simulate a database connection.
    Demonstrates dependency injection concept from the lecture.

    In a real application, this would return an actual database session.
    """
    return {
        "db": "Simulated database connection",
        "timestamp": datetime.now().isoformat()
    }

def get_model_dependency():
    """
    Dependency function to get the loaded model.
    Ensures model is available before processing requests.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator."
        )
    return model

@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    global model, models, clusterer, model_info

    try:
        logger.info("Starting up Spam Detection API...")

        # Load the best model for backward compatibility
        model = load_trained_model()

        # Load all models for multi-model predictions
        models = load_all_models()

        # Load clustering model (optional - won't fail if not available)
        try:
            clusterer = load_clusterer()
            if clusterer:
                logger.info(f"Clusterer loaded with k={clusterer.best_k}")
        except Exception as e:
            logger.warning(f"Could not load clusterer (optional): {e}")
            clusterer = None

        # Get model information
        config = load_config()
        models_dir = os.path.join(config['data']['output_dir'], 'models')
        best_model_name, best_score, _ = get_best_model_info(models_dir)

        model_info = {
            "name": best_model_name if best_model_name else "Unknown",
            "loaded_at": datetime.now().isoformat(),
            "status": "loaded",
            "models_loaded": list(models.keys()) if models else [],
            "clusterer_loaded": clusterer is not None,
            "num_clusters": clusterer.best_k if clusterer else None
        }

        logger.info(f"Best model loaded: {model_info['name']}")
        logger.info(f"All models loaded: {model_info['models_loaded']}")
        logger.info(f"Clusterer loaded: {model_info['clusterer_loaded']}")

    except Exception as e:
        logger.error(f"Error loading model on startup: {e}")
        model_info["status"] = "error"
        model_info["error"] = str(e)

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Spam Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "online"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    available_models = get_available_models()

    return ModelInfo(
        name=model_info.get("name", "Unknown"),
        loaded_at=model_info.get("loaded_at"),
        status=model_info.get("status", "unknown"),
        available_models=available_models
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single_message(
    request: MessageRequest,
    model_dep: object = Depends(get_model_dependency),
    db: dict = Depends(get_db)
):
    """
    Classify a single message as spam or not spam.

    Uses dependency injection for model and database (demonstration purpose).
    Returns prediction with confidence score and additional metadata.
    """
    try:
        # Make prediction using injected model dependency
        prediction, confidence = predict_message(model_dep, request.message)
        is_spam = prediction == "spam"

        # Update statistics
        stats["total_predictions"] += 1
        if is_spam:
            stats["spam_detected"] += 1
        else:
            stats["not_spam_detected"] += 1
        stats["confidence_scores"].append(confidence)

        # Get preprocessed message for transparency
        processed_msg = preprocess_message(request.message)

        return PredictionResponse(
            message=request.message,
            prediction=prediction,
            confidence=round(confidence, 4),
            is_spam=is_spam,
            processed_message=processed_msg,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_messages(
    request: BatchMessageRequest,
    model_dep: object = Depends(get_model_dependency),
    db: dict = Depends(get_db)
):
    """
    Classify multiple messages in a single request.

    Uses dependency injection for model and database.
    Maximum 100 messages per batch.
    """
    try:
        predictions = []
        spam_count = 0
        not_spam_count = 0

        for message in request.messages:
            # Make prediction using injected model dependency
            prediction, confidence = predict_message(model_dep, message)
            is_spam = prediction == "spam"

            # Update counts
            if is_spam:
                spam_count += 1
            else:
                not_spam_count += 1

            # Update global statistics
            stats["total_predictions"] += 1
            if is_spam:
                stats["spam_detected"] += 1
            else:
                stats["not_spam_detected"] += 1
            stats["confidence_scores"].append(confidence)

            # Get preprocessed message
            processed_msg = preprocess_message(message)

            predictions.append(PredictionResponse(
                message=message,
                prediction=prediction,
                confidence=round(confidence, 4),
                is_spam=is_spam,
                processed_message=processed_msg,
                timestamp=datetime.now().isoformat()
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            spam_count=spam_count,
            not_spam_count=not_spam_count
        )

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch prediction: {str(e)}"
        )

@app.post("/predict/multi-model", response_model=MultiModelPredictionResponse, tags=["Prediction"])
async def predict_with_multiple_models(
    request: MessageRequest,
    db: dict = Depends(get_db)
):
    """
    Classify a message using all three models and return individual predictions.

    Returns predictions from Multinomial Naive Bayes, Logistic Regression, and Linear SVC,
    plus an ensemble prediction based on majority voting. Useful for comparing model performance
    and creating visualizations/charts.
    """
    try:
        if models is None or not models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded. Please contact administrator."
            )

        # Get predictions from all models
        results = predict_with_all_models(models, request.message)

        # Update statistics based on ensemble prediction
        stats["total_predictions"] += 1
        if results['ensemble']['is_spam']:
            stats["spam_detected"] += 1
        else:
            stats["not_spam_detected"] += 1
        stats["confidence_scores"].append(results['ensemble']['confidence'])

        # Get preprocessed message
        processed_msg = preprocess_message(request.message)

        # Get cluster prediction if ensemble says it's spam and clusterer is available
        cluster_result = None
        if results['ensemble']['is_spam'] and clusterer:
            cluster_data = predict_cluster(clusterer, request.message)
            if cluster_data:
                cluster_result = ClusterPrediction(
                    cluster_id=cluster_data['cluster_id'],
                    confidence=cluster_data['confidence'],
                    top_terms=[ClusterTerm(**term) for term in cluster_data['top_terms']],
                    total_clusters=cluster_data['total_clusters']
                )

        # Build response
        return MultiModelPredictionResponse(
            message=request.message,
            processed_message=processed_msg,
            multinomial_nb=ModelPrediction(**results['models']['multinomial_nb']),
            logistic_regression=ModelPrediction(**results['models']['logistic_regression']),
            linear_svc=ModelPrediction(**results['models']['linear_svc']),
            ensemble=EnsemblePrediction(**results['ensemble']),
            cluster=cluster_result,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing model in results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing model prediction: {str(e)}. Ensure all models are trained."
        )
    except Exception as e:
        logger.error(f"Error during multi-model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing multi-model prediction: {str(e)}"
        )

@app.post("/predict/multi-model/batch", response_model=BatchMultiModelPredictionResponse, tags=["Prediction"])
async def predict_batch_with_multiple_models(
    request: BatchMessageRequest,
    db: dict = Depends(get_db)
):
    """
    Classify multiple messages using all three models.

    Returns predictions from all models for each message.
    Maximum 100 messages per batch.
    Useful for bulk analysis and chart generation.
    """
    try:
        if models is None or not models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded. Please contact administrator."
            )

        predictions = []
        spam_count = 0
        not_spam_count = 0

        for message in request.messages:
            # Get predictions from all models
            results = predict_with_all_models(models, message)

            # Update counts based on ensemble prediction
            if results['ensemble']['is_spam']:
                spam_count += 1
            else:
                not_spam_count += 1

            # Update global statistics
            stats["total_predictions"] += 1
            if results['ensemble']['is_spam']:
                stats["spam_detected"] += 1
            else:
                stats["not_spam_detected"] += 1
            stats["confidence_scores"].append(results['ensemble']['confidence'])

            # Get preprocessed message
            processed_msg = preprocess_message(message)

            # Get cluster prediction if ensemble says it's spam and clusterer is available
            cluster_result = None
            if results['ensemble']['is_spam'] and clusterer:
                cluster_data = predict_cluster(clusterer, message)
                if cluster_data:
                    cluster_result = ClusterPrediction(
                        cluster_id=cluster_data['cluster_id'],
                        confidence=cluster_data['confidence'],
                        top_terms=[ClusterTerm(**term) for term in cluster_data['top_terms']],
                        total_clusters=cluster_data['total_clusters']
                    )

            predictions.append(MultiModelPredictionResponse(
                message=message,
                processed_message=processed_msg,
                multinomial_nb=ModelPrediction(**results['models']['multinomial_nb']),
                logistic_regression=ModelPrediction(**results['models']['logistic_regression']),
                linear_svc=ModelPrediction(**results['models']['linear_svc']),
                ensemble=EnsemblePrediction(**results['ensemble']),
                cluster=cluster_result,
                timestamp=datetime.now().isoformat()
            ))

        return BatchMultiModelPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            spam_count=spam_count,
            not_spam_count=not_spam_count
        )

    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing model in results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing model prediction: {str(e)}. Ensure all models are trained."
        )
    except Exception as e:
        logger.error(f"Error during batch multi-model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch multi-model prediction: {str(e)}"
        )

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """Get prediction statistics since server start"""
    avg_confidence = (
        sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
        if stats["confidence_scores"] else 0.0
    )

    return StatsResponse(
        total_predictions=stats["total_predictions"],
        spam_detected=stats["spam_detected"],
        not_spam_detected=stats["not_spam_detected"],
        average_confidence=round(avg_confidence, 4)
    )

@app.delete("/stats", tags=["Statistics"])
async def reset_statistics():
    """Reset prediction statistics"""
    stats["total_predictions"] = 0
    stats["spam_detected"] = 0
    stats["not_spam_detected"] = 0
    stats["confidence_scores"] = []

    return {
        "message": "Statistics reset successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/examples", tags=["General"])
async def get_example_messages():
    """Get example messages for testing"""
    return {
        "spam_examples": [
            "Congratulations! You have won $1000! Click here to claim your prize now!",
            "URGENT: Your account has been suspended. Verify your identity immediately.",
            "FREE MONEY! Call now to get your cash prize. Limited time offer!",
            "You have been selected for a special offer. Reply with your bank details.",
            "WIN A FREE IPHONE! Click this link now before it expires!"
        ],
        "not_spam_examples": [
            "Hey, are you free for lunch tomorrow?",
            "The meeting has been rescheduled to 3pm.",
            "Thanks for your help with the project!",
            "Can you send me the report when you get a chance?",
            "Happy birthday! Hope you have a great day!"
        ]
    }

@app.get("/cluster/info", tags=["Clustering"])
async def get_cluster_info():
    """
    Get information about the loaded clustering model.

    Returns cluster configuration and top terms for each spam subtype.
    Useful for understanding what types of spam the model has identified.
    """
    if clusterer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clustering model not loaded. Run clustering analysis first."
        )

    try:
        # Get all cluster information
        clusters_info = []

        for cluster_id in range(clusterer.best_k):
            top_terms = clusterer.cluster_results[clusterer.best_k]['top_terms'].get(cluster_id, [])
            top_terms_list = [
                {"term": term, "score": float(score)}
                for term, score in top_terms[:15]
            ]

            clusters_info.append({
                "cluster_id": cluster_id,
                "top_terms": top_terms_list,
                "num_terms": len(top_terms_list)
            })

        return {
            "total_clusters": clusterer.best_k,
            "silhouette_score": clusterer.best_silhouette_score,
            "clusters": clusters_info,
            "description": "K-Means clustering of spam messages to identify spam subtypes"
        }

    except Exception as e:
        logger.error(f"Error getting cluster info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cluster information: {str(e)}"
        )

# Enhanced Error handlers (from Week 10 Lecture 02)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler.
    Returns consistent JSON error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error": "An error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": str(exc),
            "error": "Validation error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error. Please contact administrator.",
            "error": "Something went wrong",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
