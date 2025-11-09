"""
Pydantic schemas for request/response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    """Request schema for single message prediction."""
    message: str = Field(..., min_length=1, max_length=5000)

    @field_validator('message')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Message cannot be blank')
        return v


class BatchMessageRequest(BaseModel):
    """Request schema for batch message predictions."""
    messages: List[str] = Field(..., min_items=1, max_items=100)

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[str]) -> List[str]:
        for msg in v:
            if not msg.strip():
                raise ValueError('Messages cannot be blank')
        return v


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    message: str
    prediction: str
    confidence: float
    is_spam: bool
    processed_message: str
    timestamp: str
    cluster: Optional['ClusterInfo'] = None


class ModelPrediction(BaseModel):
    """Individual model prediction result."""
    model: str
    prediction: str
    confidence: float
    is_spam: bool


class ClusterInfo(BaseModel):
    """Cluster information for spam messages."""
    cluster_id: int
    name: str
    short_name: str
    description: str
    icon: str
    color: str
    confidence: float
    total_clusters: int
    top_terms: List[Dict[str, Any]]


class MultiModelResponse(BaseModel):
    """Response schema for multi-model prediction."""
    message: str
    models: List[ModelPrediction]
    ensemble: Dict[str, Any]
    timestamp: str
    processed_message: str = ""  # Optional field for processed message
    cluster: Optional[ClusterInfo] = None  # Optional cluster info for spam messages


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total: int
    spam: int
    ham: int
    spam_rate: float


class StatsResponse(BaseModel):
    """Response schema for statistics."""
    total_predictions: int
    spam_count: int
    ham_count: int
    spam_rate: float
    avg_confidence: float


class TrendsResponse(BaseModel):
    """Response schema for trends data."""
    trends: List[Dict[str, Any]]


class RecentPredictionsResponse(BaseModel):
    """Response schema for recent predictions."""
    predictions: List[Dict[str, Any]]
    count: int


class ClusterDetails(BaseModel):
    """Details for a single cluster."""
    cluster_id: int
    name: str
    short_name: str
    description: str
    icon: str
    color: str
    num_terms: int
    top_terms: List[Dict[str, Any]]


class ClusterInfoResponse(BaseModel):
    """Response schema for cluster information."""
    total_clusters: int
    silhouette_score: float
    description: str
    clusters: List[ClusterDetails]


class ClusterDistributionItem(BaseModel):
    """Aggregated spam cluster distribution entry."""
    cluster_id: int
    name: str
    short_name: str
    icon: str
    color: str
    count: int


class ClusterDistributionResponse(BaseModel):
    """Response schema for spam cluster distribution."""
    total_spam_with_clusters: int
    clusters: List[ClusterDistributionItem]



