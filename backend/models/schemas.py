"""
Pydantic models for request/response validation
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator

# Configure Pydantic to allow model_ prefixed fields
class BaseModelWithConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

# Request Models
class RecommendationRequest(BaseModel):
    """Recommendation request parameters"""
    k: int = Field(default=20, ge=1, le=100, description="Number of recommendations")
    model: Optional[str] = Field(default=None, description="Model name to use")
    features: Optional[Dict[str, Any]] = Field(default=None, description="Additional features")

class FeedbackRequest(BaseModelWithConfig):
    """User feedback on recommendations"""
    user_id: str = Field(..., description="User ID")
    movie_id: int = Field(..., description="Movie ID")
    feedback_type: str = Field(..., pattern="^(positive|negative|implicit)$")
    model_version: str = Field(..., description="Model version used")
    rank_position: int = Field(..., ge=1, description="Position in recommendation list")
    timestamp: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))

class ModelSwitchRequest(BaseModel):
    """Model switch request"""
    model: str = Field(..., description="Model name to switch to")
    user_percentage: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0,
        description="Percentage of users for gradual rollout"
    )

class RetrainRequest(BaseModel):
    """Model retrain request"""
    model_type: str = Field(..., description="Type of model to train")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Training parameters")
    use_latest_snapshot: bool = Field(default=True, description="Use latest data snapshot")

# Response Models
class MovieInfo(BaseModel):
    """Movie information"""
    id: int
    title: str
    genres: Optional[List[str]] = []
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    popularity: Optional[float] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    type: str
    trained_at: datetime
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    active: bool = True

class RecommendationResponse(BaseModel):
    """Recommendation response"""
    user_id: int
    recommendations: List[MovieInfo]
    model_info: ModelInfo
    is_personalized: bool = True
    latency_ms: Optional[float] = None
    cached: bool = False
    request_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    healthy: bool
    checks: Dict[str, Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None

class MetricsResponse(BaseModel):
    """Metrics response"""
    timestamp: datetime
    metrics: Dict[str, Any]

class ABTestResult(BaseModelWithConfig):
    """A/B test result"""
    model_a: str
    model_b: str
    metric: str
    value_a: float
    value_b: float
    sample_size_a: int
    sample_size_b: int
    p_value: float
    confidence_interval: List[float]
    significant: bool
    recommendation: str

# Kafka Event Schemas
class WatchEvent(BaseModel):
    """User watch event"""
    ts: int = Field(..., description="Timestamp in milliseconds")
    user_id: int = Field(..., description="User ID")
    movie_id: int = Field(..., description="Movie ID")
    minute: int = Field(..., ge=0, description="Minutes watched")

class RatingEvent(BaseModel):
    """User rating event"""
    ts: int = Field(..., description="Timestamp in milliseconds")
    user_id: int = Field(..., description="User ID")
    movie_id: int = Field(..., description="Movie ID")
    rating: int = Field(..., ge=1, le=5, description="Rating value")

class RecoRequestEvent(BaseModel):
    """Recommendation request event"""
    ts: int = Field(..., description="Timestamp in milliseconds")
    user_id: int = Field(..., description="User ID")
    request_id: str = Field(..., description="Request ID")
    model: str = Field(..., description="Model used")
    k: int = Field(..., description="Number of recommendations requested")

class RecoResponseEvent(BaseModelWithConfig):
    """Recommendation response event"""
    ts: int = Field(..., description="Timestamp in milliseconds")
    user_id: int = Field(..., description="User ID")
    request_id: str = Field(..., description="Request ID")
    status: int = Field(..., description="HTTP status code")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    k: int = Field(..., description="Number of recommendations returned")
    movie_ids: List[int] = Field(..., description="Recommended movie IDs")
    model_version: str = Field(..., description="Model version used")
    cached: bool = Field(default=False, description="Whether response was cached")

# Validation
class PaginationParams(BaseModel):
    """Common pagination parameters"""
    skip: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of items to return")

class DateRangeParams(BaseModel):
    """Date range parameters"""
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    
    @validator('end_date')
    def end_date_after_start(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v