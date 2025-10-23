"""
Recommendation API Router
Handles all recommendation-related endpoints
"""

import time
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from prometheus_client import Counter, Histogram

from models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    FeedbackRequest,
    MovieInfo
)
from app.state import app_state
from app.dependencies import get_current_user_optional
from services.recommendation_service import RecommendationService
from config.settings import get_settings

# Initialize router
router = APIRouter()
settings = get_settings()

# Metrics
recommend_requests = Counter(
    'recommend_requests_total',
    'Total recommendation requests',
    ['status', 'model', 'cached']
)
recommend_latency = Histogram(
    'recommend_latency_seconds',
    'Recommendation request latency',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    request: Request,
    params: RecommendationRequest = Depends(),
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> RecommendationResponse:
    """
    Get movie recommendations for a user
    
    - **user_id**: User ID (1-6040 for MovieLens 1M)
    - **k**: Number of recommendations (default: 20, max: 100)
    - **model**: Model name to use (optional, defaults to current model)
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Track request
    await app_state.increment_request_count()
    
    try:
        # Get recommendation service
        rec_service = RecommendationService(app_state.model_service)
        
        # Determine model to use
        model_name = params.model
        if not model_name:
            model_name = await app_state.get_current_model(user_id)
        
        # Send request event to Kafka
        if app_state.kafka_service:
            await app_state.kafka_service.produce_reco_request(
                user_id=user_id,
                request_id=request_id,
                model=model_name,
                k=params.k
            )
        
        # Get recommendations
        result = await rec_service.get_recommendations(
            user_id=user_id,
            k=params.k,
            model_name=model_name,
            features=params.features
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        recommend_requests.labels(
            status='200',
            model=model_name,
            cached=str(result.get('cached', False))
        ).inc()
        recommend_latency.observe(time.time() - start_time)
        
        # Send to Kafka if enabled
        if app_state.kafka_service:
            await app_state.kafka_service.produce_reco_response(
                user_id=user_id,
                request_id=request_id,
                status=200,
                latency_ms=latency_ms,
                movie_ids=[m['id'] for m in result['recommendations']],
                model_version=result['model_info']['version'],
                cached=result.get('cached', False)
            )
        
        # Build response
        response = RecommendationResponse(
            user_id=user_id,
            recommendations=[MovieInfo(**movie) for movie in result['recommendations']],
            model_info=result['model_info'],
            is_personalized=result.get('is_personalized', True),
            latency_ms=latency_ms,
            cached=result.get('cached', False),
            request_id=request_id
        )
        
        return response
        
    except ValueError as e:
        await app_state.increment_error_count()
        recommend_requests.labels(status='400', model=params.model or 'unknown', cached='false').inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await app_state.increment_error_count()
        recommend_requests.labels(status='500', model=params.model or 'unknown', cached='false').inc()
        
        # Send error to Kafka
        if app_state.kafka_service:
            await app_state.kafka_service.produce_reco_response(
                user_id=user_id,
                request_id=request_id,
                status=500,
                latency_ms=(time.time() - start_time) * 1000,
                movie_ids=[],
                model_version='unknown',
                cached=False
            )
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Submit feedback on a recommendation
    
    Feedback types:
    - **positive**: User liked the recommendation
    - **negative**: User disliked the recommendation  
    - **implicit**: Implicit feedback (e.g., user watched the movie)
    """
    try:
        # Store feedback
        rec_service = RecommendationService(app_state.model_service)
        await rec_service.store_feedback(feedback.dict())
        
        # Send to Kafka if enabled
        if app_state.kafka_service:
            await app_state.kafka_service.produce_feedback(feedback.dict())
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@router.get("/recommend/{user_id}/explain")
async def explain_recommendations(
    user_id: int,
    model: Optional[str] = None,
    k: int = 5,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Get explanation for recommendations
    
    Returns details about why certain movies were recommended
    """
    try:
        # Get recommendation service
        rec_service = RecommendationService(app_state.model_service)
        
        # Get explanation
        explanation = await rec_service.explain_recommendations(
            user_id=user_id,
            model_name=model or await app_state.get_current_model(user_id),
            k=k
        )
        
        return explanation
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

@router.get("/popular")
async def get_popular_movies(
    k: int = 20,
    genre: Optional[str] = None,
    year: Optional[int] = None
):
    """
    Get popular movies
    
    Returns most popular movies optionally filtered by genre and year
    """
    try:
        rec_service = RecommendationService(app_state.model_service)
        
        movies = await rec_service.get_popular_movies(
            k=k,
            genre=genre,
            year=year
        )
        
        return {
            "movies": [MovieInfo(**movie) for movie in movies],
            "filtered_by": {
                "genre": genre,
                "year": year
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular movies: {str(e)}")