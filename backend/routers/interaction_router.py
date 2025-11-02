"""
User Interaction API Router
Handles watch and rating events
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.state import app_state
from app.dependencies import get_current_user_optional
from config.settings import get_settings

# Initialize router
router = APIRouter()
settings = get_settings()

# Request models
class WatchEventRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID")
    progress: float = Field(..., ge=0, le=1, description="Watch progress (0-1)")

class RatingEventRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating (0.5-5.0)")

# Response models
class EventResponse(BaseModel):
    status: str = Field(..., description="Event status")
    message: str = Field(..., description="Status message")

@router.post("/users/{user_id}/watch", response_model=EventResponse)
async def record_watch_event(
    user_id: int,
    event: WatchEventRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> EventResponse:
    """
    Record a watch event for a user

    - **user_id**: User ID (1-6040 for MovieLens 1M)
    - **movie_id**: Movie ID
    - **progress**: Watch progress between 0 and 1
    """
    try:
        # Validate user ID
        if user_id < 1 or user_id > 6040:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID. Must be between 1 and 6040"
            )

        # Send to Kafka
        if app_state.kafka_service:
            await app_state.kafka_service.produce_watch_event(
                user_id=user_id,
                movie_id=event.movie_id,
                progress=event.progress
            )

        return EventResponse(
            status="success",
            message=f"Watch event recorded for user {user_id}, movie {event.movie_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record watch event: {str(e)}"
        )

@router.post("/users/{user_id}/rate", response_model=EventResponse)
async def record_rating_event(
    user_id: int,
    event: RatingEventRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> EventResponse:
    """
    Record a rating event for a user

    - **user_id**: User ID (1-6040 for MovieLens 1M)
    - **movie_id**: Movie ID
    - **rating**: Rating between 0.5 and 5.0
    """
    try:
        # Validate user ID
        if user_id < 1 or user_id > 6040:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID. Must be between 1 and 6040"
            )

        # Send to Kafka
        if app_state.kafka_service:
            await app_state.kafka_service.produce_rate_event(
                user_id=user_id,
                movie_id=event.movie_id,
                rating=event.rating
            )

        return EventResponse(
            status="success",
            message=f"Rating event recorded for user {user_id}, movie {event.movie_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record rating event: {str(e)}"
        )

@router.get("/users/{user_id}/history")
async def get_user_history(
    user_id: int,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Get user interaction history (placeholder for future implementation)

    - **user_id**: User ID (1-6040 for MovieLens 1M)
    """
    # This would query from the data stored by the ingestor
    return {
        "user_id": user_id,
        "message": "History endpoint not yet implemented",
        "info": "Data is being collected via Kafka and stored by the stream ingestor"
    }
