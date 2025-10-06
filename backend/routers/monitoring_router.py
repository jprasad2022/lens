"""
Monitoring and metrics API router.
"""

from datetime import datetime
from fastapi import APIRouter

from app.state import app_state


router = APIRouter()


@router.get("/metrics/summary")
async def metrics_summary():
    return {"timestamp": datetime.utcnow(), "metrics": app_state.get_metrics_summary()}


@router.get("/health")
async def health():
    return await app_state.get_health_status()




