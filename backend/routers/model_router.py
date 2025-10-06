"""
Model management API router.
"""

from fastapi import APIRouter, HTTPException

from app.state import app_state
from models.schemas import ModelSwitchRequest, RetrainRequest


router = APIRouter()


@router.get("/models")
async def list_models():
    if not app_state.model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    models = await app_state.model_service.list_models()
    # Return just the model names array for frontend compatibility
    return [model["name"] for model in models]


@router.post("/models/switch")
async def switch_model(req: ModelSwitchRequest):
    # Update the default model and active models map
    from config.settings import get_settings

    settings = get_settings()
    settings.default_model = req.model
    await app_state.update_model_info(req.model, {"name": req.model, "version": "latest", "type": req.model})
    return {"status": "ok", "active_model": req.model}


@router.post("/models/retrain")
async def retrain_model(req: RetrainRequest):
    # Stub retrain handler
    return {"status": "scheduled", "model_type": req.model_type}


