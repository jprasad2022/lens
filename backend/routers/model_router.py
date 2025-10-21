"""
Model management API router.
"""
import os
print(f"LOADING MODEL_ROUTER FROM: {__file__}")
print(f"ABSOLUTE PATH: {os.path.abspath(__file__)}")

from fastapi import APIRouter, HTTPException

from app.state import app_state
from models.schemas import ModelSwitchRequest, RetrainRequest

print("LOADING MODEL_ROUTER.PY - This should appear when backend starts")

router = APIRouter()
print(f"MODEL_ROUTER: Creating router, current routes: {router.routes}")


@router.get("/models")
async def list_models(debug: bool = False, full_debug: bool = False):
    if not app_state.model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    # If full debug mode, return everything
    if full_debug:
        models = await app_state.model_service.list_models()
        return {
            "model_metadata_keys": list(app_state.model_service.model_metadata.keys()),
            "list_models_result": models,
            "model_names": [model["name"] for model in models],
        }
    
    # If debug mode, return raw data
    if debug:
        return {
            "model_metadata_keys": list(app_state.model_service.model_metadata.keys()),
            "model_metadata": app_state.model_service.model_metadata,
            "models_cache_keys": list(app_state.model_service.models.keys()),
        }
    
    # Get unique model names from metadata, excluding :latest entries
    model_names = []
    for key in app_state.model_service.model_metadata.keys():
        if ':' not in key:  # Skip versioned entries like "popularity:latest"
            model_names.append(key)
    
    # Debug logging
    print(f"[DEBUG] Model metadata keys: {list(app_state.model_service.model_metadata.keys())}")
    print(f"[DEBUG] Filtered model names: {model_names}")
    print(f"[DEBUG] Sorted result: {sorted(model_names)}")
    
    return sorted(model_names)  # Sort for consistent order


from fastapi import Response

@router.options("/models/switch")
async def switch_model_options():
    """Handle preflight OPTIONS request"""
    return Response(status_code=200)

@router.post("/models/switch")
async def switch_model(req: ModelSwitchRequest):
    # Update the default model and active models map
    from config.settings import get_settings

    settings = get_settings()
    settings.default_model = req.model
    await app_state.update_model_info(req.model, {"name": req.model, "version": "latest", "type": req.model})
    return {"status": "ok", "active_model": req.model}


@router.get("/models2")
async def list_models2():
    return ["THIS", "IS", "MODELS2"]

@router.get("/debug-models")
async def get_raw_models():
    """Get raw model metadata for debugging"""
    if not app_state.model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    return {
        "model_metadata_keys": list(app_state.model_service.model_metadata.keys()),
        "model_metadata": app_state.model_service.model_metadata,
        "models_cache_keys": list(app_state.model_service.models.keys()),
        "list_models_result": await app_state.model_service.list_models()
    }

@router.post("/models/retrain")
async def retrain_model(req: RetrainRequest):
    # Stub retrain handler
    return {"status": "scheduled", "model_type": req.model_type}