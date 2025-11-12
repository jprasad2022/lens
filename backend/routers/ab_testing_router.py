"""
A/B Testing Router - Experiment management endpoints
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from app.dependencies import get_current_user_optional
from services.ab_switch_service import get_ab_switch_service

router = APIRouter()

class ExperimentCreate(BaseModel):
    name: str
    models: Dict[str, float]  # {"model_name": allocation_percentage}
    duration_days: int = 14

class ExperimentMetric(BaseModel):
    experiment_id: str
    model: str
    metric_name: str
    value: float

@router.post("/experiments")
async def create_experiment(
    experiment: ExperimentCreate,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """
    Create a new A/B test experiment
    
    - **name**: Experiment name
    - **models**: Dict of model names to allocation percentages (must sum to 1.0)
    - **duration_days**: Duration of experiment in days (default: 14)
    """
    try:
        ab_service = get_ab_switch_service()
        experiment_id = ab_service.create_experiment(
            name=experiment.name,
            models=experiment.models,
            duration_days=experiment.duration_days
        )
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": f"Experiment '{experiment.name}' created successfully"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")

@router.get("/experiments/{experiment_id}")
async def get_experiment_results(
    experiment_id: str,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Get results for a specific experiment"""
    try:
        ab_service = get_ab_switch_service()
        results = ab_service.get_experiment_results(experiment_id)
        return results
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment results: {str(e)}")

@router.get("/experiments")
async def list_experiments(
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """List all experiments"""
    try:
        ab_service = get_ab_switch_service()
        experiments = []
        
        for exp_id, experiment in ab_service.experiments.items():
            experiments.append({
                "experiment_id": exp_id,
                "name": experiment.name,
                "status": "active" if experiment.is_active() else "completed",
                "start_date": experiment.start_date.isoformat(),
                "end_date": experiment.end_date.isoformat(),
                "models": experiment.models
            })
        
        return {
            "experiments": experiments,
            "count": len(experiments)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")

@router.post("/experiments/{experiment_id}/record")
async def record_metric(
    experiment_id: str,
    metric: ExperimentMetric,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Record a metric for an experiment"""
    try:
        ab_service = get_ab_switch_service()
        
        if experiment_id not in ab_service.experiments:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        experiment = ab_service.experiments[experiment_id]
        experiment.record_metric(metric.model, metric.metric_name, metric.value)
        
        return {
            "status": "success",
            "message": f"Metric recorded for {metric.model}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.get("/experiments/{experiment_id}/winner")
async def get_experiment_winner(
    experiment_id: str,
    metric: str = "latency_ms",
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Get recommended winner for an experiment"""
    try:
        ab_service = get_ab_switch_service()
        winner = ab_service.recommend_winner(experiment_id, metric)
        
        if not winner:
            return {
                "status": "no_winner",
                "message": "No clear winner could be determined"
            }
        
        return {
            "winner": winner,
            "metric": metric,
            "recommendation": f"Model '{winner}' shows best performance for {metric}"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to determine winner: {str(e)}")

@router.get("/model-assignment/{user_id}")
async def get_model_assignment(
    user_id: int,
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Get model assignment for a specific user"""
    try:
        ab_service = get_ab_switch_service()
        model = ab_service.get_model_for_user(user_id)
        
        # Find active experiment
        active_experiment = None
        for exp_id, experiment in ab_service.experiments.items():
            if experiment.is_active():
                active_experiment = exp_id
                break
        
        return {
            "user_id": user_id,
            "assigned_model": model,
            "experiment_id": active_experiment,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model assignment: {str(e)}")

@router.get("/model-performance")
async def get_model_performance(
    current_user: Optional[dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Get performance metrics for all models"""
    try:
        ab_service = get_ab_switch_service()
        performance = ab_service.get_model_performance()
        
        return {
            "models": performance,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")