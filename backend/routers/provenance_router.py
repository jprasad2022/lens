"""
Provenance tracking router for full traceability
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from collections import defaultdict

router = APIRouter()

# In-memory storage for provenance traces (in production, use a database)
provenance_store: Dict[str, Dict[str, Any]] = {}
model_lineage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


@router.get("/trace/{request_id}")
async def get_provenance_trace(request_id: str) -> Dict[str, Any]:
    """
    Get full provenance trace for a specific request
    
    Returns all metadata associated with a prediction including:
    - Request ID
    - Model version used
    - Data snapshot ID
    - Pipeline git SHA
    - Container image digest
    - Timestamp
    """
    if request_id not in provenance_store:
        raise HTTPException(status_code=404, detail=f"Trace not found for request_id: {request_id}")
    
    return provenance_store[request_id]


@router.post("/trace")
async def store_provenance_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Store a provenance trace"""
    request_id = trace.get("request_id")
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")
    
    # Add server timestamp
    trace["server_timestamp"] = datetime.utcnow().isoformat()
    
    # Store trace
    provenance_store[request_id] = trace
    
    # Update model lineage
    model_version = trace.get("model_version")
    if model_version:
        model_lineage[model_version].append({
            "request_id": request_id,
            "timestamp": trace["server_timestamp"],
            "user_id": trace.get("user_id")
        })
    
    # Clean up old traces (keep last 1000)
    if len(provenance_store) > 1000:
        oldest_keys = sorted(provenance_store.keys())[:100]
        for key in oldest_keys:
            del provenance_store[key]
    
    return {"status": "stored", "request_id": request_id}


@router.get("/lineage/{model_version}")
async def get_model_lineage(
    model_version: str,
    limit: int = Query(default=100, le=1000)
) -> Dict[str, Any]:
    """
    Get lineage information for a specific model version
    
    Shows all requests that used this model version
    """
    if model_version not in model_lineage:
        return {
            "model_version": model_version,
            "usage_count": 0,
            "requests": []
        }
    
    requests = model_lineage[model_version]
    
    # Sort by timestamp descending
    sorted_requests = sorted(requests, key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "model_version": model_version,
        "usage_count": len(requests),
        "requests": sorted_requests[:limit]
    }


@router.get("/data-snapshot/{snapshot_id}")
async def get_data_snapshot_usage(
    snapshot_id: str,
    limit: int = Query(default=100, le=1000)
) -> Dict[str, Any]:
    """
    Get all models and predictions that used a specific data snapshot
    """
    results = []
    
    for request_id, trace in provenance_store.items():
        if trace.get("data_snapshot_id") == snapshot_id:
            results.append({
                "request_id": request_id,
                "model_version": trace.get("model_version"),
                "timestamp": trace.get("server_timestamp"),
                "user_id": trace.get("user_id")
            })
    
    # Sort by timestamp descending
    sorted_results = sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "data_snapshot_id": snapshot_id,
        "usage_count": len(results),
        "predictions": sorted_results[:limit]
    }


@router.get("/audit-trail")
async def get_audit_trail(
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    user_id: Optional[int] = Query(default=None),
    model_version: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000)
) -> Dict[str, Any]:
    """
    Get audit trail of predictions with filtering options
    """
    # Default time range if not specified
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(hours=24)
    
    results = []
    
    for request_id, trace in provenance_store.items():
        # Parse timestamp
        try:
            trace_time = datetime.fromisoformat(trace.get("server_timestamp", ""))
        except:
            continue
        
        # Apply filters
        if trace_time < start_time or trace_time > end_time:
            continue
        
        if user_id and trace.get("user_id") != user_id:
            continue
        
        if model_version and trace.get("model_version") != model_version:
            continue
        
        results.append({
            "request_id": request_id,
            "timestamp": trace.get("server_timestamp"),
            "user_id": trace.get("user_id"),
            "model_version": trace.get("model_version"),
            "data_snapshot_id": trace.get("data_snapshot_id"),
            "pipeline_git_sha": trace.get("pipeline_git_sha"),
            "container_digest": trace.get("container_image_digest"),
            "latency_ms": trace.get("latency_ms")
        })
    
    # Sort by timestamp descending
    sorted_results = sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "filters": {
            "user_id": user_id,
            "model_version": model_version
        },
        "total_count": len(results),
        "results": sorted_results[:limit]
    }


@router.get("/summary")
async def get_provenance_summary() -> Dict[str, Any]:
    """
    Get summary statistics of provenance tracking
    """
    model_usage = defaultdict(int)
    snapshot_usage = defaultdict(int)
    container_usage = defaultdict(int)
    
    for trace in provenance_store.values():
        if trace.get("model_version"):
            model_usage[trace["model_version"]] += 1
        if trace.get("data_snapshot_id"):
            snapshot_usage[trace["data_snapshot_id"]] += 1
        if trace.get("container_image_digest"):
            container_usage[trace["container_image_digest"]] += 1
    
    return {
        "total_traces": len(provenance_store),
        "model_versions": dict(model_usage),
        "data_snapshots": dict(snapshot_usage),
        "container_images": dict(container_usage),
        "oldest_trace": min(provenance_store.values(), 
                          key=lambda x: x.get("server_timestamp", ""), 
                          default={}).get("server_timestamp"),
        "newest_trace": max(provenance_store.values(), 
                          key=lambda x: x.get("server_timestamp", ""), 
                          default={}).get("server_timestamp")
    }