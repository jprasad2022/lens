#!/usr/bin/env python3
"""
Test and demonstrate provenance tracking
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def make_recommendation_request(user_id=123):
    """Make a recommendation request and return the response"""
    response = requests.get(f"{BASE_URL}/recommend/{user_id}?k=5")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def get_provenance_trace(request_id):
    """Get provenance trace for a request"""
    response = requests.get(f"{BASE_URL}/provenance/trace/{request_id}")
    if response.status_code == 200:
        return response.json()
    else:
        return None

def demonstrate_provenance():
    """Demonstrate full provenance tracking"""
    print("="*60)
    print("PROVENANCE TRACKING DEMONSTRATION")
    print("="*60)
    
    # Make a recommendation request
    print("\n1. Making recommendation request...")
    rec_data = make_recommendation_request(123)
    
    if not rec_data:
        print("Failed to get recommendations")
        return
    
    # Extract provenance info
    if "provenance" in rec_data and rec_data["provenance"]:
        provenance = rec_data["provenance"]
        request_id = provenance.get("request_id", "N/A")
        
        print("\n2. Provenance Information from Response:")
        print(f"   Request ID: {request_id}")
        print(f"   Model Version: {provenance.get('model_version', 'N/A')}")
        print(f"   Data Snapshot: {provenance.get('data_snapshot_id', 'N/A')}")
        print(f"   Pipeline Git SHA: {provenance.get('pipeline_git_sha', 'N/A')}")
        print(f"   Container Digest: {provenance.get('container_image_digest', 'N/A')}")
        print(f"   Timestamp: {provenance.get('timestamp', 'N/A')}")
        
        # Try to get full trace
        print("\n3. Fetching Full Trace...")
        trace = get_provenance_trace(request_id)
        if trace:
            print("   Full trace retrieved successfully!")
            print(json.dumps(trace, indent=2))
        else:
            print("   Note: Trace endpoint not available (provenance router not loaded)")
    else:
        print("\n2. Provenance not found in response")
        print("   This means provenance tracking needs to be enabled")
    
    # Show what full provenance looks like
    print("\n4. Example Complete Provenance Trace:")
    example_trace = {
        "request_id": request_id if 'request_id' in locals() else "550e8400-e29b-41d4-a716-446655440000",
        "user_id": 123,
        "model_version": "collaborative:0.3.0",
        "model_info": {
            "name": "collaborative",
            "version": "0.3.0",
            "trained_at": "2024-11-19T10:00:00Z",
            "metrics": {
                "rmse": 0.876,
                "mae": 0.681,
                "precision@10": 0.321
            }
        },
        "data_snapshot_id": "snapshot-20241119-140020",
        "pipeline_git_sha": "7811a02",
        "container_image_digest": "sha256:a3f5c8d9e2b1",
        "environment": {
            "BUILD_DATE": "2024-11-20T02:00:00Z",
            "VERSION": "0.1.0"
        },
        "request_timestamp": datetime.utcnow().isoformat(),
        "response_latency_ms": rec_data.get("latency_ms", 0) if 'rec_data' in locals() else 145.2,
        "recommendations_returned": len(rec_data.get("recommendations", [])) if 'rec_data' in locals() else 5,
        "cached": rec_data.get("cached", False) if 'rec_data' in locals() else False
    }
    
    print(json.dumps(example_trace, indent=2))
    
    # Save for documentation
    with open("provenance_example.json", "w") as f:
        json.dump(example_trace, f, indent=2)
    
    print("\n5. Provenance example saved to provenance_example.json")

if __name__ == "__main__":
    demonstrate_provenance()