#!/usr/bin/env python3
"""
Test script to verify all Milestone 4 components are working.
Run this to gather evidence for your PDF deliverable.
"""

import requests
import json
import time
import hashlib
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health_and_metrics():
    """Test health endpoints and metrics export."""
    print("\n=== Testing Health & Metrics ===")
    
    # Health check
    resp = requests.get(f"{BASE_URL}/healthz")
    print(f"Health check: {resp.status_code} - {resp.json()}")
    
    # Metrics endpoint
    resp = requests.get(f"{BASE_URL}/metrics")
    print(f"Prometheus metrics available: {resp.status_code}")
    print(f"Sample metrics (first 500 chars):\n{resp.text[:500]}...")
    
    # API metrics summary
    resp = requests.get(f"{BASE_URL}/api/metrics/summary")
    print(f"\nAPI Metrics Summary: {json.dumps(resp.json(), indent=2)}")

def test_model_registry():
    """Test model registry and versioning."""
    print("\n=== Testing Model Registry ===")
    
    resp = requests.get(f"{BASE_URL}/api/models")
    models = resp.json()
    print(f"Available models: {json.dumps(models, indent=2)}")
    
    # Get specific model info
    if models["models"]:
        model_name = models["models"][0]["name"]
        resp = requests.get(f"{BASE_URL}/api/models/{model_name}")
        print(f"\nModel '{model_name}' details: {json.dumps(resp.json(), indent=2)}")

def test_ab_testing():
    """Test A/B testing functionality."""
    print("\n=== Testing A/B Testing ===")
    
    # Create an experiment
    experiment_data = {
        "name": "Model Performance Test",
        "models": {
            "popularity": 0.5,
            "collaborative_filtering": 0.5
        },
        "duration_days": 7
    }
    
    resp = requests.post(f"{BASE_URL}/api/ab/experiments", json=experiment_data)
    if resp.status_code == 200:
        experiment = resp.json()
        exp_id = experiment["experiment_id"]
        print(f"Created experiment: {exp_id}")
        
        # Test user assignment
        user_ids = [123, 456, 789]
        for user_id in user_ids:
            resp = requests.get(f"{BASE_URL}/api/ab/model-assignment/{user_id}")
            assignment = resp.json()
            print(f"User {user_id} assigned to: {assignment['model']}")
        
        # Get experiment results
        resp = requests.get(f"{BASE_URL}/api/ab/experiments/{exp_id}")
        print(f"\nExperiment status: {json.dumps(resp.json(), indent=2)}")

def test_provenance():
    """Test provenance tracking."""
    print("\n=== Testing Provenance ===")
    
    # Make a prediction request
    user_id = 12345
    resp = requests.get(f"{BASE_URL}/api/recommendations/user/{user_id}")
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"Recommendation response includes:")
        print(f"- Request ID: {data.get('request_id', 'N/A')}")
        print(f"- Model Version: {data.get('model_version', 'N/A')}")
        print(f"- Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"- Model Used: {data.get('model_used', 'N/A')}")
        
        # Show full provenance trace
        print(f"\nFull provenance trace:")
        print(json.dumps(data, indent=2))

def test_model_performance():
    """Test model performance tracking."""
    print("\n=== Testing Model Performance ===")
    
    resp = requests.get(f"{BASE_URL}/api/ab/model-performance")
    if resp.status_code == 200:
        performance = resp.json()
        print(f"Model performance metrics: {json.dumps(performance, indent=2)}")

def calculate_availability():
    """Calculate API availability over time."""
    print("\n=== Testing API Availability ===")
    
    total_checks = 10
    successful_checks = 0
    
    for i in range(total_checks):
        try:
            resp = requests.get(f"{BASE_URL}/healthz", timeout=5)
            if resp.status_code == 200:
                successful_checks += 1
                print(f"Check {i+1}: ✓ Success")
            else:
                print(f"Check {i+1}: ✗ Failed (status {resp.status_code})")
        except Exception as e:
            print(f"Check {i+1}: ✗ Failed ({str(e)})")
        
        if i < total_checks - 1:
            time.sleep(1)
    
    availability = (successful_checks / total_checks) * 100
    print(f"\nAvailability: {availability:.1f}% ({successful_checks}/{total_checks} checks)")

def main():
    """Run all tests."""
    print("Milestone 4 Component Testing")
    print("=" * 50)
    
    try:
        test_health_and_metrics()
        test_model_registry()
        test_ab_testing()
        test_provenance()
        test_model_performance()
        calculate_availability()
        
        print("\n" + "=" * 50)
        print("All tests completed! Use the output above for your PDF deliverable.")
        print("\nRemember to also:")
        print("1. Take screenshots of Grafana dashboards")
        print("2. Show evidence of 2+ model retrains in the 7-day window")
        print("3. Document your monitoring alerts and runbook")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Make sure docker-compose is running:")
        print("  docker-compose up -d")

if __name__ == "__main__":
    main()