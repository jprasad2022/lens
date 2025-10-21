#!/usr/bin/env python3
"""Integration test script for LENS project"""

import requests
import json
import time
import sys
from datetime import datetime

def test_service(name, url, expected_status=200):
    """Test if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        success = response.status_code == expected_status
        print(f"{'✓' if success else '✗'} {name}: Status {response.status_code}")
        return success
    except Exception as e:
        print(f"✗ {name}: {str(e)}")
        return False

def main():
    print("\nLENS Integration Test Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")
    
    # Test all services
    print("Testing Services:")
    services = [
        ("Backend API", "http://localhost:8000/healthz"),
        ("Frontend", "http://localhost:3000"),
        ("Schema Registry", "http://localhost:8081/subjects"),
        ("Kafka UI", "http://localhost:8080"),
        ("Prometheus", "http://localhost:9090/-/ready"),
        ("Grafana", "http://localhost:3001/api/health"),
    ]
    
    results = []
    for name, url in services:
        results.append(test_service(name, url))
        time.sleep(0.5)
    
    # Test Schema Registry schemas
    print("\nTesting Schema Registry:")
    schema_results = []
    try:
        response = requests.get("http://localhost:8081/subjects")
        subjects = response.json()
        expected_schemas = [
            "user-interactions-value",
            "recommendation-requests-value", 
            "recommendation-responses-value",
            "model-metrics-value",
            "ab-test-events-value"
        ]
        for schema in expected_schemas:
            exists = schema in subjects
            print(f"{'✓' if exists else '✗'} Schema: {schema}")
            schema_results.append(exists)
    except Exception as e:
        print(f"✗ Schema Registry test failed: {e}")
        schema_results = [False]
    
    # Test Backend API endpoints
    print("\nTesting Backend API Endpoints:")
    try:
        # Test recommendation endpoint
        user_id = 123
        params = {
            "k": 5,
            "model": "als"
        }
        
        response = requests.get(
            f"http://localhost:8000/recommend/{user_id}",
            params=params,
            timeout=10
        )
        
        api_test_passed = response.status_code in [200, 201]
        print(f"{'✓' if api_test_passed else '✗'} Recommendation endpoint: Status {response.status_code}")
        
        if api_test_passed:
            data = response.json()
            print(f"  → Received {len(data.get('movies', []))} recommendations")
    except Exception as e:
        print(f"✗ Recommendation endpoint: {e}")
        api_test_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    service_passed = sum(results)
    service_total = len(results)
    schema_passed = sum(schema_results) 
    schema_total = len(schema_results)
    
    print(f"Services: {service_passed}/{service_total} passed")
    print(f"Schemas: {schema_passed}/{schema_total} registered")
    print(f"API Tests: {'Passed' if api_test_passed else 'Failed'}")
    
    all_passed = (service_passed == service_total and 
                  schema_passed == schema_total and 
                  api_test_passed)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! The system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the following:")
        print("\n1. Ensure all services are running: docker-compose ps")
        print("2. Check logs for errors: docker-compose logs")
        print("3. If schemas are missing, run: cd schemas && python register_schemas.py")
        print("4. Make sure all required ports are available")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)