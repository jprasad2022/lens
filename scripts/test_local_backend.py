#!/usr/bin/env python3
"""
Test backend without Kafka dependency
"""

import requests
import time
import json

def wait_for_backend(url, max_retries=30, delay=2):
    """Wait for backend to be ready"""
    print(f"Waiting for backend at {url}...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/healthz", timeout=5)
            if response.status_code == 200:
                print("✓ Backend is ready!")
                return True
        except Exception as e:
            print(f"  Attempt {i+1}/{max_retries}: {type(e).__name__}")
        
        time.sleep(delay)
    
    print("✗ Backend did not start in time")
    return False

def test_backend():
    """Test backend endpoints"""
    base_url = "http://localhost:8000"
    
    # Wait for backend
    if not wait_for_backend(base_url):
        return False
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test metrics endpoint
    print("\n2. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            lines = response.text.split('\n')
            metrics = [l for l in lines if 'http_requests_total' in l and not l.startswith('#')]
            print(f"   Found {len(metrics)} http_requests_total metrics")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test model listing
    print("\n3. Testing model listing...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Available models: {models}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test recommendation (without Kafka)
    print("\n4. Testing recommendation endpoint...")
    try:
        response = requests.get(f"{base_url}/recommend/123?k=5")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Received {len(data.get('recommendations', []))} recommendations")
            if 'provenance' in data:
                print(f"   Provenance tracking: ✓")
                print(f"   Request ID: {data['provenance'].get('request_id')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test model switching
    print("\n5. Testing model switching...")
    try:
        response = requests.post(
            f"{base_url}/models/switch",
            json={"model": "popularity"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return True

if __name__ == "__main__":
    test_backend()