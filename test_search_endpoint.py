#!/usr/bin/env python3
"""Test the search endpoint"""

import requests
import json

# Test the search endpoint
base_url = "http://localhost:8000"

# Test 1: Search for "Matrix"
print("Testing search endpoint...")
try:
    response = requests.get(f"{base_url}/search", params={"q": "Matrix"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to backend. Make sure the backend is running on port 8000")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Check if endpoint exists in API docs
print("\nChecking API documentation...")
try:
    response = requests.get(f"{base_url}/docs")
    if response.status_code == 200:
        print("API documentation is available at http://localhost:8000/docs")
except Exception as e:
    print(f"Could not check API docs: {e}")