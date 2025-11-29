#!/usr/bin/env python3
"""
Test script to verify the app can start successfully
"""
import os
import sys
import time
import subprocess
import requests

# Set environment variables for testing
os.environ['KAFKA_ENABLED'] = 'false'
os.environ['SKIP_STARTUP_DOWNLOADS'] = 'true'
os.environ['PORT'] = '8080'

print("Starting test server...")

# Start the server in a subprocess
proc = subprocess.Popen(
    [sys.executable, 'cloud_run_startup.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True
)

# Wait for server to start
time.sleep(5)

# Test health endpoint
try:
    response = requests.get('http://localhost:8080/healthz', timeout=5)
    print(f"Health check response: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    if response.status_code == 200:
        print("✅ Server started successfully!")
    else:
        print("❌ Server returned non-200 status")
        
except Exception as e:
    print(f"❌ Failed to connect to server: {e}")

# Stop the server
proc.terminate()
proc.wait()

# Print server output
print("\nServer output:")
print(proc.stdout.read())