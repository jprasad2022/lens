#!/usr/bin/env python3
"""
Test probe functionality locally
"""
import requests
import time
import json

API_URL = "http://localhost:8000"

def test_probe():
    """Test recommendation endpoint and show what would be sent to Kafka"""
    
    print("Testing Probe Functionality")
    print("="*50)
    
    # Test different users and models
    test_cases = [
        {"user_id": 1, "k": 10, "model": "popularity"},
        {"user_id": 100, "k": 20, "model": "als"},
        {"user_id": 500, "k": 10, "model": None},  # Default model
    ]
    
    for test in test_cases:
        user_id = test["user_id"]
        k = test["k"]
        model = test["model"]
        
        print(f"\nProbing user={user_id}, k={k}, model={model}")
        
        # Make request
        start_time = time.time()
        params = {"k": k}
        if model:
            params["model"] = model
            
        try:
            response = requests.get(f"{API_URL}/recommend/{user_id}", params=params)
            latency_ms = (time.time() - start_time) * 1000
            
            print(f"Status: {response.status_code}")
            print(f"Latency: {latency_ms:.1f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Got {len(data.get('recommendations', []))} recommendations")
                print(f"Model version: {data.get('model_info', {}).get('version', 'unknown')}")
                
                # Show what would be sent to Kafka
                print("\nKafka Events that would be produced:")
                
                # Request event
                request_event = {
                    "topic": "group1.reco_requests",
                    "event": {
                        "ts": int(time.time() * 1000),
                        "user_id": user_id,
                        "request_id": data.get("request_id", "unknown"),
                        "model": model or "default",
                        "k": k,
                    }
                }
                print(f"Request: {json.dumps(request_event, indent=2)}")
                
                # Response event
                response_event = {
                    "topic": "group1.reco_responses",
                    "event": {
                        "ts": int(time.time() * 1000),
                        "user_id": user_id,
                        "request_id": data.get("request_id", "unknown"),
                        "status": 200,
                        "latency_ms": latency_ms,
                        "k": len(data.get('recommendations', [])),
                        "movie_ids": [m["id"] for m in data.get("recommendations", [])][:5] + ["..."],
                        "model_version": data.get('model_info', {}).get('version', 'unknown'),
                        "cached": data.get("cached", False),
                    }
                }
                print(f"Response: {json.dumps(response_event, indent=2)}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("Summary:")
    print("- The probe script would hit /recommend endpoint")
    print("- It would produce events to group1.reco_requests BEFORE the API call")
    print("- It would produce events to group1.reco_responses AFTER the API call")
    print("- The backend ALSO produces events when handling the request")
    print("\nTo check Kafka messages:")
    print("docker exec lens-kafka-1 kafka-console-consumer --bootstrap-server kafka:9092 --topic group1.reco_requests --from-beginning --max-messages 10")
    print("docker exec lens-kafka-1 kafka-console-consumer --bootstrap-server kafka:9092 --topic group1.reco_responses --from-beginning --max-messages 10")

if __name__ == "__main__":
    test_probe()