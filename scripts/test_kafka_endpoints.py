#!/usr/bin/env python3
"""
Test script to verify Kafka endpoints are working
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
USER_ID = 1
MOVIE_ID = 1

def test_watch_event():
    """Test watch event endpoint"""
    print("\n1. Testing Watch Event:")
    url = f"{BASE_URL}/users/{USER_ID}/watch"
    payload = {
        "movie_id": MOVIE_ID,
        "progress": 0.75
    }
    
    response = requests.post(url, json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_rating_event():
    """Test rating event endpoint"""
    print("\n2. Testing Rating Event:")
    url = f"{BASE_URL}/users/{USER_ID}/rate"
    payload = {
        "movie_id": MOVIE_ID,
        "rating": 4.5
    }
    
    response = requests.post(url, json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_recommendation_request():
    """Test recommendation request (should trigger Kafka events)"""
    print("\n3. Testing Recommendation Request:")
    url = f"{BASE_URL}/recommend/{USER_ID}?k=10"
    
    response = requests.get(url)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Request ID: {data.get('request_id')}")
        print(f"   Model: {data.get('model')}")
        print(f"   Recommendations: {len(data.get('recommendations', []))}")
    return response.status_code == 200

def test_user_history():
    """Test user history endpoint"""
    print("\n4. Testing User History:")
    url = f"{BASE_URL}/users/{USER_ID}/history"
    
    response = requests.get(url)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def check_kafka_topics():
    """Check Kafka topics for messages"""
    print("\n5. Checking Kafka Topics:")
    print("   Run these commands to see messages:")
    print("   docker exec lens-kafka-1 kafka-console-consumer --bootstrap-server kafka:9092 --topic group1.watch --from-beginning --max-messages 5")
    print("   docker exec lens-kafka-1 kafka-console-consumer --bootstrap-server kafka:9092 --topic group1.rate --from-beginning --max-messages 5")
    print("   docker exec lens-kafka-1 kafka-console-consumer --bootstrap-server kafka:9092 --topic group1.reco_requests --from-beginning --max-messages 5")
    print("   docker exec lens-kafka-1 kafka-console-consumer --bootstrap-server kafka:9092 --topic group1.reco_responses --from-beginning --max-messages 5")

def main():
    print("=== Testing Kafka Endpoints ===")
    
    # Run tests
    tests = [
        test_watch_event,
        test_rating_event,
        test_recommendation_request,
        test_user_history
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append(False)
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    # Check Kafka topics
    check_kafka_topics()
    
    # Note about Kafka
    print("\n=== Important Notes ===")
    print("1. If Kafka is not configured (running locally), events are silently ignored")
    print("2. When running with docker-compose, Kafka events should be produced")
    print("3. The stream ingestor will consume and store these events")

if __name__ == "__main__":
    main()