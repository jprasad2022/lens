# Testing Guide for LENS Project

This guide explains how to test all components of the LENS system to ensure everything is working correctly.

## Prerequisites

Before testing, ensure you have:
- Docker and Docker Compose installed
- Python 3.9+ installed
- At least 8GB of free RAM
- Ports 3000, 3001, 8000, 8080, 8081, 9090, 9092 available

## Quick Start Testing

### 1. Start All Services

```bash
cd /mnt/c/Users/jaypr/cot6930/lens
docker-compose up -d
```

Wait for all services to start (approximately 2-3 minutes):
```bash
# Check service status
docker-compose ps

# Check logs for any errors
docker-compose logs -f
```

### 2. Run Integration Test Script

Create and run the integration test script:

```bash
# Create test script
cat > test_integration.py << 'EOF'
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
    print("LENS Integration Test")
    print("=" * 50)
    
    # Test all services
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
        time.sleep(1)
    
    # Test Schema Registry schemas
    print("\nTesting Schema Registry...")
    try:
        response = requests.get("http://localhost:8081/subjects")
        subjects = response.json()
        expected = ["user-interactions-value", "recommendation-requests-value"]
        for subject in expected:
            exists = subject in subjects
            print(f"{'✓' if exists else '✗'} Schema: {subject}")
    except Exception as e:
        print(f"✗ Schema Registry test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All services are running! ✨")
        return 0
    else:
        print("Some services failed. Check logs with: docker-compose logs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make it executable and run
chmod +x test_integration.py
python3 test_integration.py
```

## Component-Specific Testing

### 1. Testing Kafka and Schema Registry

#### a) Register Schemas
```bash
cd schemas
python register_schemas.py
```

Expected output:
```
Connecting to Schema Registry at http://localhost:8081
--------------------------------------------------
✓ Registered schema for subject 'user-interactions-value' with ID: 1
✓ Set compatibility to 'BACKWARD' for subject 'user-interactions-value'
✓ Registered schema for subject 'recommendation-requests-value' with ID: 2
... (more schemas)
--------------------------------------------------
Schema registration complete!
```

#### b) Verify Schemas
```bash
# List all registered subjects
curl -s http://localhost:8081/subjects | jq

# Get specific schema
curl -s http://localhost:8081/subjects/user-interactions-value/versions/latest | jq
```

#### c) Test Kafka Topics
Open Kafka UI at http://localhost:8080 and verify:
- Cluster is connected
- Schema Registry is connected
- Topics can be created

### 2. Testing Backend API

#### a) Health Check
```bash
curl http://localhost:8000/healthz
```
Expected: `{"status":"healthy"}`

#### b) API Documentation
Open http://localhost:8000/docs in browser to see FastAPI documentation

#### c) Test Recommendation Endpoint
```bash
# Create a test recommendation request
curl -X POST http://localhost:8000/api/v1/recommendations/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "model_name": "collaborative_filtering",
    "num_recommendations": 5
  }'
```

#### d) Test Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```
Should return Prometheus metrics

### 3. Testing Frontend

#### a) Access Frontend
Open http://localhost:3000 in browser

#### b) Check Console for Errors
Open browser DevTools (F12) and check for:
- No red errors in Console
- Network tab shows successful API calls
- React components render without errors

#### c) Test User Flow
1. Enter a user ID
2. Select a model
3. Click "Get Recommendations"
4. Verify recommendations appear

### 4. Testing Monitoring Stack

#### a) Prometheus
1. Open http://localhost:9090
2. Check targets: Status → Targets
3. All targets should be "UP"
4. Test a query: `up`

#### b) Grafana
1. Open http://localhost:3001
2. Login: admin/admin
3. Navigate to Dashboards
4. Check "LENS System Metrics" dashboard

### 5. Testing Event Streaming

#### a) Produce Test Event
```python
# save as produce_test_event.py
from confluent_kafka import Producer
import json
import time

producer = Producer({'bootstrap.servers': 'localhost:9093'})

event = {
    "eventId": "test_001",
    "timestamp": int(time.time() * 1000),
    "userId": "user_123",
    "sessionId": "session_456",
    "interactionType": "click",
    "itemId": "item_789"
}

producer.produce(
    'user-interactions',
    key="user_123",
    value=json.dumps(event)
)
producer.flush()
print("Event sent!")
```

Run: `python produce_test_event.py`

#### b) Verify in Kafka UI
1. Go to http://localhost:8080
2. Navigate to Topics → user-interactions
3. Click on Messages tab
4. You should see the test event

### 6. Testing Model Registry

```bash
# Check if model registry is accessible
curl http://localhost:8000/api/v1/models/

# Should return list of available models
```

## Automated Testing

### Run All Tests
```bash
# Run backend tests
cd backend
pytest -v

# Run frontend tests
cd ../frontend
npm test

# Run integration tests
cd ..
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start
```bash
# Check logs
docker-compose logs [service_name]

# Common fixes:
docker-compose down -v  # Remove volumes
docker-compose up -d --force-recreate
```

#### 2. Schema Registry Connection Failed
```bash
# Check if Kafka is ready
docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list

# Wait for Kafka to be ready before starting Schema Registry
docker-compose restart schema-registry
```

#### 3. Frontend Can't Connect to Backend
- Check CORS settings in backend
- Verify `NEXT_PUBLIC_API_URL` in frontend environment
- Check browser console for errors

#### 4. Monitoring Not Working
```bash
# Verify Prometheus can scrape targets
curl http://localhost:8000/metrics
curl http://localhost:3000/api/metrics  # if frontend metrics enabled
```

## Performance Testing

### Load Testing with Locust
```bash
# Install locust
pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class RecommendationUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def get_recommendations(self):
        self.client.post("/api/v1/recommendations/predict", json={
            "user_id": f"user_{random.randint(1, 1000)}",
            "model_name": "collaborative_filtering",
            "num_recommendations": 10
        })
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

Open http://localhost:8089 to configure and run load test

## Validation Checklist

Before considering the system ready:

- [ ] All Docker services are running (`docker-compose ps`)
- [ ] Schema Registry has all 5 schemas registered
- [ ] Backend API health check returns healthy
- [ ] Frontend loads without errors
- [ ] Recommendations endpoint returns valid data
- [ ] Kafka UI shows connection to cluster and schema registry
- [ ] Prometheus is scraping metrics from backend
- [ ] Grafana dashboards show data
- [ ] No ERROR level logs in any service
- [ ] Integration tests pass

## Clean Up

After testing:
```bash
# Stop all services
docker-compose down

# Remove all data (careful!)
docker-compose down -v

# Remove all containers and images
docker-compose down --rmi all
```