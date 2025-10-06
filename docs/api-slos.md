# API Contract and SLOs

## Service Level Objectives (SLOs)

### 1. Availability
- **Target**: 99.9% uptime (allows 43.2 minutes downtime per month)
- **Measurement**: Percentage of successful health checks
- **Monitoring**: Synthetic monitoring every 30 seconds

### 2. Latency
- **P50**: < 200ms
- **P95**: < 800ms  
- **P99**: < 2000ms
- **Measurement**: Server-side latency from request received to response sent
- **Exclusions**: Network latency, client-side processing

### 3. Error Rate
- **Target**: < 0.1% 5xx errors
- **Measurement**: Ratio of 5xx responses to total requests
- **Exclusions**: 4xx errors (client errors)

### 4. Throughput
- **Target**: 100 requests per second per instance
- **Burst**: Up to 200 RPS for 60 seconds
- **Rate Limiting**: 100 requests per minute per user

## API Endpoints Contract

### GET /recommend/{user_id}

**Description**: Get personalized movie recommendations for a user

**Parameters**:
- `user_id` (path, required): Integer 1-6040 for MovieLens 1M
- `k` (query, optional): Number of recommendations (default: 20, max: 100)
- `model` (query, optional): Model name to use (default: current active model)

**Response** (200 OK):
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "id": 1,
      "title": "Toy Story (1995)",
      "genres": ["Animation", "Children's", "Comedy"],
      "release_date": "1995-01-01",
      "vote_average": 8.3,
      "poster_path": "/path/to/poster.jpg"
    }
  ],
  "model_info": {
    "name": "als",
    "version": "v1.2.3",
    "type": "collaborative_filtering",
    "trained_at": "2025-01-01T00:00:00Z"
  },
  "is_personalized": true,
  "latency_ms": 145.3,
  "cached": false,
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses**:
- 400: Invalid user_id or parameters
- 404: User not found
- 500: Internal server error
- 503: Service unavailable

**SLO Compliance**:
- MUST respond within 800ms (P95)
- MUST return at least 1 recommendation
- MUST include model version in response

### GET /healthz

**Description**: Health check endpoint

**Response** (200 OK):
```json
{
  "status": "ok",
  "version": "v0.1.0",
  "uptime_seconds": 3600.5,
  "healthy": true,
  "checks": {
    "model_service": {"healthy": true},
    "kafka": {"healthy": true},
    "redis": {"healthy": true}
  }
}
```

**SLO Compliance**:
- MUST respond within 200ms
- MUST NOT perform heavy checks
- MUST return 200 only if service is operational

### GET /metrics

**Description**: Prometheus metrics endpoint

**Response**: Prometheus exposition format
```
# HELP recommend_requests_total Total recommendation requests
# TYPE recommend_requests_total counter
recommend_requests_total{status="200",model="als"} 1234
recommend_requests_total{status="500",model="als"} 2

# HELP recommend_latency_seconds Recommendation request latency
# TYPE recommend_latency_seconds histogram
recommend_latency_seconds_bucket{le="0.1"} 500
recommend_latency_seconds_bucket{le="0.25"} 900
```

**SLO Compliance**:
- MUST expose standard Prometheus metrics
- MUST include custom business metrics
- SHOULD NOT impact main service performance

### POST /feedback

**Description**: Submit user feedback on recommendations

**Request Body**:
```json
{
  "user_id": "123",
  "movie_id": 1,
  "feedback_type": "positive",
  "model_version": "als_v1.2.3",
  "rank_position": 3,
  "timestamp": 1704067200000
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Feedback recorded"
}
```

**SLO Compliance**:
- MUST acknowledge within 500ms
- MUST be idempotent
- SHOULD write to Kafka asynchronously

## Monitoring and Alerting

### Key Metrics
1. **Golden Signals**:
   - Latency: request duration
   - Traffic: requests per second
   - Errors: error rate
   - Saturation: CPU, memory usage

2. **Business Metrics**:
   - Recommendation click-through rate
   - Model performance (online metrics)
   - Cache hit rate
   - Personalization rate

### Alerts
1. **Critical** (Page immediately):
   - Error rate > 1% for 5 minutes
   - P95 latency > 1s for 5 minutes
   - Service down for 2 minutes

2. **Warning** (Notify team):
   - Error rate > 0.5% for 10 minutes
   - P95 latency > 800ms for 10 minutes
   - Memory usage > 80%

## Capacity Planning

### Resource Limits
- CPU: 2 cores per instance
- Memory: 2GB per instance
- Instances: Auto-scale 1-10 based on CPU > 70%
- Model cache: 3 models maximum in memory

### Load Testing Targets
- Sustained: 100 RPS for 1 hour
- Peak: 200 RPS for 5 minutes
- Concurrent users: 1000