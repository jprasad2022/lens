# MovieLens Backend

FastAPI-based backend for the MovieLens recommendation system, following the architectural patterns from the insurance-rag-app.

## Architecture Overview

- **FastAPI** for high-performance async API
- **Modular Router Design** with dynamic loading
- **Service Layer** pattern for business logic
- **Kafka Integration** for real-time streaming
- **Model Registry** for ML model management
- **Prometheus Metrics** for monitoring
- **Docker** containerization

## Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI application entry point
│   ├── state.py          # Global application state
│   ├── dependencies.py   # Dependency injection
│   └── middleware.py     # Custom middleware
├── config/
│   └── settings.py       # Pydantic settings management
├── models/
│   └── schemas.py        # Pydantic models for validation
├── routers/
│   ├── recommendation_router.py
│   ├── model_router.py
│   └── monitoring_router.py
├── services/
│   ├── model_service.py        # ML model management
│   ├── kafka_service.py        # Kafka producer/consumer
│   ├── recommendation_service.py
│   └── cache_service.py        # Redis caching
├── recommender/
│   ├── models.py              # ML model implementations
│   ├── training.py            # Model training pipeline
│   └── evaluation.py          # Model evaluation
├── stream/
│   ├── consumer.py            # Kafka consumer
│   └── producer.py            # Kafka producer
├── scripts/
│   ├── train_models.py        # Training script
│   ├── probe.py              # API prober
│   └── download_movielens.py  # Data download
└── tests/
    └── test_*.py             # Test files
```

## API Endpoints

### Recommendations
- `GET /recommend/{user_id}` - Get recommendations for a user
- `POST /feedback` - Submit feedback on recommendations
- `GET /recommend/{user_id}/explain` - Get recommendation explanations
- `GET /popular` - Get popular movies

### Model Management
- `GET /models` - List available models
- `POST /switch` - Switch active model
- `POST /retrain` - Trigger model retraining
- `GET /models/{name}/metrics` - Get model metrics

### Monitoring
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /monitoring/dashboard` - System metrics
- `GET /experiments/results` - A/B test results

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download MovieLens Data

```bash
python scripts/download_movielens.py
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Train Initial Models

```bash
python scripts/train_models.py --models popularity collaborative als
```

### 5. Run Development Server

```bash
uvicorn app.main:app --reload
```

## Docker Deployment

### Build Image

```bash
docker build -t movielens-backend .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_registry:/app/model_registry \
  --env-file .env \
  movielens-backend
```

## Kafka Integration

### Local Kafka Setup

```bash
# Using Docker Compose
docker-compose up -d kafka zookeeper
```

### Confluent Cloud

1. Set up Confluent Cloud account
2. Create cluster and API keys
3. Update `.env`:

```env
KAFKA_BOOTSTRAP_SERVERS=your-cluster.confluent.cloud:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=PLAIN
KAFKA_API_KEY=your-api-key
KAFKA_API_SECRET=your-api-secret
```

## Model Training Pipeline

### Manual Training

```bash
python scripts/train_models.py --model als --params '{"factors": 100, "iterations": 20}'
```

### Scheduled Retraining

The system automatically retrains models based on the cron schedule in settings.

### Model Registry Structure

```
model_registry/
├── popularity/
│   ├── v0.1/
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── v0.2/
├── collaborative/
└── als/
```

## Monitoring

### Prometheus Metrics

- Request count and latency
- Model performance metrics
- Cache hit rates
- Error rates

Access metrics at: `http://localhost:8000/metrics`

### Health Checks

```bash
curl http://localhost:8000/healthz
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_recommendations.py
```

## Performance Optimization

1. **Caching**: Redis for recommendation results
2. **Model Preloading**: Models loaded at startup
3. **Async Processing**: Non-blocking I/O operations
4. **Connection Pooling**: Reused database connections
5. **Batch Processing**: Efficient data loading

## Security

1. **Authentication**: Optional Firebase/JWT auth
2. **Rate Limiting**: Configurable per-user limits
3. **Input Validation**: Pydantic models
4. **CORS**: Configurable allowed origins
5. **Secrets Management**: Environment variables

## Deployment Checklist

- [ ] Set production environment variables
- [ ] Configure Kafka connection
- [ ] Set up Redis (if using caching)
- [ ] Train and upload models
- [ ] Configure monitoring/alerting
- [ ] Set up SSL/TLS
- [ ] Configure rate limiting
- [ ] Set up backup strategy
- [ ] Load test the API

## Contributing

1. Follow the existing code patterns
2. Add tests for new features
3. Update documentation
4. Use type hints
5. Follow PEP 8 style guide