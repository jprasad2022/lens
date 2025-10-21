# Milestone 2: Kafka Wiring, Baselines & First Cloud Deploy

**Team**: Group 1  
**Date**: 2025-10-20

## 1. Kafka Verification

### Topics Created
The following Kafka topics have been created:
- `group1.watch` - User watch events
- `group1.rate` - User rating events  
- `group1.reco_requests` - Recommendation requests
- `group1.reco_responses` - Recommendation responses

### Topic Verification (kcat output)
```bash
# List topics
kcat -L -b $KAFKA_BOOTSTRAP_SERVERS | grep group1

# Output:
group1.watch (3 partitions, rf=3)
group1.rate (3 partitions, rf=3)
group1.reco_requests (3 partitions, rf=3)
group1.reco_responses (3 partitions, rf=3)
```

### Consumer Configuration
```python
kafka_config = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "PLAIN",
    "sasl.username": os.getenv("KAFKA_API_KEY"),
    "sasl.password": os.getenv("KAFKA_API_SECRET"),
    "group.id": "group1-ingestor",
    "auto.offset.reset": "earliest",
}
```

## 2. Data Snapshots

### Storage Configuration
- **Storage Type**: Google Cloud Storage / Local
- **Bucket**: `lens-data-940371601491`
- **Path Pattern**: `group1/events/{event_type}/{timestamp}_{count}_events.parquet`
- **Data Format**: Parquet files with schema validation
- **Retention**: 30 days for event data, indefinite for model artifacts

### Snapshot Structure
```
group1/events/
├── watch/
│   ├── 20241020_120000_1000_events.parquet
│   └── 20241020_130000_1000_events.parquet
├── rate/
│   ├── 20241020_120000_500_events.parquet
│   └── 20241020_130000_500_events.parquet
├── reco_requests/
│   └── 20241020_120000_2000_events.parquet
└── reco_responses/
    └── 20241020_120000_2000_events.parquet
```

### Schema Validation
All events are validated against JSON schemas before storage.

## 3. Model Comparison

### Models Trained
1. **Popularity Model** - Simple popularity-based recommendations
2. **Collaborative Filtering** - Item-based collaborative filtering
3. **ALS** - Alternating Least Squares matrix factorization

### Performance Comparison

| Model | Training Time (s) | Avg Inference (ms) | Model Size (MB) | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|-------|-------------------|--------------------|--------------------|-------|-------|---------|---------|
| Popularity | 0.42 | 0.09 | 0.16 | 0.0821 | 0.1342 | 0.0512 | 0.0689 |
| Collaborative | 4.63 | 71.00 | 0.34 | 0.2145 | 0.3012 | 0.1523 | 0.1789 |
| ALS | 5.81 | 0.06 | 13.94 | 0.2567 | 0.3456 | 0.1834 | 0.2123 |

### Metric Definitions
- **HR@K** (Hit Rate @ K): Fraction of users with ≥1 relevant item in top-K recommendations
- **NDCG@K** (Normalized Discounted Cumulative Gain @ K): Ranking quality metric that considers position
- **Training Time**: Time to train on 1M ratings from MovieLens dataset
- **Inference Latency**: Average time to generate 20 recommendations per user
- **Model Size**: Serialized model size on disk

### Key Findings
- **Popularity Model**: Fastest training and inference but lowest accuracy (non-personalized)
- **Collaborative Filtering**: Good balance of accuracy and interpretability
- **ALS Model**: Best accuracy, fast inference after training, but requires more memory

### Scripts
- Model Training: `backend/scripts/train_simple_models.py`
- Model Comparison: `backend/scripts/compare_models.py`
- ALS Training: `backend/scripts/train_als_model.py`
- Model Fix Script: `backend/scripts/fix_models.py`
- Repository: https://github.com/Group-1-LENS/lens

## 4. Cloud Deployment

### Live Deployment
- **Frontend URL**: https://lens-smoky-six.vercel.app/
- **Backend API**: https://lens-api-940371601491.us-central1.run.app
- **Health Check**: https://lens-api-940371601491.us-central1.run.app/healthz
- **API Documentation**: https://lens-api-940371601491.us-central1.run.app/docs
- **Backend Region**: us-central1 (Google Cloud Run)

### Docker Configuration
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "run.py"]
```

### Registry Image
- **Registry**: Docker Hub / Google Container Registry
- **Image**: `gcr.io/lens-474418/lens-api:latest`
- **Pull Command**: `docker pull gcr.io/lens-474418/lens-api:latest`
- **Build Command**: `docker build -t gcr.io/lens-474418/lens-api:latest ./backend`

### Secrets Configuration
- Kafka credentials stored in GitHub Secrets / Cloud Run environment
- Environment variables injected at runtime
- Service account credentials for GCS access

## 5. Probing Pipeline

### Probe Configuration
- **Schedule**: Every hour via GitHub Actions (`.github/workflows/probe.yml`)
- **Requests per run**: 20
- **Models tested**: popularity, collaborative, als
- **User IDs tested**: Random selection from 1-6040 (MovieLens user range)

### Operations Log (Last 24 Hours)
```
Total probe requests: 480
Successful responses: 475 (99.0%)
Failed responses: 5 (1.0%)

Model Distribution:
- ALS responses: 160 (33.3%)
- Collaborative responses: 155 (32.3%)
- Popularity responses: 160 (33.3%)
- Error responses: 5 (1.0%)

Personalized responses: 315 (66.0%)
Non-personalized responses: 160 (33.3%)

Average latency: 48.7ms
P95 latency: 125.4ms
P99 latency: 287.3ms
```

### Kafka Events Generated
- `group1.reco_requests`: 480 events
- `group1.reco_responses`: 480 events

## Reproducibility Notes

1. **Setup Kafka Topics**:
   ```bash
   python scripts/setup_kafka.py --create --verify
   ```

2. **Train Models**:
   ```bash
   # Train all models
   docker exec lens-backend-1 python scripts/train_simple_models.py
   
   # Fix model serialization issues
   docker exec lens-backend-1 python scripts/fix_models.py
   
   # Compare models
   docker exec lens-backend-1 python scripts/compare_models.py
   ```

3. **Deploy API**:
   ```bash
   # Build and test locally
   docker-compose up -d
   
   # Deploy to cloud
   docker build -t group1/lens-backend:latest ./backend
   docker push group1/lens-backend:latest
   ```

4. **Run Probes**:
   ```bash
   # Test locally
   python scripts/probe.py --test
   
   # Run continuous probing
   python scripts/probe.py --continuous --interval 3600
   ```

## Conclusion

All deliverables for Milestone 2 have been successfully completed:
- ✅ Kafka topics created and verified with proper consumer configuration
- ✅ Stream ingestor with schema validation and parquet snapshots implemented
- ✅ Three baseline models trained and compared (Popularity, Collaborative, ALS)
- ✅ API deployed to cloud with Docker containerization
- ✅ Probe pipeline generating recommendation events to Kafka topics
