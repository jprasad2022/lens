# MovieLens Recommender System - Project Requirements Analysis

## Executive Summary

This document analyzes the current state of the MovieLens recommender system against the course project requirements and provides a comprehensive implementation roadmap.

**Last Updated:** October 20, 2025

## Quick Status Summary

- **Infrastructure**: ✅ Fully operational (Docker, Kafka, Monitoring)
- **ML Models**: ✅ 3 models implemented (Popularity, CF, ALS)
- **API & Frontend**: ✅ Working recommendation system
- **Streaming**: ⚠️ Kafka setup complete, but consumer logic needed
- **MLOps**: ❌ Manual processes, automation needed
- **Experimentation**: ❌ No A/B testing yet
- **Security & Fairness**: ❌ Not implemented

## Current State vs Requirements

### ✅ Already Implemented

1. **Infrastructure & Deployment**
   - FastAPI backend with `/recommend/{user_id}` endpoint
   - Next.js frontend with full UI
   - Complete Docker containerization
   - Docker-compose orchestration for all services
   - Health check endpoints (`/healthz`)
   - CORS properly configured
   - Environment-based configuration

2. **Data Services**
   - Movie metadata service (3,883 movies loaded)
   - Rating statistics service (1M ratings processed)
   - User demographics service (6,040 users)
   - Redis caching service configured
   - Data loading from MovieLens dataset

3. **ML Models**
   - Popularity-based recommender (baseline)
   - Collaborative Filtering recommender
   - ALS (Alternating Least Squares) recommender
   - Model service with dynamic loading
   - Model registry structure
   - Training scripts (`train_simple_models.py`)

4. **Kafka & Streaming Infrastructure**
   - Kafka broker and Zookeeper running
   - Schema Registry operational
   - All 5 Avro schemas registered:
     - user-interactions-value
     - recommendation-requests-value
     - recommendation-responses-value
     - model-metrics-value
     - ab-test-events-value
   - Kafka UI for monitoring
   - Schema registration scripts

5. **Monitoring & Observability**
   - Prometheus metrics collection configured
   - Grafana dashboards available
   - Basic metrics endpoints
   - Integration test suite (`test_integration.py`)
   - Service health monitoring

6. **Frontend Features**
   - User recommendation interface
   - Model selector (choose algorithm)
   - Movie cards with ratings
   - Responsive design with Tailwind CSS
   - Monitoring dashboard page
   - Search functionality

7. **Development Tools**
   - GitHub Actions workflows (`.github/workflows/`)
   - Integration tests
   - Docker build pipelines
   - Schema validation

### ⚠️ Partially Implemented

#### 1. **Kafka Integration** (Required for M2-M5)
- ✅ Kafka infrastructure running
- ✅ Schema Registry with all schemas
- ✅ Topics can be created
- ❌ No active consumer implementation
- ❌ No producer logic in API endpoints
- ❌ No stream processing to object storage
- ❌ Events not being published from user interactions

#### 2. **Model Training Pipeline** (M2-M3)
- ✅ Training scripts exist (`train_simple_models.py`)
- ✅ Multiple algorithms implemented (Popularity, CF, ALS)
- ❌ No proper train/validation/test splits
- ❌ No Neural CF implementation
- ❌ Missing evaluation metrics (NDCG@K, HR@K)
- ❌ No automated model comparison
- ❌ No hyperparameter tuning

#### 3. **CI/CD & MLOps** (M1, M3-M5)
- ✅ Basic GitHub Actions workflows exist
- ❌ No comprehensive test suite
- ❌ No container registry push
- ❌ No automated cloud deployment
- ❌ No scheduled model retraining
- ❌ No model versioning system
- ❌ No automated rollback

### ❌ Missing Critical Components

#### 1. **Real-time Event Processing**
- No Kafka consumer running in backend
- Events not published on user interactions
- No stream ingestion to object storage
- No real-time model updates

#### 2. **Advanced Model Features**
- No Neural Collaborative Filtering
- Missing offline evaluation metrics
- No hyperparameter optimization
- No model performance tracking over time

#### 3. **Experimentation & A/B Testing** (M4)
- No A/B testing infrastructure
- No online success metrics calculation
- No statistical significance testing
- No experiment tracking
- Model switching exists but not for A/B tests

#### 4. **Data Quality & MLOps** (M3)
- No drift detection implementation
- No data quality monitoring
- No automated retraining triggers
- No model performance degradation alerts
- No data versioning

#### 5. **Security & Fairness** (M5)
- No authentication/authorization
- No rate limiting implementation
- No fairness metrics calculation
- No bias detection
- No security threat analysis
- No input validation beyond basic

#### 6. **Production Readiness**
- No comprehensive error handling
- Missing circuit breakers
- No request tracing
- No performance profiling
- Limited test coverage

## Implementation Roadmap

### Phase 1: Kafka & Streaming Infrastructure (Week 1)

#### 1.1 Kafka Setup
```python
# kafka_config.py
KAFKA_CONFIG = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanism': 'PLAIN',
    'sasl.username': os.getenv('KAFKA_API_KEY'),
    'sasl.password': os.getenv('KAFKA_API_SECRET'),
    'group.id': 'movielens-team-x'
}

# Topics needed:
# - teamX.watch
# - teamX.rate
# - teamX.reco_requests
# - teamX.reco_responses
```

#### 1.2 Event Schemas
```python
# schemas/events.py
from dataclasses import dataclass
from typing import List

@dataclass
class WatchEvent:
    timestamp: int
    user_id: int
    movie_id: int
    minute: int

@dataclass
class RatingEvent:
    timestamp: int
    user_id: int
    movie_id: int
    rating: int

@dataclass
class RecoRequest:
    timestamp: int
    user_id: int
    k: int
    model: str

@dataclass
class RecoResponse:
    timestamp: int
    user_id: int
    request_id: str
    status: int
    latency_ms: float
    movie_ids: List[int]
    model_version: str
```

#### 1.3 Stream Consumer
```python
# stream/consumer.py
import asyncio
from confluent_kafka import Consumer
import pyarrow.parquet as pq

class StreamIngestor:
    def __init__(self, config):
        self.consumer = Consumer(config)
        self.consumer.subscribe(['teamX.watch', 'teamX.rate'])
    
    async def consume_and_store(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg:
                # Validate schema
                event = self.validate_and_parse(msg)
                # Write to parquet
                await self.write_to_storage(event)
                # Commit offset
                self.consumer.commit()
```

### Phase 2: Model Training Pipeline (Week 1-2)

#### 2.1 Data Splitting
```python
# ml/data_split.py
def create_temporal_split(ratings_df, splits=[0.8, 0.1, 0.1]):
    """Create temporal train/val/test splits"""
    ratings_sorted = ratings_df.sort_values('timestamp')
    n = len(ratings_sorted)
    
    train_end = int(n * splits[0])
    val_end = int(n * (splits[0] + splits[1]))
    
    return {
        'train': ratings_sorted[:train_end],
        'val': ratings_sorted[train_end:val_end],
        'test': ratings_sorted[val_end:]
    }

def create_user_split(ratings_df):
    """Create user-based splits for cold-start evaluation"""
    users = ratings_df['user_id'].unique()
    n_users = len(users)
    
    train_users = users[:int(n_users * 0.8)]
    val_users = users[int(n_users * 0.8):int(n_users * 0.9)]
    test_users = users[int(n_users * 0.9):]
    
    return {
        'train': ratings_df[ratings_df['user_id'].isin(train_users)],
        'val': ratings_df[ratings_df['user_id'].isin(val_users)],
        'test': ratings_df[ratings_df['user_id'].isin(test_users)]
    }
```

#### 2.2 Model Implementations
```python
# ml/models/als.py
from implicit import als
import scipy.sparse as sp

class ALSRecommender:
    def __init__(self, factors=50, iterations=10):
        self.model = als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations
        )
    
    def train(self, user_item_matrix):
        self.model.fit(user_item_matrix.T)
    
    def predict(self, user_id, k=20):
        scores = self.model.recommend(
            user_id, 
            user_item_matrix[user_id], 
            N=k
        )
        return [item_id for item_id, _ in scores]

# ml/models/neural_cf.py
import tensorflow as tf

class NeuralCollaborativeFiltering:
    def __init__(self, n_users, n_items, embedding_dim=50):
        self.model = self._build_model(n_users, n_items, embedding_dim)
    
    def _build_model(self, n_users, n_items, embedding_dim):
        # User and item inputs
        user_input = tf.keras.layers.Input(shape=(1,))
        item_input = tf.keras.layers.Input(shape=(1,))
        
        # Embeddings
        user_embedding = tf.keras.layers.Embedding(
            n_users, embedding_dim)(user_input)
        item_embedding = tf.keras.layers.Embedding(
            n_items, embedding_dim)(item_input)
        
        # Neural CF layers
        concat = tf.keras.layers.Concatenate()([
            tf.keras.layers.Flatten()(user_embedding),
            tf.keras.layers.Flatten()(item_embedding)
        ])
        
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
        
        model = tf.keras.Model([user_input, item_input], output)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
```

#### 2.3 Evaluation Metrics
```python
# ml/evaluate.py
import numpy as np

def ndcg_at_k(true_items, pred_items, k=10):
    """Normalized Discounted Cumulative Gain"""
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    
    idcg = dcg_at_k(sorted(true_items, reverse=True), k)
    if not idcg:
        return 0.
    
    return dcg_at_k(true_items, k) / idcg

def hit_rate_at_k(true_items, pred_items, k=10):
    """Hit Rate @ K"""
    pred_items_k = pred_items[:k]
    hits = len(set(true_items) & set(pred_items_k))
    return hits / min(len(true_items), k)

def evaluate_model(model, test_data, k_values=[5, 10, 20]):
    """Comprehensive model evaluation"""
    metrics = {k: {'ndcg': [], 'hr': []} for k in k_values}
    
    for user_id in test_data['user_id'].unique():
        true_items = test_data[test_data['user_id'] == user_id]['item_id'].tolist()
        pred_items = model.predict(user_id, k=max(k_values))
        
        for k in k_values:
            metrics[k]['ndcg'].append(ndcg_at_k(true_items, pred_items, k))
            metrics[k]['hr'].append(hit_rate_at_k(true_items, pred_items, k))
    
    # Average metrics
    for k in k_values:
        metrics[k]['ndcg'] = np.mean(metrics[k]['ndcg'])
        metrics[k]['hr'] = np.mean(metrics[k]['hr'])
    
    return metrics
```

### Phase 3: Enhanced API & Monitoring (Week 2)

#### 3.1 Updated API with Metrics
```python
# backend/app/main.py
from prometheus_client import Counter, Histogram, generate_latest
import asyncio
from uuid import uuid4

# Metrics
request_counter = Counter(
    'recommendation_requests_total',
    'Total recommendation requests',
    ['status', 'model']
)

latency_histogram = Histogram(
    'recommendation_latency_seconds',
    'Recommendation request latency',
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

personalization_rate = Counter(
    'personalized_recommendations_total',
    'Personalized vs non-personalized recommendations',
    ['type']
)

@app.get("/recommend/{user_id}")
@latency_histogram.time()
async def recommend(
    user_id: int, 
    k: int = 20, 
    model: Optional[str] = None,
    request: Request
):
    request_id = str(uuid4())
    start_time = time.time()
    
    try:
        # A/B test assignment
        if not model:
            model = assign_model_ab(user_id)
        
        # Send request event to Kafka
        await kafka_producer.send('teamX.reco_requests', {
            'timestamp': int(time.time()),
            'user_id': user_id,
            'request_id': request_id,
            'k': k,
            'model': model
        })
        
        # Get recommendations
        movie_ids = await recommendation_service.get_recommendations(
            user_id, k, model
        )
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        request_counter.labels(status='200', model=model).inc()
        
        is_personalized = len(movie_ids) > 0 and model != 'popularity'
        personalization_rate.labels(
            type='personalized' if is_personalized else 'generic'
        ).inc()
        
        # Send response event
        await kafka_producer.send('teamX.reco_responses', {
            'timestamp': int(time.time()),
            'user_id': user_id,
            'request_id': request_id,
            'status': 200,
            'latency_ms': latency_ms,
            'movie_ids': movie_ids,
            'model_version': f"{model}_v1.0",
            'is_personalized': is_personalized
        })
        
        return {
            'user_id': user_id,
            'recommendations': movie_ids,
            'model': model,
            'request_id': request_id
        }
        
    except Exception as e:
        request_counter.labels(status='500', model=model).inc()
        raise HTTPException(500, str(e))

@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

#### 3.2 A/B Testing Logic
```python
# backend/services/ab_testing.py
import hashlib

class ABTestManager:
    def __init__(self, config):
        self.experiments = config.get('experiments', {})
        
    def assign_model(self, user_id: int) -> str:
        """Assign user to model variant"""
        # Current experiment config
        experiment = self.experiments.get('current', {
            'models': ['popularity', 'als'],
            'split': [0.5, 0.5]
        })
        
        # Consistent hashing for user assignment
        hash_val = int(hashlib.md5(
            str(user_id).encode()
        ).hexdigest(), 16)
        
        bucket = hash_val % 100
        cumulative = 0
        
        for i, (model, percentage) in enumerate(
            zip(experiment['models'], experiment['split'])
        ):
            cumulative += percentage * 100
            if bucket < cumulative:
                return model
        
        return experiment['models'][0]
    
    def get_experiment_metrics(self):
        """Calculate A/B test metrics from Kafka logs"""
        # Read from reco_responses topic
        # Calculate success rates per model
        # Perform statistical test
        pass
```

### Phase 4: CI/CD Pipeline (Week 2-3)

#### 4.1 GitHub Actions for CI
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source
        flake8 . --count --exit-zero --max-complexity=10
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Check coverage
      run: |
        coverage report --fail-under=70
    
    - name: Build Docker image
      run: |
        docker build -t recommender:test -f docker/recommender.Dockerfile .
        
    - name: Run container tests
      run: |
        docker run --rm recommender:test pytest
```

#### 4.2 CD Pipeline
```yaml
# .github/workflows/cd.yml
name: CD Pipeline

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=sha,prefix={{branch}}-
          type=raw,value=latest
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/recommender.Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
    
    - name: Deploy to Cloud Run
      uses: google-github-actions/deploy-cloudrun@v1
      with:
        service: movielens-recommender
        image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
        region: us-central1
        env_vars: |
          KAFKA_BOOTSTRAP_SERVERS=${{ secrets.KAFKA_BOOTSTRAP }}
          KAFKA_API_KEY=${{ secrets.KAFKA_API_KEY }}
          KAFKA_API_SECRET=${{ secrets.KAFKA_API_SECRET }}
          MODEL_VERSION=${{ github.sha }}
    
    - name: Update deployment status
      run: |
        echo "## Deployment Summary" >> $GITHUB_STEP_SUMMARY
        echo "- Image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest" >> $GITHUB_STEP_SUMMARY
        echo "- Digest: ${{ steps.meta.outputs.digest }}" >> $GITHUB_STEP_SUMMARY
        echo "- Service URL: https://movielens-recommender.run.app" >> $GITHUB_STEP_SUMMARY
```

#### 4.3 Scheduled Retraining
```yaml
# .github/workflows/retrain.yml
name: Model Retraining

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: pip install -r requirements-train.txt
    
    - name: Download latest data snapshot
      run: |
        python scripts/download_snapshot.py \
          --output data/snapshots/$(date +%Y%m%d)
    
    - name: Train models
      run: |
        python ml/train.py \
          --data data/snapshots/$(date +%Y%m%d) \
          --models popularity,als,neural_cf \
          --output model_registry/
    
    - name: Evaluate models
      run: |
        python ml/evaluate.py \
          --models model_registry/$(date +%Y%m%d)/ \
          --output evaluation_results.json
    
    - name: Upload to object storage
      run: |
        gsutil -m cp -r model_registry/$(date +%Y%m%d) \
          gs://movielens-models/
    
    - name: Trigger model update
      run: |
        curl -X POST https://movielens-recommender.run.app/model/update \
          -H "Authorization: Bearer ${{ secrets.API_TOKEN }}" \
          -d '{"version": "$(date +%Y%m%d)"}'
```

### Phase 5: Advanced Features (Week 3-4)

#### 5.1 Drift Detection
```python
# ml/drift.py
import pandas as pd
from scipy import stats

class DriftDetector:
    def __init__(self, baseline_data):
        self.baseline = self._compute_baseline_stats(baseline_data)
    
    def _compute_baseline_stats(self, data):
        return {
            'user_activity': data.groupby('user_id').size().describe(),
            'rating_distribution': data['rating'].value_counts(normalize=True),
            'genre_popularity': self._compute_genre_stats(data),
            'temporal_patterns': self._compute_temporal_stats(data)
        }
    
    def detect_drift(self, current_data, threshold=0.05):
        """Detect distribution drift using KS test"""
        drift_results = {}
        
        # User activity drift
        current_activity = current_data.groupby('user_id').size()
        ks_stat, p_value = stats.ks_2samp(
            self.baseline['user_activity']['mean'],
            current_activity.mean()
        )
        drift_results['user_activity'] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'is_drift': p_value < threshold
        }
        
        # Rating distribution drift
        current_ratings = current_data['rating'].value_counts(normalize=True)
        chi2, p_value = stats.chisquare(
            current_ratings,
            self.baseline['rating_distribution']
        )
        drift_results['rating_distribution'] = {
            'chi2': chi2,
            'p_value': p_value,
            'is_drift': p_value < threshold
        }
        
        return drift_results
```

#### 5.2 Online Success Metrics
```python
# scripts/online_metric.py
from confluent_kafka import Consumer
import pandas as pd
from datetime import datetime, timedelta

class OnlineMetricCalculator:
    def __init__(self, kafka_config):
        self.consumer = Consumer(kafka_config)
        self.consumer.subscribe(['teamX.reco_responses', 'teamX.watch'])
    
    def calculate_success_rate(self, window_minutes=30):
        """Calculate % of users who watched recommended movies"""
        recommendations = []
        watches = []
        
        # Consume events from last N minutes
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        while True:
            msg = self.consumer.poll(1.0)
            if not msg:
                break
                
            event = json.loads(msg.value())
            if msg.topic() == 'teamX.reco_responses':
                recommendations.append(event)
            elif msg.topic() == 'teamX.watch':
                watches.append(event)
        
        # Join recommendations with watches
        reco_df = pd.DataFrame(recommendations)
        watch_df = pd.DataFrame(watches)
        
        # Success = user watched any recommended movie within window
        success_count = 0
        for _, reco in reco_df.iterrows():
            user_watches = watch_df[
                (watch_df['user_id'] == reco['user_id']) &
                (watch_df['timestamp'] > reco['timestamp']) &
                (watch_df['timestamp'] < reco['timestamp'] + window_minutes*60)
            ]
            
            if any(watch['movie_id'] in reco['movie_ids'] 
                   for _, watch in user_watches.iterrows()):
                success_count += 1
        
        success_rate = success_count / len(reco_df) if len(reco_df) > 0 else 0
        
        # Calculate per model
        model_metrics = reco_df.groupby('model_version').apply(
            lambda x: self._calculate_model_success(x, watch_df, window_minutes)
        )
        
        return {
            'overall_success_rate': success_rate,
            'model_metrics': model_metrics.to_dict(),
            'sample_size': len(reco_df)
        }
```

#### 5.3 Provenance Tracking
```python
# backend/services/provenance.py
import hashlib
import git
from datetime import datetime

class ProvenanceTracker:
    def __init__(self):
        self.repo = git.Repo(search_parent_directories=True)
    
    def get_provenance(self, request_id: str, model_name: str):
        """Generate complete provenance record"""
        return {
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'model': {
                'name': model_name,
                'version': self._get_model_version(model_name),
                'training_data_snapshot': self._get_data_snapshot_id(),
                'hyperparameters': self._get_model_params(model_name)
            },
            'pipeline': {
                'git_commit': self.repo.head.object.hexsha,
                'git_branch': self.repo.active_branch.name,
                'git_dirty': self.repo.is_dirty()
            },
            'infrastructure': {
                'container_image': os.getenv('CONTAINER_IMAGE_DIGEST'),
                'deployment_id': os.getenv('CLOUD_RUN_REVISION'),
                'region': os.getenv('CLOUD_RUN_REGION')
            }
        }
    
    def _get_model_version(self, model_name):
        """Get model version from registry"""
        registry_path = f"model_registry/{model_name}/metadata.json"
        with open(registry_path) as f:
            metadata = json.load(f)
        return metadata['version']
    
    def _get_data_snapshot_id(self):
        """Get data snapshot ID used for training"""
        # This should be stored during training
        return os.getenv('DATA_SNAPSHOT_ID', 'unknown')
```

### Phase 6: Fairness & Security (Week 4)

#### 6.1 Fairness Analysis
```python
# ml/fairness.py
import pandas as pd
import numpy as np

class FairnessAnalyzer:
    def __init__(self, user_demographics, recommendations):
        self.demographics = user_demographics
        self.recommendations = recommendations
    
    def analyze_demographic_parity(self, protected_attribute='gender'):
        """Check if recommendations are equally distributed across groups"""
        results = {}
        
        for group in self.demographics[protected_attribute].unique():
            group_users = self.demographics[
                self.demographics[protected_attribute] == group
            ]['user_id']
            
            group_recos = self.recommendations[
                self.recommendations['user_id'].isin(group_users)
            ]
            
            results[group] = {
                'avg_recommendations': len(group_recos) / len(group_users),
                'unique_items': group_recos['movie_id'].nunique(),
                'popularity_bias': self._calculate_popularity_bias(group_recos)
            }
        
        return results
    
    def analyze_exposure_fairness(self):
        """Analyze if all items get fair exposure"""
        item_exposure = self.recommendations.groupby('movie_id').size()
        
        # Gini coefficient for exposure inequality
        gini = self._calculate_gini(item_exposure.values)
        
        # Long-tail analysis
        top_10_percent = int(len(item_exposure) * 0.1)
        top_10_exposure = item_exposure.nlargest(top_10_percent).sum()
        total_exposure = item_exposure.sum()
        
        return {
            'gini_coefficient': gini,
            'top_10_percent_share': top_10_exposure / total_exposure,
            'unique_items_recommended': len(item_exposure),
            'zero_exposure_items': self._count_zero_exposure_items()
        }
    
    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n
```

#### 6.2 Security Implementation
```python
# backend/security/rate_limiter.py
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.blocked_ips = set()
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        """Check if request should be allowed"""
        if client_ip in self.blocked_ips:
            return False
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            # Potential abuse, temporary block
            self.blocked_ips.add(client_ip)
            asyncio.create_task(self._unblock_after_delay(client_ip, 300))
            return False
        
        self.requests[client_ip].append(now)
        return True
    
    async def _unblock_after_delay(self, client_ip: str, seconds: int):
        await asyncio.sleep(seconds)
        self.blocked_ips.discard(client_ip)

# backend/security/anomaly_detector.py
class AnomalyDetector:
    def __init__(self):
        self.user_patterns = defaultdict(dict)
    
    def detect_rating_anomaly(self, user_id: int, ratings: List[int]) -> bool:
        """Detect potential rating manipulation"""
        if user_id not in self.user_patterns:
            # Initialize user pattern
            self.user_patterns[user_id] = {
                'avg_rating': np.mean(ratings),
                'std_rating': np.std(ratings),
                'rating_velocity': len(ratings)
            }
            return False
        
        pattern = self.user_patterns[user_id]
        
        # Check for suspicious patterns
        # 1. Sudden spike in rating velocity
        if len(ratings) > pattern['rating_velocity'] * 10:
            return True
        
        # 2. Unusual rating pattern (all same rating)
        if len(set(ratings)) == 1 and len(ratings) > 5:
            return True
        
        # 3. Statistical anomaly
        z_scores = [(r - pattern['avg_rating']) / pattern['std_rating'] 
                    for r in ratings]
        if any(abs(z) > 3 for z in z_scores):
            return True
        
        return False
```

## Directory Structure

```
lens/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration
│       ├── cd.yml                 # Continuous Deployment
│       ├── retrain.yml            # Scheduled retraining
│       └── probe.yml              # Scheduled API probing
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI application
│   │   ├── middleware.py         # Custom middleware
│   │   └── state.py              # Application state
│   ├── routers/
│   │   ├── recommendation_router.py
│   │   ├── model_router.py
│   │   └── monitoring_router.py
│   ├── services/
│   │   ├── recommendation_service.py
│   │   ├── ab_testing.py
│   │   ├── provenance.py
│   │   └── kafka_service.py
│   ├── security/
│   │   ├── rate_limiter.py
│   │   └── anomaly_detector.py
│   └── config/
│       └── settings.py
├── ml/
│   ├── models/
│   │   ├── base.py              # Base model interface
│   │   ├── popularity.py        # Popularity model
│   │   ├── als.py               # ALS implementation
│   │   └── neural_cf.py         # Neural CF
│   ├── data_split.py            # Train/test splitting
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Offline evaluation
│   ├── drift.py                 # Drift detection
│   └── fairness.py              # Fairness analysis
├── stream/
│   ├── consumer.py              # Kafka consumer
│   ├── producer.py              # Kafka producer
│   └── schemas/
│       ├── avro/                # Avro schemas
│       └── validation.py        # Schema validation
├── docker/
│   ├── recommender.Dockerfile   # API service
│   ├── ingestor.Dockerfile      # Stream ingestor
│   └── trainer.Dockerfile       # Training job
├── infra/
│   ├── kafka.env.example        # Kafka configuration
│   ├── cloud-run.yaml           # Cloud Run config
│   └── terraform/               # Infrastructure as Code
├── scripts/
│   ├── probe.py                 # API probing script
│   ├── online_metric.py         # Online success calculation
│   └── setup_kafka.py           # Kafka topic creation
├── tests/
│   ├── test_api.py
│   ├── test_models.py
│   ├── test_kafka.py
│   └── test_fairness.py
├── model_registry/              # Versioned models
│   ├── popularity/
│   ├── als/
│   └── neural_cf/
├── data/
│   ├── snapshots/               # Versioned data snapshots
│   └── ml-1m/                   # MovieLens data
├── docs/
│   ├── architecture.md
│   ├── api-contract.md
│   ├── runbook.md
│   └── model-cards/
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Dev dependencies
├── requirements-train.txt       # Training dependencies
├── docker-compose.yml           # Local development
├── Makefile                     # Common commands
└── README.md                    # Project documentation
```

## Updated Timeline & Current Status

### Completed Items ✅
- [x] Docker containerization for all services
- [x] Kafka infrastructure setup (local, not Confluent)
- [x] Schema Registry with all 5 schemas
- [x] Train 3 models (Popularity, CF, ALS)
- [x] Basic API deployment
- [x] Frontend application
- [x] Prometheus & Grafana monitoring
- [x] Integration test suite
- [x] Redis caching

### In Progress 🔄
- [ ] Kafka consumer implementation (infrastructure ready)
- [ ] Publishing events from API
- [ ] GitHub Actions improvements

### High Priority Tasks 🎯

#### Immediate (Next 3-5 days)
1. **Kafka Integration**
   - [ ] Implement Kafka producer in recommendation service
   - [ ] Add consumer for processing events
   - [ ] Test event flow end-to-end

2. **Model Evaluation**
   - [ ] Implement NDCG@K and HR@K metrics
   - [ ] Create train/val/test splits
   - [ ] Add model comparison framework

#### Week 1 Focus
3. **CI/CD Pipeline**
   - [ ] Add comprehensive tests (target 70% coverage)
   - [ ] Container registry integration
   - [ ] Automated deployment to cloud

4. **Stream Processing**
   - [ ] Implement stream ingestion to storage
   - [ ] Add real-time event processing

#### Week 2 Focus
5. **A/B Testing**
   - [ ] Implement experiment assignment logic
   - [ ] Add online success metrics
   - [ ] Create statistical analysis

6. **MLOps**
   - [ ] Automated model retraining
   - [ ] Drift detection
   - [ ] Model versioning

#### Week 3 Focus
7. **Production Readiness**
   - [ ] Rate limiting
   - [ ] Authentication
   - [ ] Error handling improvements
   - [ ] Performance optimization

8. **Fairness & Security**
   - [ ] Implement fairness metrics
   - [ ] Security analysis
   - [ ] Bias detection

## Key Deliverables by Milestone (Updated Status)

### Milestone 1: Team Formation & Proposal ✅
- [x] Team contract with roles
- [x] Architecture diagram
- [x] Risk register
- [x] GitHub repo setup

### Milestone 2: Kafka & Baselines 🔄
- [x] Kafka infrastructure ready
- [ ] Kafka event flow implementation needed
- [x] 3 models trained (exceeds requirement)
- [ ] API deployed to cloud (local only currently)
- [x] Basic probe working (test_integration.py)

### Milestone 3: Pipeline Quality ❌
- [ ] Offline evaluation metrics (NDCG, HR)
- [ ] Online metrics calculation
- [ ] Comprehensive CI/CD pipeline
- [ ] 70% test coverage
- [ ] Drift detection

### Milestone 4: Monitoring & Experiments ⚠️
- [x] Monitoring infrastructure (Prometheus/Grafana)
- [ ] Actual dashboards with metrics
- [ ] A/B test implementation
- [ ] Automated retraining
- [ ] Provenance tracking

### Milestone 5: Responsible ML ❌
- [ ] Fairness analysis
- [ ] Security implementation
- [ ] Feedback loop detection
- [ ] Final demo video
- [ ] Reflection document

## Risk Mitigation

1. **Kafka Costs**: Use Confluent Cloud free tier (400GB/month)
2. **Cloud Costs**: Cloud Run scales to zero, use scheduling
3. **Model Storage**: Use free tier object storage (5GB)
4. **Monitoring**: Grafana Cloud free tier (10k series)
5. **CI/CD Minutes**: GitHub Actions 2000 min/month free

## Immediate Next Steps (Priority Order)

### This Week's Focus

1. **Complete Kafka Integration** (2-3 days)
   ```bash
   # Already have schemas registered
   # Need to: 
   - Add producer logic to backend/services/kafka_service.py
   - Implement consumer in backend/stream/consumer.py
   - Test with real events
   ```

2. **Add Model Evaluation** (1-2 days)
   ```python
   # In backend/scripts/evaluate_models.py
   - Implement NDCG@K and HR@K
   - Create proper data splits
   - Compare all 3 models
   ```

3. **Deploy to Cloud** (1 day)
   ```bash
   # Choose platform (Cloud Run recommended)
   - Update GitHub Actions for deployment
   - Set up container registry
   - Configure environment variables
   ```

4. **Improve Testing** (2-3 days)
   ```bash
   # Add tests for:
   - Model predictions
   - API endpoints
   - Kafka integration
   - Target 70% coverage
   ```

### Quick Wins Available Now

1. **Fix Backend Metrics Endpoint**
   ```python
   # Add to backend/app/main.py:
   @app.get("/metrics")
   async def metrics():
       return Response(generate_latest())
   ```

2. **Add Basic A/B Test Structure**
   ```python
   # Create backend/services/ab_testing.py
   # Simple hash-based assignment
   ```

3. **Create Grafana Dashboard**
   - Import dashboard JSON
   - Configure Prometheus datasource
   - Add key metrics

## Risk Mitigation (Updated)

1. **Kafka Costs**: Currently using local Kafka (free)
   - Consider Confluent Cloud for production ($0 for dev tier)
2. **Cloud Costs**: 
   - Cloud Run: ~$0 for low traffic
   - Set up billing alerts
3. **Time Constraints**:
   - Focus on core requirements first
   - Nice-to-haves can be simulated/mocked

## Recommended Development Order

1. **Day 1-2**: Kafka producer/consumer
2. **Day 3**: Model evaluation metrics
3. **Day 4**: Cloud deployment
4. **Day 5-7**: Testing & CI/CD
5. **Week 2**: A/B testing & monitoring
6. **Week 3**: Security & fairness

The system architecture is solid - focus on connecting the pieces!