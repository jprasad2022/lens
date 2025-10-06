# Architecture Diagram

## System Components and Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   Users                                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Load Balancer (Cloud LB)                           │
└──────────────┬──────────────────────────────────────┬──────────────────────┘
               │                                      │
               ▼                                      ▼
┌──────────────────────────────┐      ┌──────────────────────────────────────┐
│      Frontend (Next.js)       │      │         Backend API (FastAPI)        │
│   ┌────────────────────┐      │      │   ┌────────────────────────────┐   │
│   │  React Components  │      │      │   │    Recommendation Router   │   │
│   │  - Recommendations │      │◀────▶│   │    - GET /recommend/{id}   │   │
│   │  - Monitoring      │      │      │   │    - POST /feedback        │   │
│   │  - A/B Testing     │      │      │   └────────────┬───────────────┘   │
│   └────────────────────┘      │      │                │                     │
│   ┌────────────────────┐      │      │   ┌────────────▼───────────────┐   │
│   │   State Management │      │      │   │      Model Service         │   │
│   │   - React Query    │      │      │   │  - Model Loading/Caching  │   │
│   │   - Context API    │      │      │   │  - Inference Pipeline     │   │
│   └────────────────────┘      │      │   │  - Feature Engineering    │   │
└──────────────────────────────┘      │   └────────────┬───────────────┘   │
                                       │                │                     │
                                       └────────────────┼─────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼─────────────────────┐
                    │                                  ▼                      │
                    │          ┌─────────────────────────────────┐           │
                    │          │      Model Registry (GCS)       │           │
                    │          │   - Versioned Models (.pkl)     │           │
                    │          │   - Metadata (JSON)             │           │
                    │          │   - Model Cards                 │           │
                    │          └─────────────────────────────────┘           │
                    │                                                        │
┌───────────────────┼────────────────────────────────────────────────────────┼──┐
│                   ▼                                                        ▼  │
│  ┌─────────────────────────────┐              ┌─────────────────────────────┐ │
│  │      Kafka Cluster          │              │      Redis Cache            │ │
│  │  Topics:                    │              │  - Recommendation Cache     │ │
│  │  - {team}.watch             │              │  - User Features            │ │
│  │  - {team}.rate              │              │  - Hot Models               │ │
│  │  - {team}.reco_requests     │              │  TTL: 1 hour                │ │
│  │  - {team}.reco_responses    │              └─────────────────────────────┘ │
│  └──────────┬──────────────────┘                                             │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────────────┐              ┌─────────────────────────────┐ │
│  │    Stream Processor         │              │    Batch Training           │ │
│  │  - Event Validation         │              │  - Scheduled Retraining     │ │
│  │  - Real-time Features       │              │  - Model Evaluation         │ │
│  │  - Data Snapshots           │◀────────────▶│  - Offline Metrics          │ │
│  └─────────────────────────────┘              └─────────────────────────────┘ │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Monitoring & Observability │
                        │   - Prometheus Metrics       │
                        │   - Grafana Dashboards      │
                        │   - CloudWatch/Stackdriver  │
                        │   - Alerts & Paging         │
                        └──────────────────────────────┘
```

## Component Details

### Frontend Layer
- **Technology**: Next.js 14 with React
- **Hosting**: Vercel / Cloud CDN
- **Features**: SSR, API caching, real-time updates

### API Layer
- **Technology**: FastAPI (Python 3.11)
- **Hosting**: Cloud Run / Container Apps
- **Scaling**: 1-10 instances auto-scaling
- **Load Balancing**: Cloud native LB with health checks

### Model Serving
- **Model Storage**: Google Cloud Storage / S3
- **Model Loading**: Lazy loading with LRU cache (3 models)
- **Inference**: In-process, CPU-based
- **Fallback**: Popularity model for failures

### Streaming Pipeline
- **Message Broker**: Confluent Kafka / Redpanda
- **Consumer**: Async Python with schema validation
- **Processing**: Micro-batching every 30 seconds
- **Storage**: Parquet files in object storage

### Data Storage
- **Feature Store**: Redis with 1-hour TTL
- **Model Registry**: Object storage with versioning
- **Data Warehouse**: BigQuery / Snowflake for analytics
- **Snapshots**: Daily exports for training

### Monitoring Stack
- **Metrics**: Prometheus + Grafana
- **Logging**: Cloud Logging with structured logs
- **Tracing**: OpenTelemetry (optional)
- **Alerting**: PagerDuty integration

## Data Flow

1. **User Request Flow**:
   - User → Frontend → API Gateway → Backend → Model Service → Response

2. **Event Streaming Flow**:
   - User Action → Frontend → Kafka → Stream Processor → Storage

3. **Model Training Flow**:
   - Storage → Batch Job → Training → Evaluation → Model Registry

4. **Monitoring Flow**:
   - All Components → Metrics → Prometheus → Grafana → Alerts

## Security Boundaries

- **Public**: Frontend, API endpoints
- **Private**: Model registry, Kafka, Redis
- **Restricted**: Training pipeline, data warehouse

## Deployment Strategy

- **Blue-Green**: For backend API updates
- **Canary**: For model deployments (5% → 50% → 100%)
- **Rolling**: For frontend updates
- **GitOps**: Infrastructure as code with Terraform