# MovieLens Recommendation System

A cloud-native movie recommendation system with real-time ML capabilities, built following enterprise patterns from the insurance-rag-app.

## Project Structure

```
lens/
├── frontend/          # Next.js frontend application
│   ├── src/          # React components and logic
│   ├── package.json  # Frontend dependencies
│   └── README.md     # Frontend documentation
├── backend/          # FastAPI backend application
│   ├── app/          # Main application code
│   ├── services/     # Business logic services
│   ├── routers/      # API endpoints
│   ├── scripts/      # Backend utility scripts
│   ├── requirements.txt
│   └── README.md     # Backend documentation
├── schemas/          # Kafka event schemas (Avro)
│   ├── *.avsc        # Schema definitions
│   └── register_schemas.py
├── scripts/          # Project-level utility scripts
│   ├── probe.py      # API health probe
│   ├── fix-github-permissions.sh
│   └── grant-bucket-access.sh
├── docs/             # Additional documentation
│   ├── architecture-diagram.md
│   ├── architecture-rationale.md
│   ├── api-slos.md
│   ├── testing-guide.md
│   └── technical-proposal.md
├── docker-compose.yml      # Full stack deployment
└── docker-compose-minimal.yml  # Minimal deployment
```

## Architecture Overview

### Frontend (Next.js + React)
- Server-side rendering for performance
- React Query for state management
- Tailwind CSS for styling
- Real-time metrics visualization
- Configurable feature flags

### Backend (FastAPI + Python)
- RESTful API with automatic documentation
- ML model management and versioning
- Kafka integration for streaming
- Prometheus metrics
- Optional authentication

### Infrastructure
- Docker containerization
- Kubernetes-ready
- Cloud-agnostic deployment
- Monitoring and observability

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional)
- Kafka (optional)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd lens
   ```

2. **Start Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python scripts/download_movielens.py  # Download dataset
   python scripts/train_simple_models.py  # Train initial models
   uvicorn app.main:app --reload
   ```

3. **Start Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Register Kafka schemas
cd schemas
python register_schemas.py

# Verify all services are running
python test_integration.py
```

### Service Access Points

Once all services are running, you can access:

| Service | URL | Description | Credentials |
|---------|-----|-------------|-------------|
| Frontend | http://localhost:3000 | Main web application | None |
| Backend API | http://localhost:8000 | REST API endpoints | None |
| API Documentation | http://localhost:8000/docs | Interactive API docs (Swagger) | None |
| Kafka UI | http://localhost:8080 | Monitor Kafka topics and messages | None |
| Schema Registry | http://localhost:8081 | Manage Avro schemas | None |
| Prometheus | http://localhost:9090 | Metrics collection | None |
| Grafana | http://localhost:3001 | Metrics visualization dashboards | Username: `admin`<br>Password: `admin` |

**Note:** On first login to Grafana, you may be prompted to change the default password. You can skip this for local development.

## Key Features

### Recommendation Engine
- Multiple ML models (Popularity, Collaborative Filtering, ALS)
- Real-time model switching
- A/B testing capabilities
- Personalized recommendations

### Streaming Pipeline
- Kafka integration for real-time events
- Event-driven model updates
- Online learning capabilities

### Monitoring & Observability
- Prometheus metrics
- Real-time dashboards
- Performance tracking
- Error monitoring

### Security
- Optional authentication (Firebase/JWT)
- Rate limiting
- Input validation
- CORS configuration

## Configuration

### Environment Variables

Both frontend and backend use environment variables for configuration:

**Frontend (.env.local)**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENABLE_MONITORING=true
```

**Backend (.env)**
```env
DEBUG=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
DEFAULT_MODEL=popularity
```

### Feature Flags

Enable/disable features via environment variables:
- Authentication
- Monitoring dashboards
- A/B testing
- Kafka streaming

## Development

### Code Style
- Python: Black + isort
- JavaScript: ESLint + Prettier
- Git hooks with pre-commit

### Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Documentation
- API documentation: Auto-generated at `/docs`
- Code documentation: Inline docstrings
- Architecture docs: See `/docs` folder

## Deployment

### Cloud Platforms

The system is designed to run on:
- Google Cloud Run
- Azure Container Apps
- AWS ECS/Fargate
- Kubernetes

### CI/CD

GitHub Actions workflows for:
- Testing
- Building Docker images
- Deploying to cloud
- Running probes

## Monitoring

### Metrics
- Request latency (P50, P95, P99)
- Model performance
- Cache hit rates
- Error rates

### Dashboards
- Built-in monitoring UI
- Grafana integration
- Custom metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file

## Acknowledgments

- Architecture patterns inspired by production insurance-rag-app
- MovieLens dataset from GroupLens Research
- Built with FastAPI and Next.js