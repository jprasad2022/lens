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
│   ├── requirements.txt
│   └── README.md     # Backend documentation
├── infra/            # Infrastructure as code
│   ├── docker-compose.yml
│   ├── k8s/          # Kubernetes manifests
│   └── terraform/    # Cloud infrastructure
├── scripts/          # Utility scripts
│   ├── setup.sh      # Initial setup script
│   └── deploy.sh     # Deployment script
└── docs/             # Additional documentation
    ├── architecture.md
    ├── api.md
    └── deployment.md
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
   cp .env.example .env
   uvicorn app.main:app --reload
   ```

3. **Start Frontend**
   ```bash
   cd frontend
   npm install
   cp .env.example .env.local
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
docker-compose up -d
```

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