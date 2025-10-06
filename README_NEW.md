# MovieLens Recommender System

A cloud-native movie recommendation system built with FastAPI, Next.js, and Kafka for real-time event streaming.

## Project Structure

```
lens/
├── backend/          # FastAPI recommendation service
├── frontend/         # Next.js user interface
├── docker/          # Docker configurations
├── ml/              # Machine learning models
├── stream/          # Kafka streaming components
├── .github/         # CI/CD workflows
└── docs/            # Documentation
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker
- Kafka (Confluent Cloud or local)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker Setup

```bash
# Build and run with docker-compose
docker-compose up --build
```

## API Endpoints

- `GET /healthz` - Health check
- `GET /recommend/{user_id}` - Get movie recommendations
- `GET /models` - List available models
- `GET /metrics` - Prometheus metrics

## Testing

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test
```

## CI/CD

The project uses GitHub Actions for:
- Automated testing on push/PR
- Docker image building
- Deployment to cloud platforms
- Scheduled model retraining

## Data

Using MovieLens 1M dataset:
- 6,040 users
- 3,883 movies
- 1,000,209 ratings

## Models

Currently implemented:
- Popularity-based recommendations
- Collaborative Filtering (planned)
- ALS (planned)
- Neural CF (planned)

## Architecture

- **API Service**: FastAPI with async support
- **Stream Processing**: Kafka for real-time events
- **Model Training**: Scheduled batch jobs
- **Monitoring**: Prometheus + Grafana
- **A/B Testing**: Built-in experimentation framework

## License

MIT License