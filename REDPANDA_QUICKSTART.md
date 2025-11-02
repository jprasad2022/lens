# Redpanda Quick Start Guide

This guide helps you run the MovieLens recommendation system with Redpanda (free Kafka alternative).

## Prerequisites
- Docker Desktop installed and running
- Python 3.9+
- Node.js 16+

## Quick Setup (5 minutes)

### 1. Start Redpanda
```bash
# Run the setup script
./setup-redpanda.sh

# Or manually with docker compose
docker compose -f docker-compose-redpanda.yml up -d redpanda redis
```

### 2. Verify Redpanda is Running
```bash
# Test the connection
python test-redpanda.py

# Or check manually
docker exec -it redpanda rpk cluster health
```

### 3. Start Backend
```bash
cd backend

# Use Redpanda environment
cp .env.redpanda .env

# Install dependencies if needed
pip install -r requirements.txt

# Run backend
python main.py
```

### 4. Start Frontend
```bash
cd frontend

# Install dependencies if needed
npm install

# Run frontend
npm run dev
```

## Accessing Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Redpanda Console**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs

## Running Evaluations

### Online Evaluation with Redpanda
```bash
cd backend

# Run user simulation
python evaluation/run_online_eval.py

# View Kafka messages in console
# Go to http://localhost:8080 and browse topics
```

### Viewing Kafka Topics
```bash
# List all topics
docker exec -it redpanda rpk topic list

# View messages in a topic
docker exec -it redpanda rpk topic consume team1.reco_responses --format json

# Check topic details
docker exec -it redpanda rpk topic describe team1.reco_responses
```

## Troubleshooting

### Redpanda won't start
```bash
# Check logs
docker compose -f docker-compose-redpanda.yml logs redpanda

# Reset and start fresh
docker compose -f docker-compose-redpanda.yml down -v
docker compose -f docker-compose-redpanda.yml up -d
```

### Connection refused errors
```bash
# Make sure you're using the right port
# Local development: localhost:19092
# Docker internal: redpanda:9092

# Check if Redpanda is healthy
docker compose -f docker-compose-redpanda.yml ps
```

### Topics not found
```bash
# Create topics manually
docker exec -it redpanda rpk topic create team1.reco_requests
docker exec -it redpanda rpk topic create team1.reco_responses
docker exec -it redpanda rpk topic create team1.user_interactions
docker exec -it redpanda rpk topic create team1.watch
docker exec -it redpanda rpk topic create team1.rate
```

## Monitoring

Access Redpanda Console at http://localhost:8080 to:
- View all topics and messages
- Monitor consumer groups
- Check cluster health
- Browse message content

## Stopping Services

```bash
# Stop all services
docker compose -f docker-compose-redpanda.yml down

# Stop and remove all data
docker compose -f docker-compose-redpanda.yml down -v
```

## Environment Variables

Key environment variables for Redpanda:
```bash
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:19092  # For local dev
# or
KAFKA_BOOTSTRAP_SERVERS=redpanda:9092    # For Docker
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
USE_REDPANDA=true
TEAM_PREFIX=team1
```

## Cost Comparison

| Service | Cost | Setup Time | Compatibility |
|---------|------|------------|---------------|
| Confluent Cloud | ~$100+/month | 30 min | 100% Kafka |
| Redpanda Local | **FREE** | 5 min | 100% Kafka |
| Redpanda Cloud | Free tier available | 15 min | 100% Kafka |