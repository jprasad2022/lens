#!/bin/bash
# Setup script for Redpanda development environment

echo "🔴 Setting up Redpanda for MovieLens Recommendation System..."
echo "=================================================="

# Step 1: Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Step 2: Stop any existing containers
echo "📦 Stopping existing containers..."
docker compose -f docker-compose-redpanda.yml down 2>/dev/null

# Step 3: Copy environment file
echo "📄 Setting up environment configuration..."
cp backend/.env.redpanda backend/.env

# Step 4: Start Redpanda
echo "🚀 Starting Redpanda services..."
docker compose -f docker-compose-redpanda.yml up -d redpanda redis

# Wait for Redpanda to be healthy
echo "⏳ Waiting for Redpanda to be ready..."
timeout 60s bash -c 'until docker compose -f docker-compose-redpanda.yml ps redpanda | grep -q "healthy"; do sleep 2; done'

if [ $? -eq 0 ]; then
    echo "✅ Redpanda is healthy!"
else
    echo "❌ Redpanda failed to start. Check logs with: docker compose -f docker-compose-redpanda.yml logs redpanda"
    exit 1
fi

# Step 5: Create topics
echo "📋 Creating Kafka topics..."
docker exec -it redpanda rpk topic create \
    team1.watch \
    team1.rate \
    team1.reco_requests \
    team1.reco_responses \
    team1.user_interactions \
    --brokers localhost:19092

# Step 6: List topics to verify
echo "📋 Verifying topics..."
docker exec -it redpanda rpk topic list --brokers localhost:19092

# Step 7: Start Redpanda Console
echo "🎨 Starting Redpanda Console..."
docker compose -f docker-compose-redpanda.yml up -d redpanda-console

echo ""
echo "✅ Redpanda setup complete!"
echo "=================================================="
echo "📊 Redpanda Console: http://localhost:8080"
echo "🔌 Kafka Broker: localhost:19092"
echo "📝 Schema Registry: localhost:18081"
echo ""
echo "🏃 To start your backend:"
echo "   cd backend"
echo "   python main.py"
echo ""
echo "🏃 To start your frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "📖 To view Redpanda logs:"
echo "   docker compose -f docker-compose-redpanda.yml logs -f redpanda"