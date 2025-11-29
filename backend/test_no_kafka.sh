#!/bin/bash
# Test that the app can start without Kafka

echo "Testing startup with KAFKA_ENABLED=false..."

export KAFKA_ENABLED=false
export SKIP_STARTUP_DOWNLOADS=true
export PORT=8080

# Start server in background
python run.py &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:8080/healthz

# Test debug endpoint
echo -e "\n\nTesting debug services endpoint..."
curl -f http://localhost:8080/debug/services

# Kill server
kill $SERVER_PID

echo -e "\n\n✅ Test complete!"