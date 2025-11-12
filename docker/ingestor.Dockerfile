# Stream Ingestor Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    confluent-kafka \
    pyarrow \
    boto3 \
    jsonschema

# Copy application code
COPY backend/ .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p /app/data/snapshots

# Run the ingestor
CMD ["python", "-m", "stream.ingestor"]