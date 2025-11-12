# Batch Trainer Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    schedule \
    mlflow \
    optuna

# Copy application code
COPY backend/ .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directory for model registry
RUN mkdir -p /app/model_registry

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Starting Batch Trainer Service..."\n\
python -m pipeline.batch_trainer' > /app/entrypoint.sh \
&& chmod +x /app/entrypoint.sh

# Run the batch trainer
CMD ["/app/entrypoint.sh"]