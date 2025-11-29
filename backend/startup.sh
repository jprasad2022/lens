#!/bin/bash
# Startup script for Cloud Run

echo "=== LENS API Startup Script ==="
echo "Environment: Cloud Run"
echo "PORT: ${PORT:-8000}"

# Pre-download data and models
echo "Pre-downloading data and models..."
python3 -c "
import os
os.environ['SKIP_STARTUP_DOWNLOADS'] = 'true'
try:
    from utils.download_data import download_movielens_data
    print('Downloading MovieLens data...')
    download_movielens_data()
    print('✅ Data download complete')
except Exception as e:
    print(f'⚠️  Data download failed: {e}')

try:
    from utils.download_models import download_model_registry
    print('Downloading models...')
    download_model_registry()
    print('✅ Model download complete')
except Exception as e:
    print(f'⚠️  Model download failed: {e}')
"

echo "Starting application..."
exec python3 run.py