#!/usr/bin/env python3
"""
Benchmark existing trained models without retraining
"""
import sys
import os
import time
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

settings = get_settings()

def load_data():
    """Load MovieLens data for testing"""
    data_path = settings.data_path
    
    # Load ratings
    ratings_path = data_path / "ratings.dat"
    ratings_df = pd.read_csv(
        ratings_path,
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    print(f"Loaded {len(ratings_df)} ratings")
    
    return ratings_df

def load_trained_model(model_name, version="latest"):
    """Load existing trained model from registry"""
    model_registry = settings.model_registry_path
    
    # Get version
    if version == "latest":
        latest_file = model_registry / model_name / "latest.txt"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                version = f.read().strip()
        else:
            # Find the highest version
            versions = [d.name for d in (model_registry / model_name).iterdir() if d.is_dir() and d.name != "latest.txt"]
            version = sorted(versions)[-1] if versions else "0.1.0"
    
    # Load model
    model_path = model_registry / model_name / version / "model.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded {model_name} v{version}")
    return model

def measure_inference_performance(model, test_users, k=10):
    """Measure inference latency and throughput"""
    latencies = []
    
    # Sample users
    sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Warmup
    for _ in range(5):
        loop.run_until_complete(model.predict(int(sample_users[0]), k=k))
    
    # Measure latencies
    for user_id in sample_users:
        start = time.time()
        loop.run_until_complete(model.predict(int(user_id), k=k))
        latencies.append(time.time() - start)
    
    # Throughput test
    batch_start = time.time()
    for user_id in sample_users[:50]:
        loop.run_until_complete(model.predict(int(user_id), k=k))
    batch_time = time.time() - batch_start
    throughput = 50 / batch_time if batch_time > 0 else 0
    
    return {
        'avg_latency_ms': np.mean(latencies) * 1000,
        'p50_latency_ms': np.percentile(latencies, 50) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
        'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        'throughput_users_per_sec': throughput
    }

def calculate_model_size(model):
    """Calculate model size"""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name
    
    disk_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
    os.unlink(tmp_path)
    
    return disk_size_mb

def main():
    print("="*70)
    print("BENCHMARKING EXISTING MODELS")
    print("="*70)
    
    # Load test data
    ratings_df = load_data()
    test_df = ratings_df[int(0.85 * len(ratings_df)):]  # Last 15% as test
    test_users = test_df['user_id'].unique()[:1000]
    
    # Models to benchmark
    models = ['popularity', 'collaborative', 'als']
    results = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_name}")
        print('='*50)
        
        # Load model
        model = load_trained_model(model_name)
        if model is None:
            print(f"Skipping {model_name} - not found")
            continue
        
        # Get model info
        if hasattr(model, 'metadata'):
            print(f"Training time: {model.metadata.get('training_time_seconds', 'N/A'):.2f}s")
            print(f"Trained at: {model.metadata.get('trained_at', 'N/A')}")
        
        # Measure inference performance
        print("\nMeasuring inference performance...")
        perf = measure_inference_performance(model, test_users)
        
        # Model size
        size_mb = calculate_model_size(model)
        
        results[model_name] = {
            'inference': perf,
            'size_mb': size_mb,
            'metadata': model.metadata if hasattr(model, 'metadata') else {}
        }
        
        print(f"Size: {size_mb:.2f} MB")
        print(f"Avg latency: {perf['avg_latency_ms']:.1f} ms")
        print(f"P95 latency: {perf['p95_latency_ms']:.1f} ms")
        print(f"Throughput: {perf['throughput_users_per_sec']:.1f} users/sec")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<15} {'Size (MB)':<10} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Users/sec':<12}")
    print("-"*60)
    
    for model_name, res in results.items():
        inf = res['inference']
        print(f"{model_name:<15} "
              f"{res['size_mb']:<10.2f} "
              f"{inf['avg_latency_ms']:<10.1f} "
              f"{inf['p95_latency_ms']:<10.1f} "
              f"{inf['throughput_users_per_sec']:<12.1f}")
    
    # Save results
    with open('benchmark_results_existing.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to benchmark_results_existing.json")

if __name__ == "__main__":
    main()