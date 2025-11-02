#!/usr/bin/env python3
"""
Simple Model Benchmarking Script
Compares models across key metrics:
1. Offline ranking metrics (HR@K, NDCG@K)
2. Training cost (time/CPU)
3. Inference latency/throughput
4. Model size
"""
import sys
import os
import time
import pickle
import json
# import psutil  # Optional for memory tracking
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.models_impl import PopularityModel, CollaborativeFilteringModel, ALSModel
from config.settings import get_settings

settings = get_settings()

def load_data():
    """Load MovieLens data"""
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

def measure_training(model_class, train_df, **kwargs):
    """Measure training time and resources"""
    # Initialize model
    model = model_class(**kwargs)

    # Train model
    start_time = time.time()

    # Use synchronous version for simplicity
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(model.train(train_df))

    training_time = time.time() - start_time

    # Model size on disk
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name

    disk_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
    os.unlink(tmp_path)

    # Estimate memory usage based on data structures
    memory_estimate_mb = 0
    if hasattr(model, 'user_item_matrix'):
        memory_estimate_mb += model.user_item_matrix.data.nbytes / 1024 / 1024
    elif hasattr(model, 'popular_movies'):
        memory_estimate_mb += len(model.popular_movies) * 16 / 1024 / 1024  # Rough estimate
    elif hasattr(model, 'model'):
        memory_estimate_mb += 50  # ALS model estimate

    return {
        'model': model,
        'training_time_seconds': training_time,
        'memory_used_mb': memory_estimate_mb,
        'disk_size_mb': disk_size_mb
    }

def measure_inference(model, test_users, k=10):
    """Measure inference latency"""
    latencies = []

    # Sample 100 users
    sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for user_id in sample_users:
        start = time.time()
        loop.run_until_complete(model.predict(int(user_id), k=k))
        latencies.append(time.time() - start)

    # Calculate throughput
    batch_start = time.time()
    for user_id in sample_users[:50]:  # 50 users
        loop.run_until_complete(model.predict(int(user_id), k=k))
    batch_time = time.time() - batch_start
    throughput = 50 / batch_time

    return {
        'avg_latency_ms': np.mean(latencies) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
        'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        'throughput_users_per_sec': throughput
    }

def calculate_ranking_metrics(model, test_df, k=10):
    """Calculate ranking metrics"""
    # Group by user
    user_items = test_df.groupby('user_id')['movie_id'].apply(list).to_dict()

    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []

    # Sample users
    test_users = list(user_items.keys())
    sample_users = np.random.choice(test_users, min(200, len(test_users)), replace=False)

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for user_id in sample_users:
        try:
            # Get recommendations
            rec_ids = loop.run_until_complete(model.predict(int(user_id), k=k))
            if not rec_ids:
                continue

            actual_items = set(user_items[user_id])
            if not actual_items:
                continue

            # Calculate metrics
            hits = len(set(rec_ids) & actual_items)

            # Hit Rate
            hit_rates.append(1 if hits > 0 else 0)

            # Precision
            precision = hits / k
            precisions.append(precision)

            # Recall
            recall = hits / len(actual_items)
            recalls.append(recall)

            # NDCG
            dcg = sum([1 / np.log2(i + 2) for i, item_id in enumerate(rec_ids) if item_id in actual_items])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(actual_items)))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

        except Exception as e:
            continue

    return {
        'hit_rate@10': np.mean(hit_rates) if hit_rates else 0,
        'precision@10': np.mean(precisions) if precisions else 0,
        'recall@10': np.mean(recalls) if recalls else 0,
        'ndcg@10': np.mean(ndcgs) if ndcgs else 0
    }

def main():
    print("="*70)
    print("MODEL BENCHMARKING REPORT")
    print("="*70)

    # Load data
    ratings_df = load_data()

    # Split data
    ratings_df = ratings_df.sort_values('timestamp')
    n = len(ratings_df)
    train_df = ratings_df[:int(0.8 * n)]
    test_df = ratings_df[int(0.8 * n):]

    print(f"\nData split: Train={len(train_df)}, Test={len(test_df)}")

    # Models to test
    models_config = [
        ('Popularity', PopularityModel, {}),
        ('Collaborative', CollaborativeFilteringModel, {'neighborhood_size': 50}),
        ('ALS', ALSModel, {'factors': 50, 'regularization': 0.01, 'iterations': 10})
    ]

    results = {}

    # Benchmark each model
    for model_name, model_class, kwargs in models_config:
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_name}")
        print('='*50)

        # 1. Training metrics
        print("1. Training...")
        train_results = measure_training(model_class, train_df, **kwargs)
        model = train_results['model']

        print(f"   Time: {train_results['training_time_seconds']:.2f}s")
        print(f"   Memory: {train_results['memory_used_mb']:.1f} MB")
        print(f"   Size: {train_results['disk_size_mb']:.2f} MB")

        # 2. Ranking metrics
        print("2. Evaluating...")
        ranking_metrics = calculate_ranking_metrics(model, test_df)

        print(f"   HR@10: {ranking_metrics['hit_rate@10']:.3f}")
        print(f"   P@10: {ranking_metrics['precision@10']:.3f}")
        print(f"   R@10: {ranking_metrics['recall@10']:.3f}")
        print(f"   NDCG@10: {ranking_metrics['ndcg@10']:.3f}")

        # 3. Inference performance
        print("3. Inference performance...")
        test_users = test_df['user_id'].unique()[:500]
        inference_metrics = measure_inference(model, test_users)

        print(f"   Avg latency: {inference_metrics['avg_latency_ms']:.1f} ms")
        print(f"   P95 latency: {inference_metrics['p95_latency_ms']:.1f} ms")
        print(f"   Throughput: {inference_metrics['throughput_users_per_sec']:.1f} users/sec")

        # Store results
        results[model_name] = {
            'training': train_results,
            'ranking': ranking_metrics,
            'inference': inference_metrics
        }

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    # 1. Ranking comparison
    print("\n1. RANKING METRICS (Test Set)")
    print("-"*60)
    print(f"{'Model':<15} {'HR@10':<10} {'P@10':<10} {'R@10':<10} {'NDCG@10':<10}")
    print("-"*60)

    for model_name, res in results.items():
        metrics = res['ranking']
        print(f"{model_name:<15} "
              f"{metrics['hit_rate@10']:<10.3f} "
              f"{metrics['precision@10']:<10.3f} "
              f"{metrics['recall@10']:<10.3f} "
              f"{metrics['ndcg@10']:<10.3f}")

    # 2. Training cost
    print("\n2. TRAINING COST")
    print("-"*60)
    print(f"{'Model':<15} {'Time (s)':<12} {'Memory (MB)':<12} {'Size (MB)':<12}")
    print("-"*60)

    for model_name, res in results.items():
        train = res['training']
        print(f"{model_name:<15} "
              f"{train['training_time_seconds']:<12.2f} "
              f"{train['memory_used_mb']:<12.1f} "
              f"{train['disk_size_mb']:<12.2f}")

    # 3. Inference performance
    print("\n3. INFERENCE PERFORMANCE")
    print("-"*60)
    print(f"{'Model':<15} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Users/sec':<12}")
    print("-"*60)

    for model_name, res in results.items():
        inf = res['inference']
        print(f"{model_name:<15} "
              f"{inf['avg_latency_ms']:<10.1f} "
              f"{inf['p95_latency_ms']:<10.1f} "
              f"{inf['throughput_users_per_sec']:<12.1f}")

    # 4. Cost estimation
    print("\n4. ESTIMATED COST (per 1M recommendations)")
    print("-"*60)

    for model_name, res in results.items():
        # Assume $0.10 per CPU-hour
        cpu_time_per_rec = res['inference']['avg_latency_ms'] / 1000
        cost_per_million = cpu_time_per_rec * 1_000_000 / 3600 * 0.10
        print(f"{model_name:<15} ${cost_per_million:.2f}")

    # Winner analysis
    print("\n" + "="*70)
    print("WINNERS")
    print("="*70)

    # Find best for each category
    best_accuracy = max(results.items(), key=lambda x: x[1]['ranking']['ndcg@10'])[0]
    best_speed = min(results.items(), key=lambda x: x[1]['inference']['avg_latency_ms'])[0]
    best_size = min(results.items(), key=lambda x: x[1]['training']['disk_size_mb'])[0]

    print(f"🏆 Best Accuracy (NDCG@10): {best_accuracy}")
    print(f"⚡ Best Speed: {best_speed}")
    print(f"💾 Most Compact: {best_size}")

    # Save results
    with open('benchmark_results.json', 'w') as f:
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        clean_results = convert_types(results)
        json.dump(clean_results, f, indent=2)

    print(f"\n📊 Results saved to benchmark_results.json")

if __name__ == "__main__":
    main()
