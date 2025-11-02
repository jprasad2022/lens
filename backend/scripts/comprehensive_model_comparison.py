"""
Comprehensive Model Comparison
Evaluates models across multiple dimensions:
1. Offline ranking metrics (HR@K, NDCG@K)
2. Training cost (time/CPU)
3. Inference latency/throughput
4. Model size
"""
import sys
import os
import asyncio
import pickle
import json
import time
import psutil
import tracemalloc
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import concurrent.futures

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.models_impl import (
    PopularityModel,
    CollaborativeFilteringModel,
    ALSModel
)
from config.settings import get_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

class ModelBenchmark:
    """Comprehensive model benchmarking"""

    def __init__(self):
        self.results = {}

    async def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        logger.info(f"Loaded {len(ratings_df)} ratings")

        # Load movies
        movies_path = data_path / "movies.dat"
        movies_df = pd.read_csv(
            movies_path,
            sep='::',
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
        logger.info(f"Loaded {len(movies_df)} movies")

        return ratings_df, movies_df

    def measure_training_cost(self, model, train_df: pd.DataFrame) -> Dict[str, float]:
        """Measure training time and resource usage"""
        # Start monitoring
        process = psutil.Process()
        tracemalloc.start()
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Train model (using async)
        import asyncio
        asyncio.get_event_loop().run_until_complete(model.train(train_df))

        end_time = time.time()

        # Get resource usage
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'training_time_seconds': end_time - start_time,
            'cpu_usage_percent': (cpu_after - cpu_before) / 2,  # Average
            'memory_used_mb': mem_after - mem_before,
            'peak_memory_mb': peak / 1024 / 1024
        }

    def measure_model_size(self, model) -> Dict[str, float]:
        """Measure model size in memory and on disk"""
        # Serialize model to get disk size
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            pickle.dump(model, tmp)
            tmp_path = tmp.name

        disk_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        os.unlink(tmp_path)

        # Estimate memory size
        import sys

        def get_size(obj, seen=None):
            """Recursively find size of objects"""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()

            obj_id = id(obj)
            if obj_id in seen:
                return 0

            seen.add(obj_id)

            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])

            return size

        memory_size_mb = get_size(model) / 1024 / 1024

        return {
            'disk_size_mb': disk_size_mb,
            'memory_size_mb': memory_size_mb
        }

    async def measure_inference_performance(self, model, test_users: List[int], k: int = 10) -> Dict[str, float]:
        """Measure inference latency and throughput"""
        # Single prediction latency
        latencies = []

        # Measure 100 individual predictions
        sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)

        for user_id in sample_users:
            start = time.time()
            await model.predict(user_id, k=k)
            latencies.append(time.time() - start)

        # Batch prediction throughput
        batch_sizes = [1, 10, 50, 100]
        throughputs = {}

        for batch_size in batch_sizes:
            if batch_size > len(test_users):
                continue

            batch_users = np.random.choice(test_users, batch_size, replace=False)
            start = time.time()

            # Process batch
            for user_id in batch_users:
                await model.predict(user_id, k=k)

            elapsed = time.time() - start
            throughputs[f'batch_{batch_size}'] = batch_size / elapsed

        return {
            'avg_latency_ms': np.mean(latencies) * 1000,
            'p50_latency_ms': np.percentile(latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'throughput_users_per_sec': throughputs
        }

    async def calculate_ranking_metrics(self, model, test_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive ranking metrics"""
        # Group by user
        user_items = test_df.groupby('user_id')['movie_id'].apply(list).to_dict()

        metrics = {f'k_{k}': {} for k in k_values}

        for k in k_values:
            precisions = []
            recalls = []
            hit_rates = []
            ndcgs = []
            all_rec_items = set()

            # Sample users for evaluation
            test_users = list(user_items.keys())
            sample_users = np.random.choice(test_users, min(500, len(test_users)), replace=False)

            for user_id in sample_users:
                # Get recommendations
                try:
                    rec_ids = await model.predict(user_id, k=k)
                    if not rec_ids:
                        continue

                    all_rec_items.update(rec_ids)
                    actual_items = set(user_items.get(user_id, []))

                    if not actual_items:
                        continue

                    # Hit Rate
                    hits = len(set(rec_ids) & actual_items)
                    hit_rates.append(1 if hits > 0 else 0)

                    # Precision
                    precision = hits / k
                    precisions.append(precision)

                    # Recall
                    recall = hits / len(actual_items) if actual_items else 0
                    recalls.append(recall)

                    # NDCG
                    dcg = sum([1 / np.log2(i + 2) for i, item_id in enumerate(rec_ids) if item_id in actual_items])
                    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(actual_items)))])
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcgs.append(ndcg)

                except Exception as e:
                    continue

            metrics[f'k_{k}'] = {
                'hit_rate': np.mean(hit_rates) if hit_rates else 0,
                'precision': np.mean(precisions) if precisions else 0,
                'recall': np.mean(recalls) if recalls else 0,
                'ndcg': np.mean(ndcgs) if ndcgs else 0,
                'coverage': len(all_rec_items) / len(test_df['movie_id'].unique())
            }

        return metrics

    async def benchmark_model(self, model_class, model_name: str, train_df: pd.DataFrame,
                            val_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run complete benchmark for a model"""
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name}")
        print('='*60)

        results = {'model_name': model_name}

        # Initialize model
        model = model_class(**kwargs)

        # 1. Training Cost
        print("1. Measuring training cost...")
        training_cost = self.measure_training_cost(model, train_df)
        results['training_cost'] = training_cost
        print(f"   Training time: {training_cost['training_time_seconds']:.2f}s")
        print(f"   Peak memory: {training_cost['peak_memory_mb']:.2f} MB")

        # 2. Model Size
        print("2. Measuring model size...")
        model_size = self.measure_model_size(model)
        results['model_size'] = model_size
        print(f"   Disk size: {model_size['disk_size_mb']:.2f} MB")
        print(f"   Memory size: {model_size['memory_size_mb']:.2f} MB")

        # 3. Ranking Metrics
        print("3. Calculating ranking metrics...")
        ranking_metrics = await self.calculate_ranking_metrics(model, test_df)
        results['ranking_metrics'] = ranking_metrics
        for k, metrics in ranking_metrics.items():
            print(f"   {k}: HR={metrics['hit_rate']:.3f}, P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, NDCG={metrics['ndcg']:.3f}")

        # 4. Inference Performance
        print("4. Measuring inference performance...")
        test_users = test_df['user_id'].unique()[:1000]  # Sample users
        inference_perf = await self.measure_inference_performance(model, test_users)
        results['inference_performance'] = inference_perf
        print(f"   Avg latency: {inference_perf['avg_latency_ms']:.2f} ms")
        print(f"   P95 latency: {inference_perf['p95_latency_ms']:.2f} ms")

        # Save detailed results
        self.results[model_name] = results

        return results

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)

        # 1. Ranking Metrics Comparison
        print("\n1. RANKING METRICS (Test Set)")
        print("-"*80)
        print(f"{'Model':<20} {'HR@10':<10} {'P@10':<10} {'R@10':<10} {'NDCG@10':<10} {'Coverage':<10}")
        print("-"*80)

        for model_name, results in self.results.items():
            metrics = results['ranking_metrics']['k_10']
            print(f"{model_name:<20} "
                  f"{metrics['hit_rate']:<10.3f} "
                  f"{metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} "
                  f"{metrics['ndcg']:<10.3f} "
                  f"{metrics['coverage']:<10.3f}")

        # 2. Training Cost Comparison
        print("\n2. TRAINING COST")
        print("-"*80)
        print(f"{'Model':<20} {'Time (s)':<12} {'CPU %':<10} {'Memory (MB)':<12} {'Peak (MB)':<12}")
        print("-"*80)

        for model_name, results in self.results.items():
            cost = results['training_cost']
            print(f"{model_name:<20} "
                  f"{cost['training_time_seconds']:<12.2f} "
                  f"{cost['cpu_usage_percent']:<10.1f} "
                  f"{cost['memory_used_mb']:<12.1f} "
                  f"{cost['peak_memory_mb']:<12.1f}")

        # 3. Model Size Comparison
        print("\n3. MODEL SIZE")
        print("-"*80)
        print(f"{'Model':<20} {'Disk (MB)':<12} {'Memory (MB)':<12}")
        print("-"*80)

        for model_name, results in self.results.items():
            size = results['model_size']
            print(f"{model_name:<20} "
                  f"{size['disk_size_mb']:<12.2f} "
                  f"{size['memory_size_mb']:<12.2f}")

        # 4. Inference Performance Comparison
        print("\n4. INFERENCE PERFORMANCE")
        print("-"*80)
        print(f"{'Model':<20} {'Avg (ms)':<10} {'P50 (ms)':<10} {'P95 (ms)':<10} {'P99 (ms)':<10}")
        print("-"*80)

        for model_name, results in self.results.items():
            perf = results['inference_performance']
            print(f"{model_name:<20} "
                  f"{perf['avg_latency_ms']:<10.2f} "
                  f"{perf['p50_latency_ms']:<10.2f} "
                  f"{perf['p95_latency_ms']:<10.2f} "
                  f"{perf['p99_latency_ms']:<10.2f}")

        # 5. Throughput Comparison
        print("\n5. THROUGHPUT (users/sec)")
        print("-"*80)
        print(f"{'Model':<20} {'Batch 1':<12} {'Batch 10':<12} {'Batch 50':<12} {'Batch 100':<12}")
        print("-"*80)

        for model_name, results in self.results.items():
            throughput = results['inference_performance']['throughput_users_per_sec']
            print(f"{model_name:<20} ", end="")
            for batch_size in [1, 10, 50, 100]:
                key = f'batch_{batch_size}'
                if key in throughput:
                    print(f"{throughput[key]:<12.1f} ", end="")
                else:
                    print(f"{'N/A':<12} ", end="")
            print()

        # 6. Summary and Recommendations
        print("\n" + "="*80)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*80)

        # Find best model for each metric
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['ranking_metrics']['k_10']['ndcg'])[0]
        best_speed = min(self.results.items(), key=lambda x: x[1]['inference_performance']['avg_latency_ms'])[0]
        best_size = min(self.results.items(), key=lambda x: x[1]['model_size']['memory_size_mb'])[0]

        print(f"\n🏆 Best Accuracy (NDCG@10): {best_accuracy}")
        print(f"⚡ Best Speed (Latency): {best_speed}")
        print(f"💾 Best Size (Memory): {best_size}")

        # Cost per recommendation (simplified)
        print("\n💰 Estimated Cost per 1M Recommendations:")
        for model_name, results in self.results.items():
            # Assume $0.10 per CPU-hour
            cpu_time_per_rec = results['inference_performance']['avg_latency_ms'] / 1000
            cost_per_million = cpu_time_per_rec * 1_000_000 / 3600 * 0.10
            print(f"   {model_name}: ${cost_per_million:.2f}")

        # Save full report
        report_path = Path("model_comparison_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Full report saved to: {report_path}")

async def main():
    """Run comprehensive model comparison"""
    benchmark = ModelBenchmark()

    # Load data
    ratings_df, movies_df = await benchmark.load_data()

    # Split data by timestamp
    ratings_df = ratings_df.sort_values('timestamp')
    n = len(ratings_df)
    train_df = ratings_df[:int(0.7 * n)]
    val_df = ratings_df[int(0.7 * n):int(0.85 * n)]
    test_df = ratings_df[int(0.85 * n):]

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Benchmark models
    models_to_test = [
        (PopularityModel, "Popularity", {}),
        (CollaborativeFilteringModel, "Collaborative", {'neighborhood_size': 50}),
        (ALSModel, "ALS", {'factors': 100, 'regularization': 0.01})
    ]

    for model_class, model_name, kwargs in models_to_test:
        await benchmark.benchmark_model(
            model_class, model_name,
            train_df, val_df, test_df,
            **kwargs
        )

    # Generate comparison report
    benchmark.generate_comparison_report()

if __name__ == "__main__":
    asyncio.run(main())
