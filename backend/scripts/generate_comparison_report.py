#!/usr/bin/env python3
"""
Generate comprehensive comparison report combining accuracy and performance metrics
"""
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

settings = get_settings()

def load_model_metadata(model_name, version="0.3.0"):
    """Load model metadata from registry"""
    metadata_path = settings.model_registry_path / model_name / version / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def load_benchmark_results():
    """Load benchmark results"""
    if Path('benchmark_results_existing.json').exists():
        with open('benchmark_results_existing.json', 'r') as f:
            return json.load(f)
    return {}

def generate_report():
    """Generate comprehensive comparison report"""
    models = ['popularity', 'collaborative', 'als']

    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*80)

    # Load data
    benchmark = load_benchmark_results()

    # Collect all metrics
    all_metrics = {}

    for model in models:
        metadata = load_model_metadata(model)
        bench = benchmark.get(model, {})

        if metadata:
            all_metrics[model] = {
                'accuracy': metadata.get('test_metrics', {}),
                'training': {
                    'time_seconds': metadata.get('training_time_seconds', 0),
                    'model_params': metadata.get('model_params', {})
                },
                'inference': bench.get('inference', {}),
                'size_mb': bench.get('size_mb', 0)
            }

    # 1. Accuracy Comparison
    print("\n1. ACCURACY METRICS (Test Set)")
    print("-"*80)
    print(f"{'Model':<15} {'Precision@10':<12} {'Recall@10':<12} {'NDCG@10':<12} {'RMSE':<12}")
    print("-"*80)

    for model in models:
        if model in all_metrics:
            acc = all_metrics[model]['accuracy']
            print(f"{model:<15} "
                  f"{acc.get('precision@10', 0):<12.4f} "
                  f"{acc.get('recall@10', 0):<12.4f} "
                  f"{acc.get('ndcg@10', 0):<12.4f} "
                  f"{acc.get('rmse', 0):<12.4f}")

    # 2. Training Cost
    print("\n2. TRAINING COST")
    print("-"*80)
    print(f"{'Model':<15} {'Time (s)':<12} {'Parameters':<40}")
    print("-"*80)

    for model in models:
        if model in all_metrics:
            train = all_metrics[model]['training']
            params = train.get('model_params', {})
            param_str = str(params) if params else "N/A"
            if len(param_str) > 40:
                param_str = param_str[:37] + "..."

            print(f"{model:<15} "
                  f"{train.get('time_seconds', 0):<12.2f} "
                  f"{param_str:<40}")

    # 3. Inference Performance
    print("\n3. INFERENCE PERFORMANCE")
    print("-"*80)
    print(f"{'Model':<15} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Throughput':<15} {'Size (MB)':<10}")
    print("-"*80)

    for model in models:
        if model in all_metrics:
            inf = all_metrics[model]['inference']
            size = all_metrics[model]['size_mb']
            throughput = inf.get('throughput_users_per_sec', 0)

            print(f"{model:<15} "
                  f"{inf.get('avg_latency_ms', 0):<12.1f} "
                  f"{inf.get('p95_latency_ms', 0):<12.1f} "
                  f"{throughput:<15.1f} "
                  f"{size:<10.2f}")

    # 4. Cost Analysis
    print("\n4. COST ANALYSIS (per 1M recommendations)")
    print("-"*80)

    for model in models:
        if model in all_metrics:
            # Compute cost based on latency
            avg_latency_s = all_metrics[model]['inference'].get('avg_latency_ms', 0) / 1000
            cpu_hours = (avg_latency_s * 1_000_000) / 3600
            compute_cost = cpu_hours * 0.10  # $0.10 per CPU hour

            # Storage cost (monthly)
            size_gb = all_metrics[model]['size_mb'] / 1024
            storage_cost = size_gb * 0.023  # $0.023 per GB/month

            print(f"{model:<15} Compute: ${compute_cost:>6.2f}  Storage: ${storage_cost:>6.4f}/month")

    # 5. Summary Recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    # Find winners
    if all_metrics:
        # Best accuracy (NDCG)
        best_accuracy = max(all_metrics.items(),
                          key=lambda x: x[1]['accuracy'].get('ndcg@10', 0))[0]

        # Best speed
        best_speed = min(all_metrics.items(),
                        key=lambda x: x[1]['inference'].get('avg_latency_ms', float('inf')))[0]

        # Best efficiency (accuracy/latency ratio)
        efficiency_scores = {}
        for model, metrics in all_metrics.items():
            ndcg = metrics['accuracy'].get('ndcg@10', 0)
            latency = metrics['inference'].get('avg_latency_ms', 1)
            efficiency_scores[model] = (ndcg * 1000) / latency if latency > 0 else 0

        best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])[0]

        print(f"\n🏆 Best Accuracy (NDCG@10): {best_accuracy}")
        print(f"⚡ Best Speed: {best_speed}")
        print(f"💰 Best Efficiency (accuracy/latency): {best_efficiency}")

        print("\n📊 Model Selection Guide:")
        print("- For real-time serving with best accuracy: ALS")
        print("- For extremely high throughput: Popularity")
        print("- For explainable recommendations: Collaborative (but needs optimization)")
        print("- For production: Consider ALS with caching for popular users")

    # Save full report
    with open('full_comparison_report.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n📄 Full report saved to full_comparison_report.json")

if __name__ == "__main__":
    generate_report()
