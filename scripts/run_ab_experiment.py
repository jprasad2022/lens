#!/usr/bin/env python3
"""
Run A/B experiment and show statistical results
"""
import requests
import json
import time
import random
from datetime import datetime

BASE_URL = "http://localhost:8000"

def create_experiment():
    """Create A/B experiment"""
    experiment_data = {
        "name": "model_performance_test",
        "models": {
            "popularity": 0.5,
            "collaborative": 0.5
        },
        "duration_days": 7
    }
    
    response = requests.post(f"{BASE_URL}/ab/experiments", json=experiment_data)
    if response.status_code == 200:
        exp_id = response.json()["experiment_id"]
        print(f"Created experiment: {exp_id}")
        return exp_id
    else:
        print(f"Failed to create experiment: {response.status_code}")
        return None

def simulate_user_traffic(experiment_id, num_requests=100):
    """Simulate user traffic"""
    print(f"\nSimulating {num_requests} user requests...")
    
    for i in range(num_requests):
        user_id = random.randint(1, 6040)
        
        # Get recommendation
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/recommend/{user_id}?k=10")
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            model_used = data["model_info"]["name"]
            
            # Record metric
            metric_data = {
                "experiment_id": experiment_id,
                "model": model_used,
                "metric_name": "latency_ms",
                "value": latency_ms
            }
            requests.post(f"{BASE_URL}/ab/experiments/{experiment_id}/record", json=metric_data)
            
            # Record success rate
            metric_data["metric_name"] = "success_rate"
            metric_data["value"] = 1.0
            requests.post(f"{BASE_URL}/ab/experiments/{experiment_id}/record", json=metric_data)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1} requests...")
    
    print("Traffic simulation complete!")

def get_experiment_results(experiment_id):
    """Get and display experiment results"""
    response = requests.get(f"{BASE_URL}/ab/experiments/{experiment_id}")
    if response.status_code == 200:
        results = response.json()
        
        print(f"\n{'='*60}")
        print(f"A/B EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Experiment: {results['name']}")
        print(f"Status: {results['status']}")
        print(f"Start: {results['start_date']}")
        print(f"End: {results['end_date']}")
        
        # Display statistics
        if "statistics" in results and results["statistics"]:
            print(f"\nStatistical Analysis:")
            print(f"{'-'*40}")
            
            for metric, stats in results["statistics"].items():
                print(f"\nMetric: {metric}")
                
                # Model performance
                for model, perf in stats.items():
                    if model != "statistical_test" and isinstance(perf, dict):
                        mean = perf.get("mean", 0)
                        std = perf.get("std", 0)
                        count = perf.get("count", 0)
                        ci = perf.get("confidence_interval", (0, 0))
                        
                        print(f"  {model}:")
                        print(f"    Mean: {mean:.2f}")
                        print(f"    Std Dev: {std:.2f}")
                        print(f"    Sample Size: {count}")
                        print(f"    95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
                
                # Statistical test
                if "statistical_test" in stats:
                    test = stats["statistical_test"]
                    print(f"\n  Statistical Test ({test['type']}):")
                    print(f"    t-statistic: {test['t_statistic']:.4f}")
                    print(f"    p-value: {test['p_value']:.4f}")
                    print(f"    Significant: {'YES' if test['significant'] else 'NO'}")
        
        # Get winner recommendation
        winner_response = requests.get(f"{BASE_URL}/ab/experiments/{experiment_id}/winner?metric=latency_ms")
        if winner_response.status_code == 200:
            winner_data = winner_response.json()
            print(f"\n{'='*40}")
            print(f"RECOMMENDATION: {winner_data.get('recommendation', 'No clear winner')}")
            if winner_data.get('winner'):
                print(f"Winner: {winner_data['winner']}")
        
        return results

def main():
    print("Starting A/B Testing Demo...")
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("ERROR: Backend not responding")
            return
    except:
        print("ERROR: Cannot connect to backend at", BASE_URL)
        return
    
    # Create experiment
    exp_id = create_experiment()
    if not exp_id:
        return
    
    # Simulate traffic
    simulate_user_traffic(exp_id, num_requests=200)
    
    # Wait a bit for metrics to process
    print("\nWaiting for metrics processing...")
    time.sleep(2)
    
    # Get results
    results = get_experiment_results(exp_id)
    
    # Save results for documentation
    with open("ab_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ab_experiment_results.json")

if __name__ == "__main__":
    main()