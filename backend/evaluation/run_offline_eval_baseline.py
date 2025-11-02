#!/usr/bin/env python3
"""
Run offline evaluation with baseline models (no ML model loading required)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment to disable Kafka
os.environ["KAFKA_ENABLED"] = "false"

import json
import random
from evaluation.offline_evaluator import OfflineEvaluator


def run_baseline_evaluation():
    """Run offline evaluation with baseline models"""
    
    # Initialize evaluator
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(backend_dir, "data")
    
    evaluator = OfflineEvaluator(data_path=data_path)
    evaluator.load_data()
    
    # Perform chronological split
    print("\nPerforming chronological split...")
    train, test = evaluator.chronological_split()
    
    # Calculate popularity baseline from training data
    print("\nCalculating popularity baseline...")
    movie_popularity = train.groupby('movie_id').size().sort_values(ascending=False)
    top_100_movies = movie_popularity.head(100).index.tolist()
    
    # Create baseline models
    class PopularityModel:
        """Recommends most popular movies from training set"""
        def recommend(self, user_id, k=10):
            # Return top k popular movies
            return [{'movie_id': movie_id} for movie_id in top_100_movies[:k]]
    
    class RandomModel:
        """Recommends random movies"""
        def recommend(self, user_id, k=10):
            all_movies = list(evaluator.movies['movie_id'].unique())
            selected = random.sample(all_movies, min(k, len(all_movies)))
            return [{'movie_id': movie_id} for movie_id in selected]
    
    # Evaluate models
    print("\nEvaluating Popularity baseline...")
    pop_model = PopularityModel()
    pop_results = evaluator.evaluate_model(pop_model)
    
    print("\nEvaluating Random baseline...")
    random.seed(42)  # For reproducibility
    rand_model = RandomModel()
    rand_results = evaluator.evaluate_model(rand_model)
    
    # Subpopulation analysis
    print("\nPerforming subpopulation analysis...")
    subpop_results = evaluator.subpopulation_analysis()
    
    # Generate reports
    print("\n" + "="*50)
    print("POPULARITY BASELINE RESULTS")
    print("="*50)
    pop_report = evaluator.generate_report(pop_results, subpop_results)
    print(pop_report)
    
    print("\n" + "="*50)
    print("RANDOM BASELINE RESULTS")
    print("="*50)
    rand_report = evaluator.generate_report(rand_results, subpop_results)
    print(rand_report)
    
    # Save all results
    with open('offline_evaluation_baseline_results.json', 'w') as f:
        json.dump({
            'popularity_baseline': {
                'metrics': pop_results,
                'description': 'Recommends most popular movies from training set'
            },
            'random_baseline': {
                'metrics': rand_results,
                'description': 'Recommends random movies'
            },
            'subpopulations': subpop_results,
            'split_date': '2000-11-01',
            'train_size': len(train),
            'test_size': len(test)
        }, f, indent=2)


def main():
    """Run offline evaluation with baselines"""
    print("=== Offline Evaluation - Baseline Models ===")
    print()
    
    try:
        run_baseline_evaluation()
        print("\n✓ Baseline evaluation completed")
        print("Check offline_evaluation_baseline_results.json for metrics")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()