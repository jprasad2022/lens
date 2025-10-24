#!/usr/bin/env python3
"""
Compare recommendation models on various metrics
"""

import os
import sys
import time
import json
import pickle
import asyncio
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tabulate import tabulate

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from recommender.models_impl import PopularityModel, CollaborativeFilteringModel, ALSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate and compare recommendation models."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ratings_df = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self):
        """Load MovieLens data."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load ratings
        ratings_path = os.path.join(self.data_path, "ratings.dat")
        self.ratings_df = pd.read_csv(
            ratings_path, 
            sep='::', 
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        logger.info(f"Loaded {len(self.ratings_df)} ratings")
        
        # Split data (80/20)
        self.ratings_df = self.ratings_df.sort_values('timestamp')
        split_idx = int(0.8 * len(self.ratings_df))
        self.train_df = self.ratings_df[:split_idx]
        self.test_df = self.ratings_df[split_idx:]
        
        logger.info(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")
    
    def compute_hit_rate_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """Compute Hit Rate @ K."""
        rec_set = set(recommendations[:k])
        actual_set = set(actual)
        hits = len(rec_set & actual_set)
        return 1.0 if hits > 0 else 0.0
    
    def compute_ndcg_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain @ K."""
        rec_list = recommendations[:k]
        
        # Build relevance scores
        relevance = [1.0 if item in actual else 0.0 for item in rec_list]
        
        # DCG
        dcg = relevance[0] if relevance else 0.0
        for i in range(1, len(relevance)):
            dcg += relevance[i] / np.log2(i + 1)
        
        # Ideal DCG
        ideal_relevance = [1.0] * min(len(actual), k)
        ideal_relevance.extend([0.0] * (k - len(ideal_relevance)))
        
        idcg = ideal_relevance[0] if ideal_relevance else 0.0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        # NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    async def evaluate_model(self, model, model_name: str) -> Dict:
        """Evaluate a single model."""
        logger.info(f"\nEvaluating {model_name}...")
        
        results = {
            "model": model_name,
            "training_time": 0,
            "inference_time_avg": 0,
            "model_size_mb": 0,
            "hr@10": 0,
            "hr@20": 0,
            "ndcg@10": 0,
            "ndcg@20": 0,
        }
        
        # Training phase
        start_time = time.time()
        await model.train(self.train_df)
        results["training_time"] = time.time() - start_time
        
        # Get model size
        import tempfile
        model_path = os.path.join(tempfile.gettempdir(), f"{model_name}_model.pkl")
        model.save(model_path)
        results["model_size_mb"] = os.path.getsize(model_path) / (1024 * 1024)
        
        # Evaluation phase
        test_users = self.test_df['user_id'].unique()[:100]  # Sample 100 users for evaluation
        
        hr_10_scores = []
        hr_20_scores = []
        ndcg_10_scores = []
        ndcg_20_scores = []
        inference_times = []
        
        for user_id in test_users:
            # Get actual movies from test set
            actual_movies = self.test_df[self.test_df['user_id'] == user_id]['movie_id'].tolist()
            
            if not actual_movies:
                continue
            
            # Get recommendations
            start_time = time.time()
            try:
                recommendations = await model.predict(user_id, k=20)
                inference_times.append(time.time() - start_time)
                
                # Compute metrics
                hr_10_scores.append(self.compute_hit_rate_at_k(recommendations, actual_movies, 10))
                hr_20_scores.append(self.compute_hit_rate_at_k(recommendations, actual_movies, 20))
                ndcg_10_scores.append(self.compute_ndcg_at_k(recommendations, actual_movies, 10))
                ndcg_20_scores.append(self.compute_ndcg_at_k(recommendations, actual_movies, 20))
                
            except Exception as e:
                logger.warning(f"Error predicting for user {user_id}: {e}")
                continue
        
        # Aggregate results
        results["hr@10"] = np.mean(hr_10_scores) if hr_10_scores else 0
        results["hr@20"] = np.mean(hr_20_scores) if hr_20_scores else 0
        results["ndcg@10"] = np.mean(ndcg_10_scores) if ndcg_10_scores else 0
        results["ndcg@20"] = np.mean(ndcg_20_scores) if ndcg_20_scores else 0
        results["inference_time_avg"] = np.mean(inference_times) * 1000 if inference_times else 0  # Convert to ms
        
        logger.info(f"Completed evaluation of {model_name}")
        
        return results
    
    async def compare_models(self) -> List[Dict]:
        """Compare all models."""
        models = {
            "Popularity": PopularityModel(),
            "Collaborative": CollaborativeFilteringModel(),
            "ALS": ALSModel(factors=50, iterations=10),
        }
        
        results = []
        for model_name, model in models.items():
            result = await self.evaluate_model(model, model_name)
            results.append(result)
        
        return results
    
    def print_comparison_table(self, results: List[Dict]):
        """Print comparison table."""
        # Prepare data for table
        headers = [
            "Model", 
            "Training Time (s)", 
            "Avg Inference (ms)", 
            "Model Size (MB)",
            "HR@10", 
            "HR@20", 
            "NDCG@10", 
            "NDCG@20"
        ]
        
        rows = []
        for r in results:
            row = [
                r["model"],
                f"{r['training_time']:.2f}",
                f"{r['inference_time_avg']:.2f}",
                f"{r['model_size_mb']:.2f}",
                f"{r['hr@10']:.4f}",
                f"{r['hr@20']:.4f}",
                f"{r['ndcg@10']:.4f}",
                f"{r['ndcg@20']:.4f}",
            ]
            rows.append(row)
        
        print("\n" + "="*100)
        print("MODEL COMPARISON RESULTS")
        print("="*100)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Metric definitions
        print("\nMetric Definitions:")
        print("- HR@K (Hit Rate @ K): Fraction of users for whom at least one relevant item appears in top-K recommendations")
        print("- NDCG@K (Normalized Discounted Cumulative Gain @ K): Measures ranking quality, considering position of relevant items")
        print("- Training Time: Time to train model on 80% of data")
        print("- Avg Inference: Average time to generate recommendations for one user")
        print("- Model Size: Serialized model size on disk")
        
        # Save results to JSON
        output_path = "model_comparison_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "data_stats": {
                    "total_ratings": len(self.ratings_df),
                    "train_ratings": len(self.train_df),
                    "test_ratings": len(self.test_df),
                }
            }, f, indent=2)
        
        print(f"\nResults saved to {output_path}")

async def main():
    parser = argparse.ArgumentParser(description="Compare recommendation models")
    parser.add_argument("--data-path", default="data", help="Path to MovieLens data")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.data_path)
    
    # Load data
    evaluator.load_data()
    
    # Compare models
    results = await evaluator.compare_models()
    
    # Print results
    evaluator.print_comparison_table(results)

if __name__ == "__main__":
    asyncio.run(main())