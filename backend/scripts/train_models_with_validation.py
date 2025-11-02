"""
Train recommendation models with proper validation and testing
"""
import sys
import os
import asyncio
import pickle
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from recommender.models_impl import PopularityModel, CollaborativeFilteringModel, ALSModel
from config.settings import get_settings

settings = get_settings()

class ModelEvaluator:
    """Evaluate recommendation models"""

    @staticmethod
    def precision_at_k(relevant_items, recommended_items, k):
        """Calculate precision@k"""
        recommended_k = recommended_items[:k]
        relevant_in_k = len(set(recommended_k) & set(relevant_items))
        return relevant_in_k / k if k > 0 else 0

    @staticmethod
    def recall_at_k(relevant_items, recommended_items, k):
        """Calculate recall@k"""
        recommended_k = recommended_items[:k]
        relevant_in_k = len(set(recommended_k) & set(relevant_items))
        return relevant_in_k / len(relevant_items) if len(relevant_items) > 0 else 0

    @staticmethod
    def ndcg_at_k(relevant_items, recommended_items, k):
        """Calculate NDCG@k"""
        def dcg_at_k(scores, k):
            scores = np.array(scores)[:k]
            if len(scores) == 0:
                return 0.0
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

        recommended_k = recommended_items[:k]
        relevance = [1 if item in relevant_items else 0 for item in recommended_k]

        if sum(relevance) == 0:
            return 0.0

        dcg = dcg_at_k(relevance, k)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)

        return dcg / idcg if idcg > 0 else 0.0

async def load_and_split_data(test_size=0.2, random_state=42):
    """Load MovieLens data and split into train/test sets"""
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

    # Load movies
    movies_path = data_path / "movies.dat"
    movies_df = pd.read_csv(
        movies_path,
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    print(f"Loaded {len(movies_df)} movies")

    # Split by users to avoid data leakage
    users = ratings_df['user_id'].unique()
    train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)

    train_df = ratings_df[ratings_df['user_id'].isin(train_users)]
    test_df = ratings_df[ratings_df['user_id'].isin(test_users)]

    print(f"Train set: {len(train_df)} ratings from {len(train_users)} users")
    print(f"Test set: {len(test_df)} ratings from {len(test_users)} users")

    return train_df, test_df, movies_df

async def evaluate_model(model, test_df, k_values=[5, 10, 20]):
    """Evaluate a model on test set"""
    # Group test data by user
    user_items = defaultdict(list)
    for _, row in test_df.iterrows():
        if row['rating'] >= 4:  # Consider 4+ ratings as relevant
            user_items[row['user_id']].append(row['movie_id'])

    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'recall@{k}': [] for k in k_values})
    metrics.update({f'ndcg@{k}': [] for k in k_values})

    evaluator = ModelEvaluator()
    sample_users = np.random.choice(list(user_items.keys()), min(100, len(user_items)), replace=False)

    for user_id in sample_users:
        relevant_items = user_items[user_id]
        if len(relevant_items) == 0:
            continue

        try:
            recommendations = await model.predict(user_id, k=max(k_values))

            for k in k_values:
                metrics[f'precision@{k}'].append(
                    evaluator.precision_at_k(relevant_items, recommendations, k)
                )
                metrics[f'recall@{k}'].append(
                    evaluator.recall_at_k(relevant_items, recommendations, k)
                )
                metrics[f'ndcg@{k}'].append(
                    evaluator.ndcg_at_k(relevant_items, recommendations, k)
                )
        except:
            # Skip users that cause errors (e.g., cold start)
            continue

    # Calculate average metrics
    avg_metrics = {}
    for metric_name, values in metrics.items():
        if values:
            avg_metrics[metric_name] = np.mean(values)
        else:
            avg_metrics[metric_name] = 0.0

    return avg_metrics

async def save_model_with_metrics(model_name: str, model: any, metrics: dict, version: str = "0.1.0"):
    """Save model with evaluation metrics"""
    model_dir = settings.model_registry_path / model_name / version
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata with metrics
    metadata = {
        "name": model_name,
        "version": version,
        "type": model_name,
        "trained_at": model.metadata.get("trained_at", ""),
        "metrics": {
            **model.metadata.get("metrics", {}),
            **metrics
        },
        "parameters": {}
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update latest version
    latest_path = settings.model_registry_path / model_name / "latest.txt"
    with open(latest_path, 'w') as f:
        f.write(version)

    print(f"✓ Saved {model_name} model to {model_dir}")

async def main():
    print("=" * 50)
    print("Training Models with Validation")
    print("=" * 50)

    # Load and split data
    train_df, test_df, movies_df = await load_and_split_data(test_size=0.2)

    # Train and evaluate each model
    models_to_train = [
        ("popularity", PopularityModel),
        ("collaborative", CollaborativeFilteringModel),
        ("als", ALSModel)
    ]

    results = {}

    for model_name, ModelClass in models_to_train:
        print(f"\n{'='*30}")
        print(f"Training {model_name} model...")
        print(f"{'='*30}")

        # Train model
        start_time = time.time()
        model = ModelClass()
        await model.train(train_df)
        train_time = time.time() - start_time

        print(f"✓ Training completed in {train_time:.2f}s")

        # Evaluate model
        print(f"Evaluating on test set...")
        start_time = time.time()
        metrics = await evaluate_model(model, test_df)
        eval_time = time.time() - start_time

        # Add timing info
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time

        # Print results
        print(f"\nResults for {model_name}:")
        for metric, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")

        results[model_name] = metrics

        # Save model with metrics
        await save_model_with_metrics(model_name, model, metrics)

    # Compare models
    print(f"\n{'='*50}")
    print("Model Comparison")
    print(f"{'='*50}")

    # Create comparison table
    metric_names = ['precision@10', 'recall@10', 'ndcg@10']
    print(f"\n{'Model':<20} " + " ".join(f"{m:<15}" for m in metric_names))
    print("-" * 70)

    for model_name, metrics in results.items():
        values = [f"{metrics.get(m, 0):.4f}" for m in metric_names]
        print(f"{model_name:<20} " + " ".join(f"{v:<15}" for v in values))

    # Find best model for each metric
    print("\nBest models by metric:")
    for metric in metric_names:
        best_model = max(results.items(), key=lambda x: x[1].get(metric, 0))
        print(f"  {metric}: {best_model[0]} ({best_model[1].get(metric, 0):.4f})")

    print("\n✅ All models trained and evaluated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
