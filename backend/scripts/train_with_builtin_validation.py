"""
Train models with built-in validation and hyperparameter tuning
"""
import sys
import os
import asyncio
import pickle
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from recommender.models_impl_validated import (
    PopularityModel,
    CollaborativeFilteringModel,
    ALSModel
)
from config.settings import get_settings

settings = get_settings()

async def load_data():
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

    return ratings_df, movies_df

async def save_model(model_name: str, model: any, version: str = "0.2.0"):
    """Save model to model registry"""
    model_dir = settings.model_registry_path / model_name / version
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(model.metadata, f, indent=2)

    # Update latest version
    latest_path = settings.model_registry_path / model_name / "latest.txt"
    with open(latest_path, 'w') as f:
        f.write(version)

    print(f"✓ Saved {model_name} model to {model_dir}")

async def main():
    print("=" * 60)
    print("Training Models with Built-in Validation")
    print("=" * 60)

    # Load data
    ratings_df, movies_df = await load_data()

    # Train Popularity Model
    print("\n" + "="*40)
    print("Training Popularity Model with Validation")
    print("="*40)
    popularity_model = PopularityModel()
    await popularity_model.train_with_validation(ratings_df, val_size=0.1)
    await save_model("popularity", popularity_model)

    # Print validation metrics
    val_metrics = popularity_model.metadata.get("validation_metrics", {})
    print(f"\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Train Collaborative Filtering Model
    print("\n" + "="*40)
    print("Training Collaborative Filtering Model")
    print("="*40)

    # Try different hyperparameters
    print("\nTesting hyperparameters...")
    best_collab = None
    best_score = 0

    for min_sim in [0.0, 0.1, 0.2]:
        for n_size in [20, 50, 100]:
            print(f"\nTrying min_similarity={min_sim}, neighborhood_size={n_size}")
            collab_model = CollaborativeFilteringModel(
                min_similarity=min_sim,
                neighborhood_size=n_size
            )
            await collab_model.train_with_validation(ratings_df, val_size=0.1)

            val_ndcg = collab_model.metadata["validation_metrics"]["ndcg@10"]
            if val_ndcg > best_score:
                best_score = val_ndcg
                best_collab = collab_model

    print(f"\nBest Collaborative Model: NDCG@10={best_score:.4f}")
    await save_model("collaborative", best_collab)

    # Train ALS Model with hyperparameter search
    print("\n" + "="*40)
    print("Training ALS Model with Hyperparameter Search")
    print("="*40)
    als_model = ALSModel()
    await als_model.train_with_hyperparameter_search(ratings_df)
    await save_model("als", als_model)

    # Print best parameters
    print(f"\nBest ALS Parameters:")
    for param, value in als_model.metadata.get("best_params", {}).items():
        print(f"  {param}: {value}")

    # Summary
    print("\n" + "="*60)
    print("Training Complete - Model Summary")
    print("="*60)

    models = [
        ("Popularity", popularity_model),
        ("Collaborative", best_collab),
        ("ALS", als_model)
    ]

    print(f"\n{'Model':<15} {'Precision@10':<15} {'Recall@10':<15} {'NDCG@10':<15}")
    print("-" * 60)

    for name, model in models:
        metrics = model.metadata.get("validation_metrics", {})
        print(f"{name:<15} "
              f"{metrics.get('precision@10', 0):<15.4f} "
              f"{metrics.get('recall@10', 0):<15.4f} "
              f"{metrics.get('ndcg@10', 0):<15.4f}")

    print("\n✅ All models trained with validation successfully!")
    print("📊 Models saved with validation metrics in metadata")
    print("🎯 Use version 0.2.0 for these validated models")

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
