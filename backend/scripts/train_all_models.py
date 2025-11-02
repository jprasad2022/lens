"""
Train all recommendation models properly
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
from recommender.models_impl import PopularityModel, CollaborativeFilteringModel, ALSModel
from config.settings import get_settings

settings = get_settings()

async def load_data():
    """Load MovieLens data"""
    data_path = settings.data_path  # Use data_path instead of movielens_path

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

async def save_model(model_name: str, model: any, version: str = "0.1.0"):
    """Save model to model registry"""
    model_dir = settings.model_registry_path / model_name / version
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        "name": model_name,
        "version": version,
        "type": model_name,
        "trained_at": model.metadata.get("trained_at", ""),
        "metrics": {
            "num_items": model.metadata.get("n_movies", 0),
            "num_users": model.metadata.get("n_users", 0),
            "training_time": model.metadata.get("training_time_seconds", 0)
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
    print("Training All Recommendation Models")
    print("=" * 50)

    # Load data
    ratings_df, movies_df = await load_data()

    # Train Popularity Model
    print("\n🏋️ Training Popularity Model...")
    popularity_model = PopularityModel()
    await popularity_model.train(ratings_df)
    await save_model("popularity", popularity_model)

    # Train Collaborative Filtering Model
    print("\n🏋️ Training Collaborative Filtering Model...")
    collab_model = CollaborativeFilteringModel()
    await collab_model.train(ratings_df)
    await save_model("collaborative", collab_model)

    # Train ALS Model
    print("\n🏋️ Training ALS Model...")
    als_model = ALSModel()
    await als_model.train(ratings_df)
    await save_model("als", als_model)

    print("\n✅ All models trained successfully!")

    # Test predictions
    print("\n🧪 Testing predictions for user 100...")
    test_user = 100

    pop_recs = await popularity_model.predict(test_user, k=5)
    print(f"Popularity recommendations: {pop_recs}")

    collab_recs = await collab_model.predict(test_user, k=5)
    print(f"Collaborative recommendations: {collab_recs}")

    als_recs = await als_model.predict(test_user, k=5)
    print(f"ALS recommendations: {als_recs}")

if __name__ == "__main__":
    asyncio.run(main())
