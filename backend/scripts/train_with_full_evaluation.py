"""
Complete training pipeline with Train/Validation/Test splits
"""
import sys
import os
import asyncio
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.models_impl_full_validation import (
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

async def save_model(model_name: str, model: any, version: str = "0.3.0"):
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

    # Save model card with results
    model_card = f"""# {model_name.capitalize()} Model v{version}

## Model Description
Model type: {model_name}
Trained with proper train/validation/test methodology

## Training Methodology
- **Train Set**: 70% of data (earliest by timestamp)
- **Validation Set**: 15% of data (middle)
- **Test Set**: 15% of data (latest)
- Split by timestamp to simulate real-world conditions

## Performance Metrics

### Validation Set
- Precision@10: {model.metadata['validation_metrics']['precision@10']:.4f}
- Recall@10: {model.metadata['validation_metrics']['recall@10']:.4f}
- NDCG@10: {model.metadata['validation_metrics']['ndcg@10']:.4f}
- RMSE: {model.metadata['validation_metrics']['rmse']:.4f}

### Test Set (Final)
- Precision@10: {model.metadata['test_metrics']['precision@10']:.4f}
- Recall@10: {model.metadata['test_metrics']['recall@10']:.4f}
- NDCG@10: {model.metadata['test_metrics']['ndcg@10']:.4f}
- RMSE: {model.metadata['test_metrics']['rmse']:.4f}

## Model Parameters
{json.dumps(model.metadata.get('model_params', {}), indent=2)}

## Training Details
- Training Time: {model.metadata['training_time_seconds']:.2f} seconds
- Trained At: {model.metadata['trained_at']}
"""

    model_card_path = model_dir / "model_card.md"
    with open(model_card_path, 'w') as f:
        f.write(model_card)

    # Update latest version
    latest_path = settings.model_registry_path / model_name / "latest.txt"
    with open(latest_path, 'w') as f:
        f.write(version)

    logger.info(f"✓ Saved {model_name} model to {model_dir}")

async def main():
    print("=" * 70)
    print("Training Models with Train/Validation/Test Methodology")
    print("=" * 70)

    # Load data
    ratings_df, movies_df = await load_data()

    # Results storage
    all_results = {}

    # Train Popularity Model
    print("\n" + "="*50)
    print("1. POPULARITY MODEL")
    print("="*50)

    popularity_model = PopularityModel()
    results = await popularity_model.train_validate_test(ratings_df, split_method="time")
    all_results['popularity'] = results
    await save_model("popularity", popularity_model)

    # Train Collaborative Filtering Model
    print("\n" + "="*50)
    print("2. COLLABORATIVE FILTERING MODEL")
    print("="*50)

    # Quick hyperparameter search on validation set
    print("\nSearching for best hyperparameters...")
    best_cf = None
    best_cf_score = 0

    for min_sim in [0.0, 0.1]:
        for n_size in [30, 50]:
            cf_model = CollaborativeFilteringModel(
                min_similarity=min_sim,
                neighborhood_size=n_size,
                min_neighbors=3
            )
            results = await cf_model.train_validate_test(ratings_df)

            val_score = results['val_metrics'].ndcg
            if val_score > best_cf_score:
                best_cf_score = val_score
                best_cf = cf_model
                all_results['collaborative'] = results

    await save_model("collaborative", best_cf)

    # Train ALS Model with Cross-Validation
    print("\n" + "="*50)
    print("3. ALS MODEL WITH CROSS-VALIDATION")
    print("="*50)

    als_model = ALSModel()

    # Cross-validation for hyperparameter tuning
    print("\nPerforming 3-fold cross-validation...")
    param_grid = {
        'factors': [50, 100],
        'regularization': [0.01, 0.1],
        'alpha': [20, 40]
    }

    best_params = await als_model.cross_validate(ratings_df, param_grid)
    print(f"\nBest CV parameters: {best_params}")

    # Train final model with best parameters
    als_model.factors = best_params['factors']
    als_model.regularization = best_params['regularization']
    als_model.alpha = best_params['alpha']
    als_model.metadata['best_params'] = best_params

    results = await als_model.train_validate_test(ratings_df)
    all_results['als'] = results
    await save_model("als", als_model)

    # Final Comparison
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)

    # Comparison table
    print(f"\n{'Model':<20} {'Val P@10':<10} {'Val R@10':<10} {'Val NDCG':<10} "
          f"{'Test P@10':<10} {'Test R@10':<10} {'Test NDCG':<10}")
    print("-" * 80)

    models = {
        'Popularity': popularity_model,
        'Collaborative': best_cf,
        'ALS': als_model
    }

    for name, model in models.items():
        val_m = model.metadata['validation_metrics']
        test_m = model.metadata['test_metrics']
        print(f"{name:<20} "
              f"{val_m['precision@10']:<10.4f} "
              f"{val_m['recall@10']:<10.4f} "
              f"{val_m['ndcg@10']:<10.4f} "
              f"{test_m['precision@10']:<10.4f} "
              f"{test_m['recall@10']:<10.4f} "
              f"{test_m['ndcg@10']:<10.4f}")

    # Check for overfitting
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS")
    print("="*70)
    print("(Validation Score - Test Score) / Validation Score")
    print("-" * 40)

    for name, model in models.items():
        val_ndcg = model.metadata['validation_metrics']['ndcg@10']
        test_ndcg = model.metadata['test_metrics']['ndcg@10']
        overfit_pct = ((val_ndcg - test_ndcg) / val_ndcg) * 100 if val_ndcg > 0 else 0

        status = "✓ Good" if overfit_pct < 10 else "⚠ Overfitting" if overfit_pct < 20 else "❌ Severe Overfitting"
        print(f"{name:<20} {overfit_pct:>6.1f}%  {status}")

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("📊 Models saved with full train/val/test metrics")
    print("📁 Version: 0.3.0")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
