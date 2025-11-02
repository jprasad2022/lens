#!/usr/bin/env python3
"""
Train ALS (Alternating Least Squares) recommendation model
Requires the implicit library to be installed
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ALSModelTrainer:
    def __init__(self, data_dir="./data", model_dir="./model_registry"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.ml_path = self.data_dir

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import implicit
            logger.info(f"✓ implicit library found (version {getattr(implicit, '__version__', 'unknown')})")
            return True
        except ImportError as e:
            logger.error(f"❌ implicit library not found: {e}")
            logger.error("Please install it with: pip install implicit==0.7.2")
            return False

    def load_ratings_pandas(self):
        """Load ratings data using pandas"""
        logger.info("Loading ratings data...")

        # Load ratings from MovieLens format
        ratings_path = self.ml_path / 'ratings.dat'

        # MovieLens 1M format: UserID::MovieID::Rating::Timestamp
        ratings_df = pd.read_csv(
            ratings_path,
            sep='::',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python',
            encoding='latin-1'
        )

        logger.info(f"Loaded {len(ratings_df)} ratings")
        logger.info(f"Users: {ratings_df['user_id'].nunique()}, Movies: {ratings_df['movie_id'].nunique()}")

        return ratings_df

    def train_als_model(self, ratings_df):
        """Train ALS model using the same approach as models_impl.py"""
        from implicit.als import AlternatingLeastSquares

        logger.info("\n🏋️ Training ALS Model...")
        start_time = time.time()

        # Model parameters (matching models_impl.py)
        factors = 50
        iterations = 10
        regularization = 0.01

        # Create mappings
        users = ratings_df['user_id'].unique()
        movies = ratings_df['movie_id'].unique()

        movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies)}
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}

        # Create sparse matrix
        logger.info("Creating sparse user-item matrix...")
        row = [user_id_to_idx[uid] for uid in ratings_df['user_id']]
        col = [movie_id_to_idx[mid] for mid in ratings_df['movie_id']]
        data = ratings_df['rating'].values

        user_item_matrix = csr_matrix(
            (data, (row, col)),
            shape=(len(users), len(movies))
        )

        logger.info(f"Matrix shape: {user_item_matrix.shape}")
        logger.info(f"Matrix density: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4%}")

        # Initialize and train ALS model
        logger.info("Training ALS model...")
        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            use_gpu=False,
            random_state=42
        )

        # Train on implicit feedback (ratings as confidence)
        model.fit(user_item_matrix)

        training_time = time.time() - start_time
        logger.info(f"✓ ALS model trained in {training_time:.2f}s")

        # Create model dictionary to save
        model_data = {
            'type': 'als',
            'version': '0.1.0',
            'model': model,
            'user_item_matrix': user_item_matrix,
            'movie_idx_to_id': movie_idx_to_id,
            'user_id_to_idx': user_id_to_idx,
            'movie_id_to_idx': movie_id_to_idx,
            'factors': factors,
            'iterations': iterations,
            'regularization': regularization,
            'n_users': len(users),
            'n_movies': len(movies),
            'n_ratings': len(ratings_df),
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': training_time
        }

        return model_data

    def evaluate_model(self, model_data, ratings_df, n_samples=5):
        """Simple evaluation by generating sample recommendations"""
        logger.info("\n📊 Evaluating model with sample recommendations...")

        model = model_data['model']
        user_item_matrix = model_data['user_item_matrix']
        movie_idx_to_id = model_data['movie_idx_to_id']
        user_id_to_idx = model_data['user_id_to_idx']

        # Get sample users
        sample_users = ratings_df['user_id'].sample(n_samples).values

        for user_id in sample_users:
            if user_id in user_id_to_idx:
                user_idx = user_id_to_idx[user_id]

                # Get recommendations
                recommendations, scores = model.recommend(
                    user_idx,
                    user_item_matrix[user_idx],
                    N=5,
                    filter_already_liked_items=True
                )

                logger.info(f"\nUser {user_id} recommendations:")
                for rec_idx, score in zip(recommendations, scores):
                    movie_id = movie_idx_to_id[rec_idx]
                    logger.info(f"  Movie {movie_id}: score {score:.3f}")

    def save_model(self, model_data):
        """Save model to model registry"""
        model_type = 'als'
        version = model_data['version']
        model_path = self.model_dir / model_type / version
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model pickle
        logger.info(f"Saving model to {model_path}...")
        with open(model_path / 'model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            'name': model_type,
            'model_type': model_type,
            'version': version,
            'trained_at': model_data['training_date'],
            'framework': 'implicit',
            'algorithm': 'Alternating Least Squares',
            'parameters': {
                'factors': model_data['factors'],
                'iterations': model_data['iterations'],
                'regularization': model_data['regularization']
            },
            'metrics': {
                'n_users': model_data['n_users'],
                'n_movies': model_data['n_movies'],
                'n_ratings': model_data['n_ratings'],
                'training_time_seconds': model_data['training_time_seconds']
            }
        }

        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create model card
        model_card = f"""# ALS (Alternating Least Squares) Model

**Version**: {version}
**Trained**: {model_data['training_date']}
**Framework**: implicit {model_data.get('implicit_version', '0.7.2')}

## Description
Matrix factorization model using Alternating Least Squares algorithm for collaborative filtering recommendations.

## Parameters
- **Factors**: {model_data['factors']} (latent factors)
- **Iterations**: {model_data['iterations']}
- **Regularization**: {model_data['regularization']}

## Training Data
- **Dataset**: MovieLens 1M
- **Users**: {model_data['n_users']:,}
- **Movies**: {model_data['n_movies']:,}
- **Ratings**: {model_data['n_ratings']:,}
- **Training Time**: {model_data['training_time_seconds']:.2f} seconds

## Algorithm
ALS factorizes the user-item interaction matrix into two lower-rank matrices:
- User factors matrix (users × factors)
- Item factors matrix (items × factors)

The algorithm alternates between:
1. Fixing item factors and solving for user factors
2. Fixing user factors and solving for item factors

This process is repeated for the specified number of iterations.

## Usage
This model is automatically loaded by the LENS recommendation service.
It provides personalized recommendations based on collaborative filtering.
"""

        with open(model_path / 'model_card.md', 'w') as f:
            f.write(model_card)

        # Create latest.txt
        latest_file = self.model_dir / model_type / 'latest.txt'
        with open(latest_file, 'w') as f:
            f.write(version)

        logger.info(f"✓ Saved ALS model to {model_path}")

    def train(self):
        """Main training pipeline"""
        logger.info("=" * 60)
        logger.info("ALS Model Training Pipeline")
        logger.info("=" * 60)

        # Check dependencies
        try:
            from implicit.als import AlternatingLeastSquares
            logger.info("✓ implicit library imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import implicit: {e}")
            logger.error("Please ensure implicit is installed in your virtual environment")
            return False

        # Check data exists
        if not self.ml_path.exists():
            logger.error(f"❌ Data directory not found: {self.ml_path}")
            return False

        try:
            # Load data
            ratings_df = self.load_ratings_pandas()

            # Train model
            model_data = self.train_als_model(ratings_df)

            # Evaluate
            self.evaluate_model(model_data, ratings_df)

            # Save model
            self.save_model(model_data)

            logger.info("\n" + "=" * 60)
            logger.info("✅ ALS model training complete!")
            logger.info("=" * 60)

            # Show model registry structure
            logger.info("\nModel Registry Structure:")
            als_path = self.model_dir / 'als' / '0.1.0'
            if als_path.exists():
                logger.info("als/")
                logger.info("  0.1.0/")
                for file in als_path.iterdir():
                    size_kb = file.stat().st_size / 1024
                    if size_kb > 1024:
                        size_str = f"{size_kb/1024:.1f} MB"
                    else:
                        size_str = f"{size_kb:.1f} KB"
                    logger.info(f"    {file.name} ({size_str})")

            return True

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    # Parse arguments
    data_dir = "./data"
    model_dir = "./model_registry"

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python train_als_model.py [data_dir] [model_dir]")
        print("\nThis script trains an ALS (Alternating Least Squares) model for recommendations.")
        print("Requires the 'implicit' library to be installed.")
        sys.exit(0)

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        model_dir = sys.argv[2]

    trainer = ALSModelTrainer(data_dir, model_dir)
    success = trainer.train()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
