"""
Real Recommender Model Implementations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import pickle
import time
import logging
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares
import asyncio

logger = logging.getLogger(__name__)

class BaseRecommenderModel:
    """Base class for recommender models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metadata = {
            "name": model_name,
            "version": "0.1.0",
            "trained_at": None,
            "training_time_seconds": 0,
            "model_size_bytes": 0,
            "metrics": {},
        }

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train the model."""
        raise NotImplementedError

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Generate recommendations."""
        raise NotImplementedError

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_model_size(self) -> int:
        """Get model size in bytes."""
        import sys
        return sys.getsizeof(self)

class PopularityModel(BaseRecommenderModel):
    """Simple popularity-based recommender."""

    def __init__(self):
        super().__init__("popularity")
        self.popular_movies = []

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train by computing movie popularity."""
        start_time = time.time()

        # Compute popularity scores
        movie_stats = ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'rating_count', 'avg_rating']

        # Weighted score: combination of rating count and average rating
        movie_stats['popularity_score'] = (
            movie_stats['rating_count'] * 0.7 +
            movie_stats['avg_rating'] * movie_stats['rating_count'] * 0.3
        )

        # Sort by popularity
        movie_stats = movie_stats.sort_values('popularity_score', ascending=False)
        self.popular_movies = movie_stats['movie_id'].tolist()

        # Update metadata
        self.metadata.update({
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": time.time() - start_time,
            "n_movies": len(self.popular_movies),
            "n_ratings": len(ratings_df),
        })

        logger.info(f"Popularity model trained in {self.metadata['training_time_seconds']:.2f}s")

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Return top-k popular movies."""
        return self.popular_movies[:k]

class CollaborativeFilteringModel(BaseRecommenderModel):
    """Item-based collaborative filtering."""

    def __init__(self):
        super().__init__("collaborative")
        self.item_similarity = None
        self.user_item_matrix = None
        self.movie_idx_to_id = {}
        self.movie_id_to_idx = {}
        self.user_id_to_idx = {}

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train item-based collaborative filtering model."""
        start_time = time.time()

        # Create user-item matrix
        users = ratings_df['user_id'].unique()
        movies = ratings_df['movie_id'].unique()

        # Create mappings
        self.movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies)}
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users)}

        # Create sparse matrix
        row = [self.user_id_to_idx[uid] for uid in ratings_df['user_id']]
        col = [self.movie_id_to_idx[mid] for mid in ratings_df['movie_id']]
        data = ratings_df['rating'].values

        self.user_item_matrix = csr_matrix(
            (data, (row, col)),
            shape=(len(users), len(movies))
        )

        # Compute item-item similarity
        item_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix, dense_output=False)

        # Update metadata
        self.metadata.update({
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": time.time() - start_time,
            "n_users": len(users),
            "n_movies": len(movies),
            "n_ratings": len(ratings_df),
            "sparsity": 1 - (len(ratings_df) / (len(users) * len(movies))),
        })

        logger.info(f"Collaborative filtering model trained in {self.metadata['training_time_seconds']:.2f}s")

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Generate recommendations using item-based CF."""
        if self.item_similarity is None:
            return []

        # Check if user exists
        if user_id not in self.user_id_to_idx:
            # Return popular items for cold start users
            return list(self.movie_idx_to_id.values())[:k]

        # Get user's rated movies
        user_idx = self.user_id_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            # Return popular items for cold start users
            return list(self.movie_idx_to_id.values())[:k]

        # Compute scores for unrated items
        scores = np.zeros(self.item_similarity.shape[0])
        for item_idx in rated_items:
            scores += self.item_similarity[item_idx].toarray().flatten() * user_ratings[item_idx]

        # Remove already rated items
        scores[rated_items] = -np.inf

        # Get top-k items
        top_items = np.argsort(scores)[-k:][::-1]

        # Convert to movie IDs
        recommendations = [self.movie_idx_to_id[idx] for idx in top_items]
        return recommendations

class ALSModel(BaseRecommenderModel):
    """Alternating Least Squares matrix factorization."""

    def __init__(self, factors=50, iterations=10, regularization=0.01):
        super().__init__("als")
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.user_item_matrix = None
        self.movie_idx_to_id = {}
        self.user_id_to_idx = {}

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train ALS model."""
        start_time = time.time()

        # Create mappings
        users = ratings_df['user_id'].unique()
        movies = ratings_df['movie_id'].unique()

        self.movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies)}
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}

        # Create sparse matrix
        row = [self.user_id_to_idx[uid] for uid in ratings_df['user_id']]
        col = [movie_id_to_idx[mid] for mid in ratings_df['movie_id']]
        data = ratings_df['rating'].values

        self.user_item_matrix = csr_matrix(
            (data, (row, col)),
            shape=(len(users), len(movies))
        )

        # Initialize and train ALS model
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False,
            random_state=42
        )

        # Train on implicit feedback (ratings as confidence)
        self.model.fit(self.user_item_matrix)

        # Update metadata
        self.metadata.update({
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": time.time() - start_time,
            "n_users": len(users),
            "n_movies": len(movies),
            "n_ratings": len(ratings_df),
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
        })

        logger.info(f"ALS model trained in {self.metadata['training_time_seconds']:.2f}s")

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Generate recommendations using ALS."""
        if self.model is None:
            return []

        # Check if user exists
        if user_id not in self.user_id_to_idx:
            # Cold start - return popular items
            return list(self.movie_idx_to_id.values())[:k]

        user_idx = self.user_id_to_idx[user_id]

        # Get recommendations
        recommendations, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=k,
            filter_already_liked_items=True
        )

        # Convert to movie IDs
        movie_ids = [self.movie_idx_to_id[idx] for idx in recommendations]
        return movie_ids

# Neural Collaborative Filtering can be added here if needed
class NeuralCFModel(BaseRecommenderModel):
    """Neural Collaborative Filtering (placeholder for now)."""

    def __init__(self):
        super().__init__("neural_cf")

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train neural CF model."""
        # This would require TensorFlow/PyTorch implementation
        pass

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Generate recommendations using neural CF."""
        # Placeholder
        return []
