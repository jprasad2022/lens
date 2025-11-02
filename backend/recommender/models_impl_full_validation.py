"""
Recommender Models with Train/Validation/Test Split
Compatibility shim - redirect to the correct module for pickled models
"""
# For backward compatibility with pickled models
from recommender.models_impl import *

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import pickle
import time
import logging
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    precision: float
    recall: float
    ndcg: float
    rmse: float = 0.0

@dataclass
class DataSplits:
    """Container for train/val/test splits"""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

class ModelValidator:
    """Validation utilities with proper train/val/test splits"""

    @staticmethod
    def split_ratings(ratings_df: pd.DataFrame,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     random_state: int = 42) -> DataSplits:
        """Split ratings into train/validation/test sets by timestamp"""

        # Sort by timestamp
        ratings_df = ratings_df.sort_values('timestamp')

        # Calculate split points
        n_total = len(ratings_df)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val

        # Split by time (most realistic for recommender systems)
        train_df = ratings_df.iloc[:n_train]
        val_df = ratings_df.iloc[n_train:n_train + n_val]
        test_df = ratings_df.iloc[n_train + n_val:]

        # Ensure all sets have ratings from multiple users
        train_users = set(train_df['user_id'].unique())
        val_users = set(val_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())

        # Log statistics
        logger.info(f"Data split by timestamp:")
        logger.info(f"  Train: {len(train_df)} ratings, {len(train_users)} users")
        logger.info(f"  Val: {len(val_df)} ratings, {len(val_users)} users")
        logger.info(f"  Test: {len(test_df)} ratings, {len(test_users)} users")
        logger.info(f"  User overlap - Train∩Val: {len(train_users & val_users)}, Train∩Test: {len(train_users & test_users)}")

        return DataSplits(train=train_df, val=val_df, test=test_df)

    @staticmethod
    def split_ratings_by_user(ratings_df: pd.DataFrame,
                             val_size: float = 0.15,
                             test_size: float = 0.15,
                             random_state: int = 42) -> DataSplits:
        """Alternative: Split by holding out last interactions per user"""

        train_list = []
        val_list = []
        test_list = []

        for user_id, user_ratings in ratings_df.groupby('user_id'):
            # Sort user's ratings by timestamp
            user_ratings = user_ratings.sort_values('timestamp')
            n_ratings = len(user_ratings)

            if n_ratings < 10:
                # Users with few ratings go entirely to training
                train_list.append(user_ratings)
            else:
                # Calculate split indices
                n_test = max(1, int(n_ratings * test_size))
                n_val = max(1, int(n_ratings * val_size))

                # Split user's timeline
                train_end = n_ratings - n_test - n_val
                val_end = n_ratings - n_test

                train_list.append(user_ratings.iloc[:train_end])
                val_list.append(user_ratings.iloc[train_end:val_end])
                test_list.append(user_ratings.iloc[val_end:])

        # Combine all users
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

        logger.info(f"Data split by user timeline:")
        logger.info(f"  Train: {len(train_df)} ratings")
        logger.info(f"  Val: {len(val_df)} ratings")
        logger.info(f"  Test: {len(test_df)} ratings")

        return DataSplits(train=train_df, val=val_df, test=test_df)

    @staticmethod
    async def evaluate(model, eval_df: pd.DataFrame, train_df: pd.DataFrame, k: int = 10) -> ValidationMetrics:
        """Evaluate model on validation or test set"""
        if len(eval_df) == 0:
            return ValidationMetrics(0, 0, 0, 0)

        # Get items that exist in training set
        train_items = set(train_df['movie_id'].unique())

        # Group evaluation data by user
        user_items = defaultdict(list)
        user_ratings = defaultdict(dict)

        for _, row in eval_df.iterrows():
            user_items[row['user_id']].append(row['movie_id'])
            user_ratings[row['user_id']][row['movie_id']] = row['rating']

        # Calculate metrics
        precisions = []
        recalls = []
        ndcgs = []
        rmses = []

        # Sample users for efficiency
        eval_users = list(user_items.keys())
        sample_size = min(1000, len(eval_users))
        sample_users = np.random.choice(eval_users, sample_size, replace=False)

        for user_id in sample_users:
            true_items = user_items[user_id]
            true_ratings = user_ratings[user_id]

            # Only consider items rated 4+ as relevant
            relevant_items = [item for item in true_items if true_ratings[item] >= 4]

            if not relevant_items:
                continue

            try:
                # Get recommendations
                recommendations = await model.predict(user_id, k=k*2)  # Get more to filter

                # Filter to only items that exist in training set
                recommendations = [item for item in recommendations if item in train_items][:k]

                if not recommendations:
                    continue

                # Precision & Recall
                recommended_set = set(recommendations[:k])
                relevant_set = set(relevant_items)

                precision = len(recommended_set & relevant_set) / len(recommended_set)
                recall = len(recommended_set & relevant_set) / len(relevant_set)

                precisions.append(precision)
                recalls.append(recall)

                # NDCG
                dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommendations[:k])
                         if item in relevant_set)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)

                # RMSE for items in both recommendation and true ratings
                squared_errors = []
                for item in recommendations:
                    if item in true_ratings:
                        # Assume predicted rating is 4.0 for all recommendations
                        error = 4.0 - true_ratings[item]
                        squared_errors.append(error ** 2)

                if squared_errors:
                    rmses.append(np.sqrt(np.mean(squared_errors)))

            except Exception as e:
                logger.debug(f"Error evaluating user {user_id}: {e}")
                continue

        # Return average metrics
        return ValidationMetrics(
            precision=np.mean(precisions) if precisions else 0,
            recall=np.mean(recalls) if recalls else 0,
            ndcg=np.mean(ndcgs) if ndcgs else 0,
            rmse=np.mean(rmses) if rmses else 0
        )

class BaseRecommenderModel:
    """Base class with proper train/val/test methodology"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metadata = {
            "name": model_name,
            "version": "0.3.0",
            "trained_at": None,
            "training_time_seconds": 0,
            "model_size_bytes": 0,
            "metrics": {},
            "validation_metrics": {},
            "test_metrics": {},
            "training_history": [],
            "best_params": {}
        }
        self.validator = ModelValidator()

    async def train_validate_test(self, ratings_df: pd.DataFrame,
                                 split_method: str = "time") -> Dict[str, Any]:
        """Complete training pipeline with train/val/test splits"""

        # Split data
        if split_method == "time":
            splits = self.validator.split_ratings(ratings_df)
        else:
            splits = self.validator.split_ratings_by_user(ratings_df)

        # Train on training set
        logger.info(f"\nTraining {self.model_name} model...")
        start_time = time.time()
        await self.train(splits.train)
        train_time = time.time() - start_time

        # Validate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = await self.validator.evaluate(self, splits.val, splits.train)

        # Test on test set (final evaluation)
        logger.info("Final evaluation on test set...")
        test_metrics = await self.validator.evaluate(self, splits.test, splits.train)

        # Store all metrics
        self.metadata.update({
            "training_time_seconds": train_time,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split_method": split_method,
            "validation_metrics": {
                "precision@10": val_metrics.precision,
                "recall@10": val_metrics.recall,
                "ndcg@10": val_metrics.ndcg,
                "rmse": val_metrics.rmse
            },
            "test_metrics": {
                "precision@10": test_metrics.precision,
                "recall@10": test_metrics.recall,
                "ndcg@10": test_metrics.ndcg,
                "rmse": test_metrics.rmse
            }
        })

        # Log results
        logger.info(f"\nValidation Metrics:")
        for metric, value in self.metadata["validation_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info(f"\nTest Metrics (Final):")
        for metric, value in self.metadata["test_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

        return {
            "train_time": train_time,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train the model - to be implemented by subclasses"""
        raise NotImplementedError

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Generate recommendations - to be implemented by subclasses"""
        raise NotImplementedError

    def save(self, filepath: str):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class PopularityModel(BaseRecommenderModel):
    """Popularity model with train/val/test evaluation"""

    def __init__(self):
        super().__init__("popularity")
        self.popular_movies = []
        self.movie_scores = {}
        self.fallback_movies = []  # For completely new items

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train with time-weighted popularity"""
        start_time = time.time()

        # Global statistics for Bayesian smoothing
        global_mean_rating = ratings_df['rating'].mean()
        min_ratings_required = 5

        # Time decay (recent ratings matter more)
        current_time = ratings_df['timestamp'].max()
        time_decay_days = 365  # 1 year half-life

        movie_stats = []

        for movie_id, group in ratings_df.groupby('movie_id'):
            ratings = group['rating'].values
            timestamps = group['timestamp'].values

            # Time weights (exponential decay)
            days_ago = (current_time - timestamps) / 86400
            time_weights = np.exp(-days_ago / time_decay_days)

            # Weighted statistics
            weighted_sum = np.sum(ratings * time_weights)
            weight_sum = np.sum(time_weights)
            weighted_mean = weighted_sum / weight_sum if weight_sum > 0 else 0

            # Bayesian smoothing
            n = len(ratings)
            smoothed_rating = (
                (weighted_mean * n + global_mean_rating * min_ratings_required) /
                (n + min_ratings_required)
            )

            # Popularity score (considers both rating and frequency)
            # Using log to prevent extremely popular items from dominating
            popularity_score = smoothed_rating * np.log1p(n) * np.sqrt(weight_sum)

            movie_stats.append({
                'movie_id': movie_id,
                'n_ratings': n,
                'weighted_mean': weighted_mean,
                'smoothed_rating': smoothed_rating,
                'popularity_score': popularity_score,
                'recency_weight': np.mean(time_weights)
            })

        # Sort by popularity score
        movie_stats_df = pd.DataFrame(movie_stats)
        movie_stats_df = movie_stats_df.sort_values('popularity_score', ascending=False)

        self.popular_movies = movie_stats_df['movie_id'].tolist()
        self.movie_scores = dict(zip(
            movie_stats_df['movie_id'],
            movie_stats_df['popularity_score']
        ))

        # Store top movies as fallback
        self.fallback_movies = self.popular_movies[:100]

        # Update metadata
        self.metadata.update({
            "n_movies": len(self.popular_movies),
            "n_ratings": len(ratings_df),
            "time_decay_days": time_decay_days,
            "min_ratings_required": min_ratings_required,
            "global_mean_rating": global_mean_rating
        })


    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Return top-k popular movies"""
        return self.popular_movies[:k]


class CollaborativeFilteringModel(BaseRecommenderModel):
    """Collaborative filtering with proper evaluation"""

    def __init__(self, min_similarity: float = 0.1,
                 neighborhood_size: int = 50,
                 min_neighbors: int = 5):
        super().__init__("collaborative")
        self.min_similarity = min_similarity
        self.neighborhood_size = neighborhood_size
        self.min_neighbors = min_neighbors
        self.item_similarity = None
        self.user_item_matrix = None
        self.movie_idx_to_id = {}
        self.movie_id_to_idx = {}
        self.user_id_to_idx = {}
        self.global_mean_rating = 3.5
        self.item_bias = {}
        self.user_bias = {}

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train with bias terms and similarity threshold"""
        start_time = time.time()

        # Calculate global statistics
        self.global_mean_rating = ratings_df['rating'].mean()

        # Calculate biases
        for movie_id, group in ratings_df.groupby('movie_id'):
            self.item_bias[movie_id] = group['rating'].mean() - self.global_mean_rating

        for user_id, group in ratings_df.groupby('user_id'):
            self.user_bias[user_id] = group['rating'].mean() - self.global_mean_rating

        # Create mappings
        users = sorted(ratings_df['user_id'].unique())
        movies = sorted(ratings_df['movie_id'].unique())

        self.movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies)}
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users)}

        # Create matrix with bias-adjusted ratings
        row = [self.user_id_to_idx[uid] for uid in ratings_df['user_id']]
        col = [self.movie_id_to_idx[mid] for mid in ratings_df['movie_id']]

        # Adjust ratings by removing biases
        adjusted_ratings = []
        for _, r in ratings_df.iterrows():
            adj_rating = (r['rating'] - self.global_mean_rating -
                         self.user_bias.get(r['user_id'], 0) -
                         self.item_bias.get(r['movie_id'], 0))
            adjusted_ratings.append(adj_rating)

        self.user_item_matrix = csr_matrix(
            (adjusted_ratings, (row, col)),
            shape=(len(users), len(movies))
        )

        # Compute item-item similarity
        item_matrix = self.user_item_matrix.T.tocsr()

        # Normalize items before computing similarity
        item_norms = np.array(np.sqrt(item_matrix.multiply(item_matrix).sum(axis=1))).flatten()
        item_norms[item_norms == 0] = 1e-10

        # Compute similarities in chunks to save memory
        n_items = item_matrix.shape[0]
        chunk_size = 1000

        similarity_chunks = []
        for i in range(0, n_items, chunk_size):
            end_idx = min(i + chunk_size, n_items)
            chunk_sim = cosine_similarity(
                item_matrix[i:end_idx],
                item_matrix,
                dense_output=False
            )
            similarity_chunks.append(chunk_sim)

        # Combine chunks
        self.item_similarity = csr_matrix(np.vstack([
            chunk.toarray() for chunk in similarity_chunks
        ]))

        # Apply threshold by zeroing out small similarities
        if self.min_similarity > 0:
            # Get the data array
            data = self.item_similarity.data
            # Set values below threshold to 0
            data[np.abs(data) < self.min_similarity] = 0
            # Eliminate zeros to maintain sparsity
            self.item_similarity.eliminate_zeros()

        # Update metadata
        self.metadata.update({
            "n_users": len(users),
            "n_movies": len(movies),
            "n_ratings": len(ratings_df),
            "sparsity": 1 - (len(ratings_df) / (len(users) * len(movies))),
            "global_mean": self.global_mean_rating
        })

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Predict with fallback strategies"""
        if self.item_similarity is None:
            return []

        # Cold start - return popular items with high bias
        if user_id not in self.user_id_to_idx:
            popular_items = sorted(
                self.item_bias.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [item_id for item_id, _ in popular_items[:k]]

        # Get user data
        user_idx = self.user_id_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings != 0)[0]

        if len(rated_items) == 0:
            # No ratings - return popular items
            popular_items = sorted(
                self.item_bias.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [item_id for item_id, _ in popular_items[:k]]

        # Calculate scores for all items
        scores = np.zeros(self.user_item_matrix.shape[1])
        n_neighbors = np.zeros(self.user_item_matrix.shape[1])

        # For each unrated item
        for item_idx in range(self.user_item_matrix.shape[1]):
            if user_ratings[item_idx] != 0:  # Skip rated items
                scores[item_idx] = -np.inf
                continue

            # Get similarities with rated items
            item_sims = self.item_similarity[item_idx, rated_items].toarray().flatten()

            # Get top neighbors
            neighbor_indices = np.argsort(item_sims)[-self.neighborhood_size:][::-1]
            neighbor_indices = [rated_items[i] for i in neighbor_indices if item_sims[i] > 0]

            if len(neighbor_indices) >= self.min_neighbors:
                # Calculate weighted prediction
                weights = []
                ratings = []

                for neighbor_idx in neighbor_indices:
                    sim = self.item_similarity[item_idx, neighbor_idx]
                    if sim > 0:
                        weights.append(sim)
                        ratings.append(user_ratings[neighbor_idx])

                if weights:
                    # Weighted average
                    weighted_sum = sum(w * r for w, r in zip(weights, ratings))
                    weight_sum = sum(weights)

                    # Add back biases
                    prediction = (self.global_mean_rating +
                                self.user_bias.get(user_id, 0) +
                                self.item_bias.get(self.movie_idx_to_id[item_idx], 0) +
                                weighted_sum / weight_sum)

                    scores[item_idx] = prediction
                    n_neighbors[item_idx] = len(weights)

        # Get top k items
        # Prioritize items with enough neighbors
        valid_items = np.where(n_neighbors >= self.min_neighbors)[0]

        if len(valid_items) < k:
            # Add popular items to fill
            popular_scores = np.array([
                self.item_bias.get(self.movie_idx_to_id[idx], 0)
                for idx in range(len(scores))
            ])
            scores = np.where(scores == -np.inf, popular_scores, scores)

        top_items = np.argsort(scores)[-k:][::-1]
        recommendations = [
            self.movie_idx_to_id[idx]
            for idx in top_items
            if scores[idx] > -np.inf
        ]

        return recommendations[:k]


class ALSModel(BaseRecommenderModel):
    """ALS with cross-validation for hyperparameter tuning"""

    def __init__(self, factors: int = 100,
                 regularization: float = 0.1,
                 iterations: int = 30,
                 alpha: float = 40):
        super().__init__("als")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model = None
        self.user_item_matrix = None
        self.movie_idx_to_id = {}
        self.movie_id_to_idx = {}
        self.user_id_to_idx = {}
        self.popularity_fallback = []

    async def cross_validate(self, ratings_df: pd.DataFrame,
                           param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Perform cross-validation to find best hyperparameters"""

        # Create folds (3-fold CV)
        n_folds = 3
        fold_size = len(ratings_df) // n_folds

        best_params = {}
        best_score = 0

        # Grid search
        for factors in param_grid.get('factors', [self.factors]):
            for reg in param_grid.get('regularization', [self.regularization]):
                for alpha in param_grid.get('alpha', [self.alpha]):

                    fold_scores = []

                    for fold in range(n_folds):
                        # Create train/val split for this fold
                        val_start = fold * fold_size
                        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(ratings_df)

                        val_indices = list(range(val_start, val_end))
                        train_indices = list(range(0, val_start)) + list(range(val_end, len(ratings_df)))

                        train_fold = ratings_df.iloc[train_indices]
                        val_fold = ratings_df.iloc[val_indices]

                        # Train model with these params
                        self.factors = factors
                        self.regularization = reg
                        self.alpha = alpha

                        await self.train(train_fold)

                        # Evaluate
                        val_metrics = await self.validator.evaluate(
                            self, val_fold, train_fold, k=10
                        )
                        fold_scores.append(val_metrics.ndcg)

                    # Average across folds
                    avg_score = np.mean(fold_scores)

                    logger.info(f"CV: factors={factors}, reg={reg}, alpha={alpha} -> NDCG@10={avg_score:.4f}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'factors': factors,
                            'regularization': reg,
                            'alpha': alpha,
                            'cv_score': avg_score
                        }

        return best_params

    async def train(self, ratings_df: pd.DataFrame) -> None:
        """Train ALS model"""
        start_time = time.time()

        # Create mappings
        users = sorted(ratings_df['user_id'].unique())
        movies = sorted(ratings_df['movie_id'].unique())

        self.movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies)}
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies)}
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users)}

        # Create implicit feedback matrix
        row = [self.user_id_to_idx[uid] for uid in ratings_df['user_id']]
        col = [self.movie_id_to_idx[mid] for mid in ratings_df['movie_id']]

        # Convert ratings to implicit confidence
        # Higher ratings = higher confidence
        data = 1 + self.alpha * ratings_df['rating'].values

        self.user_item_matrix = csr_matrix(
            (data, (row, col)),
            shape=(len(users), len(movies))
        )

        # Calculate popularity for fallback
        item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        popular_indices = np.argsort(item_popularity)[-100:][::-1]
        self.popularity_fallback = [self.movie_idx_to_id[idx] for idx in popular_indices]

        # Train ALS
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False,
            random_state=42,
            calculate_training_loss=True
        )

        self.model.fit(self.user_item_matrix, show_progress=False)

        # Store training metrics
        self.metadata.update({
            "n_users": len(users),
            "n_movies": len(movies),
            "n_ratings": len(ratings_df),
            "model_params": {
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
                "alpha": self.alpha
            }
        })

    async def predict(self, user_id: int, k: int = 20) -> List[int]:
        """Generate recommendations"""
        if self.model is None:
            return []

        # Handle cold start
        if user_id not in self.user_id_to_idx:
            return self.popularity_fallback[:k]

        # Get recommendations
        user_idx = self.user_id_to_idx[user_id]

        try:
            # Get user factors and compute scores
            recommendations, scores = self.model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=k,
                filter_already_liked_items=True
            )

            return [self.movie_idx_to_id[idx] for idx in recommendations]

        except Exception as e:
            logger.debug(f"Error getting recommendations for user {user_id}: {e}")
            return self.popularity_fallback[:k]
