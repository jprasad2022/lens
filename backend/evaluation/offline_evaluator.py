"""
Offline Evaluation for Recommendation System
Implements chronological split, ranking metrics, and subpopulation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json
from pathlib import Path
from sklearn.metrics import ndcg_score, precision_score, recall_score


class OfflineEvaluator:
    """Performs offline evaluation with chronological split"""

    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.train_data = None
        self.test_data = None

    def load_data(self):
        """Load MovieLens data"""
        print("Loading data...")

        # Load ratings with timestamps
        ratings_file = self.data_path / "ratings.dat"
        self.ratings = pd.read_csv(
            ratings_file,
            sep='::',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')

        # Load movie metadata
        movies_file = self.data_path / "movies.dat"
        self.movies = pd.read_csv(
            movies_file,
            sep='::',
            names=['movie_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )

        print(f"Loaded {len(self.ratings)} ratings and {len(self.movies)} movies")

    def chronological_split(self, split_date: str = '2000-11-01'):
        """
        Split data chronologically to avoid data leakage
        This ensures we only use past data to predict future ratings
        """
        split_timestamp = pd.to_datetime(split_date)

        # IMPORTANT: Chronological split prevents leakage
        self.train_data = self.ratings[self.ratings['datetime'] < split_timestamp]
        self.test_data = self.ratings[self.ratings['datetime'] >= split_timestamp]

        print(f"Chronological split at {split_date}:")
        print(f"  Train: {len(self.train_data)} ratings ({len(self.train_data['user_id'].unique())} users)")
        print(f"  Test: {len(self.test_data)} ratings ({len(self.test_data['user_id'].unique())} users)")

        return self.train_data, self.test_data

    def evaluate_model(self, model, k_values=[5, 10, 20]):
        """Evaluate recommendation model on test set"""
        results = {
            'precision': {},
            'recall': {},
            'ndcg': {},
            'map': {},
            'coverage': 0
        }

        test_users = self.test_data['user_id'].unique()
        all_recommended_movies = set()

        precisions = {k: [] for k in k_values}
        recalls = {k: [] for k in k_values}
        ndcgs = {k: [] for k in k_values}
        aps = []

        for user_id in test_users[:100]:  # Sample for efficiency
            # Get ground truth (movies user actually liked in test set)
            user_test = self.test_data[self.test_data['user_id'] == user_id]
            ground_truth = set(user_test[user_test['rating'] >= 4]['movie_id'].values)

            if len(ground_truth) == 0:
                continue

            # Get recommendations from model (trained only on train data)
            try:
                recommendations = model.recommend(user_id, k=max(k_values))
                recommended_movies = [r['movie_id'] for r in recommendations]
                all_recommended_movies.update(recommended_movies)

                # Calculate metrics for different k values
                for k in k_values:
                    top_k = recommended_movies[:k]
                    hits = len(set(top_k) & ground_truth)

                    # Precision@k
                    precision_k = hits / k
                    precisions[k].append(precision_k)

                    # Recall@k
                    recall_k = hits / len(ground_truth) if len(ground_truth) > 0 else 0
                    recalls[k].append(recall_k)

                    # NDCG@k
                    relevance = [1 if movie_id in ground_truth else 0 for movie_id in top_k]
                    if sum(relevance) > 0:
                        ndcg = self._calculate_ndcg(relevance, k)
                        ndcgs[k].append(ndcg)

                # Average Precision
                ap = self._calculate_ap(recommended_movies, ground_truth)
                aps.append(ap)

            except Exception as e:
                print(f"Error for user {user_id}: {e}")
                continue

        # Aggregate results
        for k in k_values:
            results['precision'][f'P@{k}'] = np.mean(precisions[k]) if precisions[k] else 0
            results['recall'][f'R@{k}'] = np.mean(recalls[k]) if recalls[k] else 0
            results['ndcg'][f'NDCG@{k}'] = np.mean(ndcgs[k]) if ndcgs[k] else 0

        results['map']['MAP'] = np.mean(aps) if aps else 0
        results['coverage'] = len(all_recommended_movies) / len(self.movies)

        return results

    def subpopulation_analysis(self):
        """Analyze performance across different user groups"""
        # Group users by activity level
        user_activity = self.train_data.groupby('user_id').size()

        # Define subpopulations
        low_activity = user_activity[user_activity < 20].index
        medium_activity = user_activity[(user_activity >= 20) & (user_activity < 100)].index
        high_activity = user_activity[user_activity >= 100].index

        subpops = {
            'low_activity': low_activity,
            'medium_activity': medium_activity,
            'high_activity': high_activity
        }

        # Analyze genre preferences by user groups
        genre_analysis = {}
        for group_name, users in subpops.items():
            group_ratings = self.train_data[self.train_data['user_id'].isin(users)]
            # Get top-rated genres for this group
            group_ratings = group_ratings.merge(self.movies, on='movie_id')
            # This would need more processing to split genres
            genre_analysis[group_name] = {
                'user_count': len(users),
                'avg_ratings': group_ratings.groupby('user_id').size().mean()
            }

        return genre_analysis

    def _calculate_ndcg(self, relevance, k):
        """Calculate NDCG@k for a single user"""
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance[:k]))
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k]))
        return dcg / idcg if idcg > 0 else 0

    def _calculate_ap(self, recommendations, ground_truth):
        """Calculate Average Precision for a single user"""
        if not ground_truth:
            return 0

        hits = 0
        sum_precision = 0

        for i, movie_id in enumerate(recommendations):
            if movie_id in ground_truth:
                hits += 1
                sum_precision += hits / (i + 1)

        return sum_precision / len(ground_truth) if len(ground_truth) > 0 else 0

    def generate_report(self, results: Dict, subpop_results: Dict) -> str:
        """Generate offline evaluation report"""
        report = f"""
Offline Evaluation Report
========================

Methodology
-----------
- Chronological split to prevent data leakage
- Training data: Before 2000-11-01
- Test data: After 2000-11-01
- Evaluated on users with >= 4 star ratings in test period

Ranking Metrics
--------------
Precision@K (% of recommendations that were relevant):
  - P@5:  {results['precision']['P@5']:.3f}
  - P@10: {results['precision']['P@10']:.3f}
  - P@20: {results['precision']['P@20']:.3f}

Recall@K (% of relevant items that were recommended):
  - R@5:  {results['recall']['R@5']:.3f}
  - R@10: {results['recall']['R@10']:.3f}
  - R@20: {results['recall']['R@20']:.3f}

NDCG@K (ranking quality):
  - NDCG@5:  {results['ndcg']['NDCG@5']:.3f}
  - NDCG@10: {results['ndcg']['NDCG@10']:.3f}
  - NDCG@20: {results['ndcg']['NDCG@20']:.3f}

Mean Average Precision: {results['map']['MAP']:.3f}
Catalog Coverage: {results['coverage']:.2%}

Subpopulation Analysis
---------------------
"""
        for group, stats in subpop_results.items():
            report += f"\n{group}:"
            report += f"\n  Users: {stats['user_count']}"
            report += f"\n  Avg ratings per user: {stats['avg_ratings']:.1f}"

        return report


# Example usage script
if __name__ == "__main__":
    evaluator = OfflineEvaluator()
    evaluator.load_data()

    # Perform chronological split
    train, test = evaluator.chronological_split()

    # Here you would train your model on train data only
    # model = YourRecommendationModel()
    # model.train(train)

    # Evaluate (mock example)
    class MockModel:
        def recommend(self, user_id, k=10):
            # This would be your actual model
            return [{'movie_id': i} for i in range(1, k+1)]

    model = MockModel()
    results = evaluator.evaluate_model(model)
    subpop_results = evaluator.subpopulation_analysis()

    # Generate report
    report = evaluator.generate_report(results, subpop_results)
    print(report)

    # Save results
    with open('offline_evaluation_results.json', 'w') as f:
        json.dump({
            'metrics': results,
            'subpopulations': subpop_results
        }, f, indent=2)
