"""
Schema Validation using Pandera
Ensures data quality throughout the pipeline
"""

import pandera as pa
from pandera import DataFrameSchema, Column, Check, Index
import pandas as pd
from typing import Dict, Any


# Define schemas for data validation

# Ratings schema
ratings_schema = DataFrameSchema({
    "user_id": Column(int, Check.greater_than(0), Check.less_than_or_equal_to(6040)),
    "movie_id": Column(int, Check.greater_than(0)),
    "rating": Column(float, Check.in_range(0.5, 5.0)),
    "timestamp": Column(int, Check.greater_than(0))
})

# Movies schema
movies_schema = DataFrameSchema({
    "movie_id": Column(int, Check.greater_than(0), unique=True),
    "title": Column(str, Check.str_length(min_value=1)),
    "genres": Column(str)
})

# Recommendation request schema
reco_request_schema = pa.DataFrameSchema({
    "user_id": Column(int, Check.greater_than(0)),
    "k": Column(int, Check.in_range(1, 100)),
    "model": Column(str, Check.isin(["als", "collaborative", "popularity", "content", "hybrid", "neural"])),
    "timestamp": Column(pd.Timestamp)
})

# Recommendation response schema  
reco_response_schema = pa.DataFrameSchema({
    "user_id": Column(int),
    "movie_id": Column(int),
    "rank": Column(int, Check.greater_than(0)),
    "score": Column(float, Check.greater_than_or_equal_to(0)),
    "model_version": Column(str)
})


class DataValidator:
    """Validates data against predefined schemas"""
    
    @staticmethod
    def validate_ratings(df: pd.DataFrame) -> pd.DataFrame:
        """Validate ratings data"""
        try:
            return ratings_schema.validate(df)
        except pa.errors.SchemaError as e:
            print(f"Ratings validation failed: {e}")
            raise
    
    @staticmethod
    def validate_movies(df: pd.DataFrame) -> pd.DataFrame:
        """Validate movies data"""
        try:
            return movies_schema.validate(df)
        except pa.errors.SchemaError as e:
            print(f"Movies validation failed: {e}")
            raise
    
    @staticmethod
    def validate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
        """Validate recommendation responses"""
        try:
            return reco_response_schema.validate(df)
        except pa.errors.SchemaError as e:
            print(f"Recommendations validation failed: {e}")
            raise


# Drift Detection
class DriftDetector:
    """Detects distribution drift in data"""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for drift detection"""
        stats = {}
        
        # For ratings data
        if 'rating' in data.columns:
            stats['rating_mean'] = data['rating'].mean()
            stats['rating_std'] = data['rating'].std()
            stats['rating_distribution'] = data['rating'].value_counts(normalize=True).to_dict()
        
        # User activity statistics
        if 'user_id' in data.columns:
            user_counts = data['user_id'].value_counts()
            stats['user_activity_mean'] = user_counts.mean()
            stats['user_activity_std'] = user_counts.std()
            stats['active_users'] = len(user_counts)
        
        # Movie popularity statistics
        if 'movie_id' in data.columns:
            movie_counts = data['movie_id'].value_counts()
            stats['movie_popularity_mean'] = movie_counts.mean()
            stats['movie_popularity_std'] = movie_counts.std()
            stats['unique_movies'] = len(movie_counts)
            
            # Gini coefficient for movie popularity
            stats['movie_gini'] = self._calculate_gini(movie_counts.values)
        
        return stats
    
    def _calculate_gini(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        sorted_values = sorted(values)
        n = len(values)
        cumsum = 0
        for i, value in enumerate(sorted_values):
            cumsum += (n - i) * value
        return (n + 1 - 2 * cumsum / sum(sorted_values)) / n
    
    def detect_drift(self, new_data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """Detect drift between reference and new data"""
        new_stats = self._compute_stats(new_data)
        drift_report = {}
        
        # Check rating distribution drift
        if 'rating_mean' in new_stats and 'rating_mean' in self.reference_stats:
            rating_drift = abs(new_stats['rating_mean'] - self.reference_stats['rating_mean']) / self.reference_stats['rating_mean']
            drift_report['rating_drift'] = {
                'drift_score': rating_drift,
                'is_drifted': rating_drift > threshold,
                'reference_mean': self.reference_stats['rating_mean'],
                'current_mean': new_stats['rating_mean']
            }
        
        # Check user activity drift
        if 'user_activity_mean' in new_stats and 'user_activity_mean' in self.reference_stats:
            activity_drift = abs(new_stats['user_activity_mean'] - self.reference_stats['user_activity_mean']) / self.reference_stats['user_activity_mean']
            drift_report['user_activity_drift'] = {
                'drift_score': activity_drift,
                'is_drifted': activity_drift > threshold,
                'reference_mean': self.reference_stats['user_activity_mean'],
                'current_mean': new_stats['user_activity_mean']
            }
        
        # Check movie popularity drift (using Gini coefficient)
        if 'movie_gini' in new_stats and 'movie_gini' in self.reference_stats:
            gini_drift = abs(new_stats['movie_gini'] - self.reference_stats['movie_gini'])
            drift_report['popularity_concentration_drift'] = {
                'drift_score': gini_drift,
                'is_drifted': gini_drift > 0.05,  # Gini is 0-1, so smaller threshold
                'reference_gini': self.reference_stats['movie_gini'],
                'current_gini': new_stats['movie_gini']
            }
        
        drift_report['summary'] = {
            'has_drift': any(metric.get('is_drifted', False) for metric in drift_report.values()),
            'drift_metrics': len([m for m in drift_report.values() if m.get('is_drifted', False)])
        }
        
        return drift_report


# Backpressure Handler
class BackpressureHandler:
    """Handles system backpressure"""
    
    def __init__(self, max_queue_size: int = 1000, max_latency_ms: int = 5000):
        self.max_queue_size = max_queue_size
        self.max_latency_ms = max_latency_ms
        self.request_queue = []
        self.latencies = []
    
    def should_throttle(self) -> bool:
        """Determine if system should throttle requests"""
        # Check queue size
        if len(self.request_queue) > self.max_queue_size:
            return True
        
        # Check average latency
        if self.latencies:
            avg_latency = sum(self.latencies[-100:]) / len(self.latencies[-100:])
            if avg_latency > self.max_latency_ms:
                return True
        
        return False
    
    def add_request(self, request_id: str):
        """Add request to queue"""
        if not self.should_throttle():
            self.request_queue.append(request_id)
            return True
        return False
    
    def complete_request(self, request_id: str, latency_ms: float):
        """Mark request as completed"""
        if request_id in self.request_queue:
            self.request_queue.remove(request_id)
        self.latencies.append(latency_ms)


# Example usage
if __name__ == "__main__":
    # Test schema validation
    test_ratings = pd.DataFrame({
        'user_id': [1, 2, 3],
        'movie_id': [100, 200, 300],
        'rating': [4.0, 5.0, 3.5],
        'timestamp': [1000000, 1000001, 1000002]
    })
    
    validator = DataValidator()
    validated_ratings = validator.validate_ratings(test_ratings)
    print("✓ Ratings schema validation passed")
    
    # Test drift detection
    reference_data = test_ratings.copy()
    new_data = test_ratings.copy()
    new_data['rating'] = new_data['rating'] + 0.5  # Simulate drift
    
    detector = DriftDetector(reference_data)
    drift_report = detector.detect_drift(new_data)
    print(f"\nDrift Detection Report:")
    print(json.dumps(drift_report, indent=2))