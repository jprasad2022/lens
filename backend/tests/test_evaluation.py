"""
Unit tests for evaluation modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.offline_evaluator import OfflineEvaluator
from quality.schema_validator import DataValidator, DriftDetector, BackpressureHandler


class TestOfflineEvaluator:
    """Test offline evaluation functionality"""
    
    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data"""
        return pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'movie_id': [100, 200, 100, 300, 200, 400],
            'rating': [4.0, 5.0, 3.0, 4.0, 5.0, 2.0],
            'timestamp': [
                1000000000,  # Before split
                1000000000,  # Before split
                1000000000,  # Before split
                2000000000,  # After split
                2000000000,  # After split
                2000000000   # After split
            ]
        })
    
    def test_chronological_split(self, sample_ratings, tmp_path):
        """Test chronological data split"""
        evaluator = OfflineEvaluator(str(tmp_path))
        evaluator.ratings = sample_ratings
        evaluator.ratings['datetime'] = pd.to_datetime(evaluator.ratings['timestamp'], unit='s')
        
        train, test = evaluator.chronological_split('2001-01-01')
        
        # Verify split
        assert len(train) == 3  # First 3 ratings
        assert len(test) == 3   # Last 3 ratings
        assert train['timestamp'].max() < test['timestamp'].min()
    
    def test_ndcg_calculation(self):
        """Test NDCG calculation"""
        evaluator = OfflineEvaluator()
        
        # Perfect ranking
        relevance = [1, 1, 0, 0]
        ndcg = evaluator._calculate_ndcg(relevance, k=4)
        assert ndcg == 1.0
        
        # Imperfect ranking
        relevance = [0, 1, 1, 0]
        ndcg = evaluator._calculate_ndcg(relevance, k=4)
        assert 0 < ndcg < 1.0
    
    def test_average_precision(self):
        """Test Average Precision calculation"""
        evaluator = OfflineEvaluator()
        
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {2, 4}
        
        ap = evaluator._calculate_ap(recommendations, ground_truth)
        expected_ap = (1/2 + 2/4) / 2  # hits at positions 2 and 4
        assert abs(ap - expected_ap) < 0.001


class TestDataValidator:
    """Test data validation functionality"""
    
    def test_valid_ratings(self):
        """Test validation of valid ratings data"""
        valid_ratings = pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': [100, 200, 300],
            'rating': [4.0, 5.0, 3.5],
            'timestamp': [1000000, 1000001, 1000002]
        })
        
        validator = DataValidator()
        result = validator.validate_ratings(valid_ratings)
        assert len(result) == 3
    
    def test_invalid_ratings(self):
        """Test validation catches invalid ratings"""
        invalid_ratings = pd.DataFrame({
            'user_id': [1, 2, 9999],  # user_id > 6040
            'movie_id': [100, 200, 300],
            'rating': [4.0, 5.0, 6.0],  # rating > 5.0
            'timestamp': [1000000, 1000001, 1000002]
        })
        
        validator = DataValidator()
        with pytest.raises(Exception):
            validator.validate_ratings(invalid_ratings)


class TestDriftDetector:
    """Test drift detection functionality"""
    
    def test_rating_drift_detection(self):
        """Test detection of rating distribution drift"""
        reference_data = pd.DataFrame({
            'rating': [3.0, 3.5, 4.0, 4.0, 4.5] * 20
        })
        
        # No drift
        new_data_no_drift = pd.DataFrame({
            'rating': [3.0, 3.5, 4.0, 4.0, 4.5] * 10
        })
        
        # With drift
        new_data_with_drift = pd.DataFrame({
            'rating': [1.0, 1.5, 2.0, 2.0, 2.5] * 10
        })
        
        detector = DriftDetector(reference_data)
        
        # Test no drift
        report_no_drift = detector.detect_drift(new_data_no_drift)
        assert not report_no_drift['rating_drift']['is_drifted']
        
        # Test with drift
        report_with_drift = detector.detect_drift(new_data_with_drift)
        assert report_with_drift['rating_drift']['is_drifted']
    
    def test_gini_calculation(self):
        """Test Gini coefficient calculation"""
        detector = DriftDetector(pd.DataFrame())
        
        # Perfect equality
        values_equal = [1, 1, 1, 1]
        gini_equal = detector._calculate_gini(values_equal)
        assert gini_equal < 0.1
        
        # High inequality
        values_unequal = [1, 1, 1, 100]
        gini_unequal = detector._calculate_gini(values_unequal)
        assert gini_unequal > 0.5


class TestBackpressureHandler:
    """Test backpressure handling"""
    
    def test_queue_throttling(self):
        """Test throttling based on queue size"""
        handler = BackpressureHandler(max_queue_size=3)
        
        # Add requests up to limit
        assert handler.add_request("req1")
        assert handler.add_request("req2")
        assert handler.add_request("req3")
        
        # Should throttle now
        assert handler.should_throttle()
        assert not handler.add_request("req4")
        
        # Complete one request
        handler.complete_request("req1", latency_ms=100)
        assert not handler.should_throttle()
        assert handler.add_request("req4")
    
    def test_latency_throttling(self):
        """Test throttling based on latency"""
        handler = BackpressureHandler(max_latency_ms=1000)
        
        # Add high latency requests
        for i in range(5):
            handler.add_request(f"req{i}")
            handler.complete_request(f"req{i}", latency_ms=2000)
        
        # Should throttle due to high latency
        assert handler.should_throttle()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])