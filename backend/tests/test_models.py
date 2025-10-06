"""
Tests for recommendation models
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommender.models import PopularityModel, CollaborativeFilteringModel, ALSModel


@pytest.mark.asyncio
async def test_popularity_model():
    """Test popularity model predictions"""
    model = PopularityModel()
    predictions = await model.predict(user_id=1, k=5)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 5
    assert all(isinstance(item, int) for item in predictions)


@pytest.mark.asyncio
async def test_collaborative_filtering_model():
    """Test collaborative filtering model"""
    model = CollaborativeFilteringModel()
    predictions = await model.predict(user_id=100, k=10)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 10
    assert all(isinstance(item, int) for item in predictions)


@pytest.mark.asyncio  
async def test_als_model():
    """Test ALS model"""
    model = ALSModel()
    predictions = await model.predict(user_id=50, k=20)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 20
    assert all(isinstance(item, int) for item in predictions)


def test_model_deterministic():
    """Test that models give consistent results"""
    model = PopularityModel()
    
    # Same user, same k should give same results
    pred1 = model.predict(user_id=1, k=5)
    pred2 = model.predict(user_id=1, k=5)
    
    # For now these are stubs, so they should be identical
    # In real implementation, we'd set random seed
    assert pred1 == pred2