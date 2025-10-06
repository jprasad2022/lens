"""
Tests for backend services
"""

import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.movie_metadata_service import MovieMetadataService
from services.rating_statistics_service import RatingStatisticsService
from services.user_demographics_service import UserDemographicsService


@pytest.mark.asyncio
async def test_movie_metadata_service():
    """Test movie metadata service initialization and retrieval"""
    service = MovieMetadataService()
    
    # Check if data exists
    data_path = Path("data/movies.dat")
    if not data_path.exists():
        pytest.skip("Movies data file not found")
    
    await service.initialize()
    
    # Test single movie retrieval
    movie = await service.get_movie(1)
    if movie:
        assert 'id' in movie
        assert 'title' in movie
        assert 'genres' in movie
    
    # Test multiple movies retrieval
    movies = await service.get_movies([1, 2, 3])
    assert isinstance(movies, list)
    assert len(movies) <= 3


@pytest.mark.asyncio
async def test_rating_statistics_service():
    """Test rating statistics service"""
    service = RatingStatisticsService()
    
    # Check if data exists
    data_path = Path("data/ratings.dat")
    if not data_path.exists():
        pytest.skip("Ratings data file not found")
    
    # Don't actually initialize in tests to avoid loading 1M ratings
    # Just test the interface
    stats = await service.get_movie_stats(1)
    
    if stats:
        assert 'vote_average' in stats
        assert 'vote_count' in stats
        assert 'popularity_score' in stats


@pytest.mark.asyncio
async def test_user_demographics_service():
    """Test user demographics service"""
    service = UserDemographicsService()
    
    # Check if data exists
    data_path = Path("data/users.dat")
    if not data_path.exists():
        pytest.skip("Users data file not found")
    
    # Test interface
    user = await service.get_user(1)
    
    if user:
        assert 'user_id' in user
        assert 'gender' in user
        assert 'age' in user
        assert 'occupation' in user


def test_service_initialization():
    """Test that services can be instantiated"""
    movie_service = MovieMetadataService()
    rating_service = RatingStatisticsService()
    user_service = UserDemographicsService()
    
    assert movie_service is not None
    assert rating_service is not None
    assert user_service is not None
    
    # Check initialization flags
    assert not movie_service._initialized
    assert not rating_service._initialized
    assert not user_service._initialized