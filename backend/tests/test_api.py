"""
Basic API tests for the recommendation service
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_recommend_endpoint():
    """Test the recommendation endpoint"""
    response = client.get("/recommend/1?k=10")
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 10
    assert data["user_id"] == 1


def test_recommend_with_model_param():
    """Test recommendation with specific model"""
    response = client.get("/recommend/1?k=5&model=popularity")
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) <= 5
    assert data["model_info"]["name"] == "popularity"


def test_recommend_invalid_user():
    """Test recommendation with invalid user ID"""
    response = client.get("/recommend/0?k=10")
    assert response.status_code == 400  # Assuming we validate user_id > 0


def test_recommend_invalid_k():
    """Test recommendation with invalid k value"""
    response = client.get("/recommend/1?k=0")
    assert response.status_code == 422  # FastAPI validation error


def test_models_endpoint():
    """Test the models listing endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_models_current_endpoint():
    """Test the current model endpoint"""
    response = client.get("/models/current")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert data["model"] is not None