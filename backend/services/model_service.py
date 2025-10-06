"""
Model Service
Handles model loading, caching, and inference
"""

import asyncio
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from config.settings import get_settings
from recommender.models import PopularityModel, CollaborativeFilteringModel, ALSModel
from services.cache_service import CacheService

settings = get_settings()

class ModelService:
    """Service for managing recommendation models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.cache_service = CacheService() if settings.redis_enabled else None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize model service"""
        # Create model registry directory if it doesn't exist
        settings.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache service
        if self.cache_service:
            await self.cache_service.initialize()
        
        # Load available models metadata
        await self._load_metadata()
    
    async def _load_metadata(self):
        """Load model metadata from registry"""
        metadata_files = settings.model_registry_path.glob("*/metadata.json")
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    model_name = metadata['name']
                    self.model_metadata[model_name] = metadata
            except Exception as e:
                print(f"Failed to load metadata from {metadata_file}: {e}")
    
    async def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load a model into memory"""
        async with self._lock:
            # Check if already loaded
            cache_key = f"{model_name}:{version or 'latest'}"
            if cache_key in self.models:
                return self.models[cache_key]
            
            try:
                # Determine model path
                if version:
                    model_path = settings.model_registry_path / model_name / version / "model.pkl"
                else:
                    # Find latest version
                    model_dir = settings.model_registry_path / model_name
                    if model_dir.exists():
                        # Check for latest.txt (Windows) or latest symlink (Unix)
                        latest_txt = model_dir / "latest.txt"
                        latest_link = model_dir / "latest"
                        
                        if latest_txt.exists():
                            # Windows: read version from text file
                            with open(latest_txt, 'r') as f:
                                version = f.read().strip()
                            model_path = model_dir / version / "model.pkl"
                        elif latest_link.exists():
                            # Unix: follow symlink
                            version = latest_link.readlink().name
                            model_path = model_dir / version / "model.pkl"
                        else:
                            # Fallback: find newest version
                            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
                            if versions:
                                version = versions[-1]
                                model_path = model_dir / version / "model.pkl"
                            else:
                                raise ValueError(f"No versions found for model {model_name}")
                    else:
                        # Model doesn't exist, create a default one
                        return await self._create_default_model(model_name)
                
                # Load model from disk
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load metadata
                    metadata_path = model_path.parent / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            # Ensure minimal fields
                            metadata.setdefault('name', model_name)
                            metadata.setdefault('version', version or 'latest')
                            metadata.setdefault('type', model_name)
                            metadata.setdefault('trained_at', datetime.utcnow().isoformat())
                            metadata.setdefault('metrics', {})
                            metadata.setdefault('parameters', {})
                            self.model_metadata[cache_key] = metadata
                    
                    # Cache the model
                    self.models[cache_key] = model
                    
                    # Update app state
                    from app.state import app_state
                    # Convert trained_at to datetime where possible
                    trained_at = metadata.get('trained_at')
                    try:
                        from datetime import datetime as _dt
                        if isinstance(trained_at, str):
                            trained_at_dt = _dt.fromisoformat(trained_at.replace('Z', '+00:00'))
                        elif trained_at is None:
                            trained_at_dt = _dt.utcnow()
                        else:
                            trained_at_dt = trained_at
                    except Exception:
                        trained_at_dt = datetime.utcnow()

                    await app_state.update_model_info(model_name, {
                        'name': metadata.get('name', model_name),
                        'version': metadata.get('version', version or 'latest'),
                        'type': metadata.get('type', model_name),
                        'trained_at': trained_at_dt,
                        'metrics': metadata.get('metrics', {}),
                        'parameters': metadata.get('parameters', {}),
                        'active': True,
                    })
                    
                    return model
                else:
                    return await self._create_default_model(model_name)
                    
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise
    
    async def _create_default_model(self, model_name: str) -> Any:
        """Create a default model if none exists"""
        print(f"Creating default model for {model_name}")
        
        # Create appropriate model based on name
        if model_name == "popularity":
            model = PopularityModel()
        elif model_name == "collaborative":
            model = CollaborativeFilteringModel()
        elif model_name == "als":
            model = ALSModel()
        else:
            model = PopularityModel()  # Default fallback
        
        # Train on sample data if available
        movielens_path = settings.movielens_path
        if movielens_path.exists():
            # Load and train model
            await model.train(movielens_path)
            
            # Save model
            await self.save_model(model_name, model, {"type": model_name, "version": "v0.1"})
        
        # Cache the model
        self.models[f"{model_name}:latest"] = model
        
        return model
    
    async def save_model(self, name: str, model: Any, metadata: Dict[str, Any]):
        """Save a model to the registry"""
        # Generate version
        version = metadata.get('version', f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        # Create directory
        model_dir = settings.model_registry_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata['name'] = name
        metadata['version'] = version
        metadata['saved_at'] = datetime.utcnow().isoformat()
        metadata['path'] = str(model_path)
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update metadata cache
        self.model_metadata[f"{name}:{version}"] = metadata
        
        return version
    
    async def get_recommendations(
        self, 
        model_name: str,
        user_id: int,
        k: int = 20,
        features: Optional[Dict] = None
    ) -> List[int]:
        """Get recommendations from a model"""
        try:
            # Load model
            model = await self.load_model(model_name)
            
            # Check cache first
            if self.cache_service:
                cache_key = f"reco:{model_name}:{user_id}:{k}"
                cached = await self.cache_service.get(cache_key)
                if cached:
                    return cached
            
            # Get predictions
            if isinstance(model, dict):
                # Handle dictionary-based models
                if model.get('type') == 'popularity':
                    # Use top_items from the model
                    top_items = model.get('top_items', [])
                    if not top_items:
                        print(f"WARNING: No top_items found in popularity model, using fallback")
                        recommendations = list(range(1, k + 1))
                    else:
                        # top_items is already a list of movie IDs
                        recommendations = top_items[:k]
                else:
                    # Fallback for other dictionary models
                    recommendations = list(range(1, k + 1))
            else:
                # Handle object-based models
                recommendations = await model.predict(user_id, k, features)
        except Exception as e:
            print(f"ERROR in get_recommendations: {e}")
            print(f"Model name: {model_name}, User ID: {user_id}, k: {k}")
            raise
        
        # Cache results
        if self.cache_service:
            await self.cache_service.set(
                cache_key, 
                recommendations,
                ttl=settings.response_cache_ttl
            )
        
        return recommendations
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        models = []
        
        for model_name in self.model_metadata:
            metadata = self.model_metadata[model_name]
            models.append({
                "name": metadata.get('name', model_name),
                "version": metadata.get('version', 'unknown'),
                "type": metadata.get('type', 'unknown'),
                "trained_at": metadata.get('trained_at'),
                "metrics": metadata.get('metrics', {}),
                "active": model_name in self.models
            })
        
        return models
    
    async def health_check(self) -> Dict[str, Any]:
        """Check model service health"""
        return {
            "healthy": True,
            "models_loaded": len(self.models),
            "models_available": len(self.model_metadata),
            "cache_enabled": self.cache_service is not None
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.cache_service:
            await self.cache_service.close()
        
        # Clear models from memory
        self.models.clear()
        self.model_metadata.clear()