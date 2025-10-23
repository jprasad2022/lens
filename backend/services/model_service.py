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
from recommender.models_impl import PopularityModel, CollaborativeFilteringModel, ALSModel
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
        print(f"Loading metadata from: {settings.model_registry_path}")
        metadata_files = list(settings.model_registry_path.glob("*/*/metadata.json"))
        print(f"Found {len(metadata_files)} metadata files")
        
        # Clear existing metadata to avoid duplicates
        self.model_metadata.clear()
        
        for i, metadata_file in enumerate(metadata_files):
            try:
                print(f"  [{i+1}/{len(metadata_files)}] Loading: {metadata_file}")
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    model_name = metadata['name']
                    print(f"    Model name: {model_name}")
                    print(f"    Current keys: {list(self.model_metadata.keys())}")
                    # Only add base model name, not versioned entries
                    if ':' not in model_name:
                        self.model_metadata[model_name] = metadata
                    print(f"    After loading keys: {list(self.model_metadata.keys())}")
            except Exception as e:
                print(f"Failed to load metadata from {metadata_file}: {e}")
        
        print(f"Final model_metadata keys: {list(self.model_metadata.keys())}")
    
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
                        loaded_data = pickle.load(f)
                    
                    # Debug: Log what was loaded
                    print(f"[MODEL DEBUG] Loaded object type: {type(loaded_data)}")
                    print(f"[MODEL DEBUG] Module: {loaded_data.__class__.__module__ if hasattr(loaded_data, '__class__') else 'N/A'}")
                    print(f"[MODEL DEBUG] Class: {loaded_data.__class__.__name__ if hasattr(loaded_data, '__class__') else 'N/A'}")
                    print(f"[MODEL DEBUG] Has user_id_to_idx: {hasattr(loaded_data, 'user_id_to_idx') if not isinstance(loaded_data, dict) else 'N/A'}")
                    
                    # Check if it's a dict (old format) or actual model object
                    if isinstance(loaded_data, dict):
                        # Old format - create a new model instance
                        model = await self._create_model_from_dict(model_name, loaded_data)
                    else:
                        model = loaded_data
                    
                    # Load metadata (but don't store it in model_metadata - it's already loaded)
                    metadata_path = model_path.parent / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    # Cache the model
                    self.models[cache_key] = model
                    
                    # Update app state
                    from app.state import app_state
                    # Use base model name for app state
                    base_model_name = model_name.split(':')[0]
                    await app_state.update_model_info(base_model_name, metadata)
                    
                    return model
                else:
                    return await self._create_default_model(model_name)
                    
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                raise
    
    async def _create_model_from_dict(self, model_name: str, data: Dict) -> Any:
        """Create a model instance from dictionary data"""
        print(f"Creating {model_name} model from dict data")
        
        # Create appropriate model based on name
        if model_name == "popularity":
            model = PopularityModel()
        elif model_name == "collaborative":
            model = CollaborativeFilteringModel()
        elif model_name == "als":
            model = ALSModel()
        else:
            model = PopularityModel()  # Default fallback
        
        # Set attributes from dict if available
        if model_name == "popularity":
            if 'top_items' in data:
                model.popular_movies = data['top_items']
            elif 'popular_movies' in data:
                model.popular_movies = data['popular_movies']
        elif model_name == "collaborative":
            if 'item_similarity' in data:
                model.item_similarity = data.get('item_similarity')
            if 'user_item_matrix' in data:
                model.user_item_matrix = data.get('user_item_matrix')
        elif model_name == "als":
            # ALS model attributes will be set if needed
            pass
        
        return model
    
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
        # Load model
        model = await self.load_model(model_name)
        
        # Check cache first
        if self.cache_service:
            cache_key = f"reco:{model_name}:{user_id}:{k}"
            cached = await self.cache_service.get(cache_key)
            if cached:
                return cached
        
        # Get predictions
        recommendations = await model.predict(user_id, k)
        
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
        seen_models = set()
        
        for model_name in self.model_metadata:
            metadata = self.model_metadata[model_name]
            # Extract base model name (remove :version suffix)
            base_name = metadata.get('name', model_name.split(':')[0])
            
            # Skip if we've already added this model
            if base_name in seen_models:
                continue
            
            seen_models.add(base_name)
            models.append({
                "name": base_name,
                "version": metadata.get('version', 'unknown'),
                "type": metadata.get('type', 'unknown'),
                "trained_at": metadata.get('trained_at'),
                "metrics": metadata.get('metrics', {}),
                "active": model_name in self.models or f"{base_name}:latest" in self.models
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