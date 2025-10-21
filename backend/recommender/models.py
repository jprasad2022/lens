"""
Minimal recommender model stubs for development.
Each model exposes async train() and predict() for compatibility.
"""

import asyncio
from typing import Any, Dict, List, Optional


class BaseModel:
    async def train(self, data_path) -> None:
        await asyncio.sleep(0)

    async def predict(self, user_id: int, k: int = 20, features: Optional[Dict[str, Any]] = None) -> List[int]:
        raise NotImplementedError


class PopularityModel(BaseModel):
    async def predict(self, user_id: int, k: int = 20, features: Optional[Dict[str, Any]] = None) -> List[int]:
        # Return deterministic top-k ids for stub
        return list(range(1, k + 1))


class CollaborativeFilteringModel(BaseModel):
    async def predict(self, user_id: int, k: int = 20, features: Optional[Dict[str, Any]] = None) -> List[int]:
        # Simple user-based deterministic ids
        start = (user_id % 1000) + 1
        return list(range(start, start + k))


class ALSModel(BaseModel):
    async def predict(self, user_id: int, k: int = 20, features: Optional[Dict[str, Any]] = None) -> List[int]:
        # Another deterministic pattern
        start = ((user_id * 7) % 1000) + 1
        return list(range(start, start + k))





