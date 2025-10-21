"""
Simple in-memory async cache service used when Redis is disabled.
"""

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, timedelta


class CacheService:
    """Lightweight cache with TTL semantics for local development."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()
            self._expiry.clear()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            exp = self._expiry.get(key)
            if exp and exp < datetime.utcnow():
                self._store.pop(key, None)
                self._expiry.pop(key, None)
                return None
            return self._store.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        async with self._lock:
            self._store[key] = value
            self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)





