"""
Application State Management
Singleton pattern for shared state across the application
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

class AppState:
    """Global application state"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Core services
        self.model_service: Optional[Any] = None
        self.kafka_service: Optional[Any] = None
        self.redis_client: Optional[Any] = None
        self.movie_metadata_service: Optional[Any] = None
        self.rating_statistics_service: Optional[Any] = None
        self.user_demographics_service: Optional[Any] = None
        self.ab_switch_service: Optional[Any] = None

        # Background tasks
        self.consumer_task: Optional[asyncio.Task] = None
        self.retrain_task: Optional[asyncio.Task] = None

        # State tracking
        self.startup_time: datetime = datetime.utcnow()
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_model_update: Optional[datetime] = None
        self.active_models: Dict[str, Any] = {}

        # A/B testing state
        self.ab_test_active: bool = False
        self.ab_test_models: List[str] = []
        self.ab_test_results: Dict[str, Dict] = {}

        # Locks for thread safety
        self._model_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()

        self._initialized = True

    async def get_health_status(self) -> Dict[str, Any]:
        """Get application health status"""
        health = {
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
            "healthy": True,
            "checks": {}
        }

        # Check model service
        if self.model_service:
            try:
                model_health = await self.model_service.health_check()
                health["checks"]["model_service"] = model_health
                if not model_health.get("healthy", False):
                    health["healthy"] = False
            except Exception as e:
                health["checks"]["model_service"] = {
                    "healthy": False,
                    "error": str(e)
                }
                health["healthy"] = False

        # Check Kafka service
        if self.kafka_service:
            try:
                kafka_health = await self.kafka_service.health_check()
                health["checks"]["kafka"] = kafka_health
                if not kafka_health.get("healthy", False):
                    health["healthy"] = False
            except Exception as e:
                health["checks"]["kafka"] = {
                    "healthy": False,
                    "error": str(e)
                }

        # Check Redis
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health["checks"]["redis"] = {"healthy": True}
            except Exception as e:
                health["checks"]["redis"] = {
                    "healthy": False,
                    "error": str(e)
                }

        # Add metrics
        health["metrics"] = {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "active_models": len(self.active_models),
            "last_model_update": self.last_model_update.isoformat() if self.last_model_update else None
        }

        return health

    async def increment_request_count(self):
        """Thread-safe request count increment"""
        async with self._metrics_lock:
            self.request_count += 1

    async def increment_error_count(self):
        """Thread-safe error count increment"""
        async with self._metrics_lock:
            self.error_count += 1

    async def update_model_info(self, model_name: str, model_info: Dict[str, Any]):
        """Update active model information"""
        async with self._model_lock:
            self.active_models[model_name] = model_info
            self.last_model_update = datetime.utcnow()

    async def get_current_model(self, user_id: Optional[int] = None) -> str:
        """Get current model based on A/B testing or default"""
        # Use AB switch service if available
        if self.ab_switch_service and user_id:
            from services.ab_switch_service import get_ab_switch_service
            ab_service = get_ab_switch_service()
            return ab_service.get_model_for_user(user_id)
        
        # Fallback to simple A/B test assignment
        if self.ab_test_active and self.ab_test_models and user_id:
            # Simple A/B test assignment based on user ID
            model_index = user_id % len(self.ab_test_models)
            return self.ab_test_models[model_index]

        # Return default model
        from config.settings import get_settings
        return get_settings().default_model

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "active_models": len(self.active_models),
            "ab_test_active": self.ab_test_active,
        }

# Global instance
app_state = AppState()
