"""
Kafka service factory - returns real or stub implementation based on configuration.
"""

import os
from typing import Any, Dict, List


class KafkaServiceStub:
    """No-op Kafka service used for local development."""

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "connected": False}

    async def produce_reco_response(
        self,
        user_id: int,
        request_id: str,
        status: int,
        latency_ms: float,
        movie_ids: List[int],
        model_version: str,
        cached: bool,
    ) -> None:
        return None

    async def produce_feedback(self, payload: Dict[str, Any]) -> None:
        return None

    async def produce_reco_request(
        self,
        user_id: int,
        request_id: str,
        model: str,
        k: int,
    ) -> None:
        return None

    async def produce_watch_event(self, user_id: int, movie_id: int, progress: float) -> None:
        return None

    async def produce_rate_event(self, user_id: int, movie_id: int, rating: float) -> None:
        return None

    async def produce_interaction_event(self, payload: Dict[str, Any]) -> None:
        return None


def get_kafka_service():
    """Factory function to get appropriate Kafka service."""
    # Check if Kafka is configured
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS") and os.getenv("KAFKA_BOOTSTRAP_SERVERS") != "localhost:9092":
        # Use real implementation
        from services.kafka_service_impl import KafkaServiceImpl
        return KafkaServiceImpl()
    else:
        # Use stub for local development
        return KafkaServiceStub()


# For backwards compatibility
KafkaService = KafkaServiceStub





