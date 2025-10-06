"""
Kafka service stub for local development when Kafka is not configured.
"""

from typing import Any, Dict, List


class KafkaService:
    """No-op Kafka service used unless real Kafka config is provided."""

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




