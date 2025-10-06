"""
Stream consumer starter (stub).
"""

from services.kafka_service import KafkaService


async def start_consumer(kafka: KafkaService):
    # No-op for local dev without Kafka
    return None




