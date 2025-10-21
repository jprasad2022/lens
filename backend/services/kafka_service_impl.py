"""
Real Kafka Service Implementation for Confluent Cloud
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.serialization import StringSerializer, StringDeserializer
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer

logger = logging.getLogger(__name__)

class KafkaServiceImpl:
    """Kafka service implementation for Confluent Cloud."""

    def __init__(self):
        # Get team prefix from environment
        self.team_prefix = os.getenv("TEAM_PREFIX", "team1")
        
        # Kafka configuration
        self.kafka_config = {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
        }
        
        # Add authentication if using Confluent Cloud
        if self.kafka_config["security.protocol"] == "SASL_SSL":
            self.kafka_config.update({
                "sasl.mechanisms": "PLAIN",
                "sasl.username": os.getenv("KAFKA_API_KEY"),
                "sasl.password": os.getenv("KAFKA_API_SECRET"),
            })
        
        # Schema Registry configuration
        self.schema_registry_url = os.getenv("SCHEMA_REGISTRY_URL", "")
        self.schema_registry_config = {}
        if self.schema_registry_url:
            self.schema_registry_config = {
                "url": self.schema_registry_url,
                "basic.auth.user.info": f"{os.getenv('SCHEMA_REGISTRY_API_KEY')}:{os.getenv('SCHEMA_REGISTRY_API_SECRET')}"
            }
        
        # Topic names
        self.topics = {
            "watch": f"{self.team_prefix}.watch",
            "rate": f"{self.team_prefix}.rate",
            "reco_requests": f"{self.team_prefix}.reco_requests",
            "reco_responses": f"{self.team_prefix}.reco_responses",
        }
        
        self.producer = None
        self.admin_client = None
        self.schema_registry_client = None

    async def initialize(self) -> None:
        """Initialize Kafka connections and create topics if needed."""
        try:
            # Create producer
            self.producer = Producer(self.kafka_config)
            
            # Create admin client
            self.admin_client = AdminClient(self.kafka_config)
            
            # Create schema registry client if configured
            if self.schema_registry_config:
                self.schema_registry_client = SchemaRegistryClient(self.schema_registry_config)
            
            # Create topics if they don't exist
            await self._create_topics_if_needed()
            
            logger.info(f"Kafka service initialized with topics: {list(self.topics.values())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka service: {e}")
            raise

    async def _create_topics_if_needed(self):
        """Create topics if they don't exist."""
        existing_topics = self.admin_client.list_topics(timeout=10).topics
        
        topics_to_create = []
        for topic_name in self.topics.values():
            if topic_name not in existing_topics:
                # Configure topic with 3 partitions and replication factor of 3
                topics_to_create.append(NewTopic(
                    topic_name,
                    num_partitions=3,
                    replication_factor=3,
                    config={
                        "retention.ms": "604800000",  # 7 days
                        "compression.type": "gzip",
                    }
                ))
        
        if topics_to_create:
            fs = self.admin_client.create_topics(topics_to_create)
            for topic, f in fs.items():
                try:
                    f.result()
                    logger.info(f"Topic {topic} created")
                except Exception as e:
                    logger.error(f"Failed to create topic {topic}: {e}")

    def _delivery_report(self, err, msg):
        """Callback for message delivery reports."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

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
        """Produce recommendation response event."""
        try:
            event = {
                "ts": int(datetime.utcnow().timestamp() * 1000),
                "user_id": user_id,
                "request_id": request_id,
                "status": status,
                "latency_ms": latency_ms,
                "k": len(movie_ids),
                "movie_ids": movie_ids,
                "model_version": model_version,
                "cached": cached,
            }
            
            self.producer.produce(
                self.topics["reco_responses"],
                key=str(user_id),
                value=json.dumps(event),
                callback=self._delivery_report
            )
            
            # Flush to ensure delivery
            self.producer.flush(timeout=1)
            
        except Exception as e:
            logger.error(f"Failed to produce reco_response: {e}")

    async def produce_reco_request(
        self,
        user_id: int,
        request_id: str,
        model: str,
        k: int,
    ) -> None:
        """Produce recommendation request event."""
        try:
            event = {
                "ts": int(datetime.utcnow().timestamp() * 1000),
                "user_id": user_id,
                "request_id": request_id,
                "model": model,
                "k": k,
            }
            
            self.producer.produce(
                self.topics["reco_requests"],
                key=str(user_id),
                value=json.dumps(event),
                callback=self._delivery_report
            )
            
        except Exception as e:
            logger.error(f"Failed to produce reco_request: {e}")

    async def produce_watch_event(self, user_id: int, movie_id: int, progress: float) -> None:
        """Produce watch event."""
        try:
            event = {
                "ts": int(datetime.utcnow().timestamp() * 1000),
                "user_id": user_id,
                "movie_id": movie_id,
                "progress": progress,
            }
            
            self.producer.produce(
                self.topics["watch"],
                key=f"{user_id}_{movie_id}",
                value=json.dumps(event),
                callback=self._delivery_report
            )
            
        except Exception as e:
            logger.error(f"Failed to produce watch event: {e}")

    async def produce_rate_event(self, user_id: int, movie_id: int, rating: float) -> None:
        """Produce rating event."""
        try:
            event = {
                "ts": int(datetime.utcnow().timestamp() * 1000),
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
            }
            
            self.producer.produce(
                self.topics["rate"],
                key=f"{user_id}_{movie_id}",
                value=json.dumps(event),
                callback=self._delivery_report
            )
            
        except Exception as e:
            logger.error(f"Failed to produce rate event: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check Kafka connectivity."""
        try:
            metadata = self.admin_client.list_topics(timeout=5)
            return {
                "healthy": True,
                "connected": True,
                "topics": list(self.topics.values()),
                "cluster_id": metadata.cluster_id,
            }
        except Exception as e:
            return {
                "healthy": False,
                "connected": False,
                "error": str(e)
            }

    async def close(self) -> None:
        """Clean up resources."""
        if self.producer:
            self.producer.flush()