"""
Stream Ingestor with Schema Validation and Object Storage
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from confluent_kafka import Consumer, KafkaError
from jsonschema import validate, ValidationError
import boto3
import redis
from pathlib import Path

logger = logging.getLogger(__name__)

# Event schemas for validation
SCHEMAS = {
    "watch": {
        "type": "object",
        "properties": {
            "ts": {"type": "integer"},
            "user_id": {"type": "integer"},
            "movie_id": {"type": "integer"},
            "progress": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["ts", "user_id", "movie_id", "progress"]
    },
    "rate": {
        "type": "object",
        "properties": {
            "ts": {"type": "integer"},
            "user_id": {"type": "integer"},
            "movie_id": {"type": "integer"},
            "rating": {"type": "number", "minimum": 0.5, "maximum": 5},
        },
        "required": ["ts", "user_id", "movie_id", "rating"]
    },
    "reco_requests": {
        "type": "object",
        "properties": {
            "ts": {"type": "integer"},
            "user_id": {"type": "integer"},
            "request_id": {"type": "string"},
            "model": {"type": "string"},
            "k": {"type": "integer", "minimum": 1},
        },
        "required": ["ts", "user_id", "request_id", "model", "k"]
    },
    "reco_responses": {
        "type": "object",
        "properties": {
            "ts": {"type": "integer"},
            "user_id": {"type": "integer"},
            "request_id": {"type": "string"},
            "status": {"type": "integer"},
            "latency_ms": {"type": "number"},
            "k": {"type": "integer"},
            "movie_ids": {"type": "array", "items": {"type": "integer"}},
            "model_version": {"type": "string"},
            "cached": {"type": "boolean"},
        },
        "required": ["ts", "user_id", "request_id", "status", "latency_ms", "k", "movie_ids", "model_version", "cached"]
    }
}

class StreamIngestor:
    """Ingest Kafka streams with validation and storage."""
    
    def __init__(self, team_prefix: str = "team1"):
        self.team_prefix = team_prefix
        
        # Kafka configuration
        self.kafka_config = {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
            "group.id": f"{team_prefix}-ingestor",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
        }
        
        # Add authentication if needed
        if self.kafka_config["security.protocol"] == "SASL_SSL":
            self.kafka_config.update({
                "sasl.mechanisms": "PLAIN",
                "sasl.username": os.getenv("KAFKA_API_KEY"),
                "sasl.password": os.getenv("KAFKA_API_SECRET"),
            })
        
        # Storage configuration
        self.storage_type = os.getenv("STORAGE_TYPE", "local")  # local, s3, gcs
        self.storage_bucket = os.getenv("STORAGE_BUCKET", "movielens-data")
        self.storage_prefix = os.getenv("STORAGE_PREFIX", f"{team_prefix}/events")
        
        # Redis configuration (optional caching)
        self.redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        if self.redis_enabled:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
        
        # Batch configuration
        self.batch_size = int(os.getenv("INGESTOR_BATCH_SIZE", 1000))
        self.batch_timeout = int(os.getenv("INGESTOR_BATCH_TIMEOUT", 60))  # seconds
        
        # Event buffers
        self.buffers = {
            "watch": [],
            "rate": [],
            "reco_requests": [],
            "reco_responses": [],
        }
        
        self.last_flush = datetime.utcnow()
        self.stats = {
            "valid": 0,
            "invalid": 0,
            "flushed": 0,
        }

    def validate_event(self, event: Dict[str, Any], event_type: str) -> bool:
        """Validate event against schema."""
        try:
            validate(event, SCHEMAS[event_type])
            return True
        except ValidationError as e:
            logger.warning(f"Invalid {event_type} event: {e}")
            self.stats["invalid"] += 1
            return False

    async def process_message(self, message):
        """Process a single Kafka message."""
        try:
            # Parse message
            event = json.loads(message.value().decode('utf-8'))
            topic = message.topic()
            
            # Determine event type from topic
            event_type = topic.split('.')[-1]  # e.g., "team1.watch" -> "watch"
            
            # Validate event
            if not self.validate_event(event, event_type):
                return
            
            # Add metadata
            event["_ingested_at"] = datetime.utcnow().isoformat()
            event["_partition"] = message.partition()
            event["_offset"] = message.offset()
            
            # Add to buffer
            self.buffers[event_type].append(event)
            self.stats["valid"] += 1
            
            # Cache in Redis if enabled
            if self.redis_enabled and event_type in ["watch", "rate"]:
                await self._cache_event(event, event_type)
            
            # Check if we should flush
            if len(self.buffers[event_type]) >= self.batch_size:
                await self.flush_buffer(event_type)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            self.stats["invalid"] += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _cache_event(self, event: Dict[str, Any], event_type: str):
        """Cache event in Redis for quick access."""
        try:
            if event_type == "watch":
                key = f"watch:{event['user_id']}:{event['movie_id']}"
                self.redis_client.setex(key, 3600, json.dumps(event))  # 1 hour TTL
            elif event_type == "rate":
                key = f"rating:{event['user_id']}:{event['movie_id']}"
                self.redis_client.setex(key, 86400, json.dumps(event))  # 24 hour TTL
        except Exception as e:
            logger.warning(f"Failed to cache event: {e}")

    async def flush_buffer(self, event_type: str):
        """Flush buffer to storage."""
        if not self.buffers[event_type]:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.buffers[event_type])
            
            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{event_type}_{timestamp}_{len(df)}_events.parquet"
            
            # Save based on storage type
            if self.storage_type == "local":
                await self._save_local(df, event_type, filename)
            elif self.storage_type == "s3":
                await self._save_s3(df, event_type, filename)
            elif self.storage_type == "gcs":
                await self._save_gcs(df, event_type, filename)
            
            logger.info(f"Flushed {len(df)} {event_type} events to {filename}")
            self.stats["flushed"] += len(df)
            
            # Clear buffer
            self.buffers[event_type] = []
            
        except Exception as e:
            logger.error(f"Failed to flush {event_type} buffer: {e}")

    async def _save_local(self, df: pd.DataFrame, event_type: str, filename: str):
        """Save to local filesystem."""
        base_path = Path(f"data/snapshots/{self.storage_prefix}/{event_type}")
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        filepath = base_path / filename
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        
        # Also save as CSV for easy inspection
        csv_filename = filename.replace('.parquet', '.csv')
        df.to_csv(base_path / csv_filename, index=False)

    async def _save_s3(self, df: pd.DataFrame, event_type: str, filename: str):
        """Save to S3."""
        s3_client = boto3.client('s3')
        key = f"{self.storage_prefix}/{event_type}/{filename}"
        
        # Convert DataFrame to Parquet bytes
        table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf, compression='snappy')
        
        # Upload to S3
        s3_client.put_object(
            Bucket=self.storage_bucket,
            Key=key,
            Body=buf.getvalue().to_pybytes()
        )

    async def _save_gcs(self, df: pd.DataFrame, event_type: str, filename: str):
        """Save to Google Cloud Storage."""
        # Implementation for GCS
        pass

    async def periodic_flush(self):
        """Periodically flush buffers."""
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            # Check if we need to flush any buffers
            for event_type in self.buffers:
                if self.buffers[event_type]:
                    logger.info(f"Periodic flush for {event_type}")
                    await self.flush_buffer(event_type)

    async def run(self):
        """Run the ingestor."""
        # Create consumer
        consumer = Consumer(self.kafka_config)
        
        # Subscribe to topics
        topics = [
            f"{self.team_prefix}.watch",
            f"{self.team_prefix}.rate",
            f"{self.team_prefix}.reco_requests",
            f"{self.team_prefix}.reco_responses",
        ]
        
        consumer.subscribe(topics)
        logger.info(f"Subscribed to topics: {topics}")
        
        # Start periodic flush task
        flush_task = asyncio.create_task(self.periodic_flush())
        
        try:
            while True:
                # Poll for messages
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break
                
                # Process message
                await self.process_message(msg)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Flush remaining buffers
            for event_type in self.buffers:
                if self.buffers[event_type]:
                    await self.flush_buffer(event_type)
            
            # Clean up
            flush_task.cancel()
            consumer.close()
            
            # Print statistics
            logger.info(f"Ingestor statistics:")
            logger.info(f"  Valid events: {self.stats['valid']}")
            logger.info(f"  Invalid events: {self.stats['invalid']}")
            logger.info(f"  Flushed events: {self.stats['flushed']}")

async def main():
    """Run the stream ingestor."""
    ingestor = StreamIngestor()
    await ingestor.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())