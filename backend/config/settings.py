"""
Application Settings
Following the pattern from insurance-rag-app
"""

from typing import Optional, List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Configuration
    app_name: str = "MovieLens Recommender API"
    version: str = "0.1.0"
    debug: bool = Field(default=False)
    api_prefix: str = ""
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:8000", "https://lens-smoky-six.vercel.app"]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"[CORS DEBUG] Loaded allowed_origins: {self.allowed_origins}")
        print(f"[CORS DEBUG] Type: {type(self.allowed_origins)}")
        for i, origin in enumerate(self.allowed_origins):
            print(f"[CORS DEBUG] Origin {i}: '{origin}' (length: {len(origin)})")

    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            # If it's a string (from env var), parse it as JSON
            import json
            try:
                return json.loads(v)
            except Exception:
                return v.split(",")  # Fallback to comma-separated
        return v

    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    reload: bool = Field(default=False)

    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_security_protocol: str = Field(default="PLAINTEXT")
    kafka_sasl_mechanism: Optional[str] = Field(default=None)
    kafka_api_key: Optional[str] = Field(default=None)
    kafka_api_secret: Optional[str] = Field(default=None)
    kafka_group_id: str = Field(default="movielens-recommender")
    kafka_auto_offset_reset: str = Field(default="earliest")

    # Kafka Topics
    kafka_watch_topic: str = Field(default="movielens.watch")
    kafka_rate_topic: str = Field(default="movielens.rate")
    kafka_reco_requests_topic: str = Field(default="movielens.reco_requests")
    kafka_reco_responses_topic: str = Field(default="movielens.reco_responses")

    # Model Configuration
    model_registry_path: Path = Field(default=Path("model_registry"))
    default_model: str = Field(default="popularity")
    model_cache_size: int = Field(default=3)
    enable_model_versioning: bool = Field(default=True)

    # Data Paths
    data_path: Path = Field(default=Path("data"))
    movielens_path: Path = Field(default=Path("data/ml-1m"))
    snapshot_path: Path = Field(default=Path("data/snapshots"))

    # Redis Configuration (Optional)
    redis_url: Optional[str] = Field(default=None)
    redis_ttl: int = Field(default=3600)  # 1 hour
    enable_redis_cache: bool = Field(default=False)

    # Cloud Storage (Optional)
    gcs_bucket: Optional[str] = Field(default=None)
    aws_s3_bucket: Optional[str] = Field(default=None)
    storage_provider: str = Field(default="local")  # local, gcs, s3

    # Firebase Auth (Optional)
    firebase_credentials_path: Optional[str] = Field(default=None)
    enable_auth: bool = Field(default=False)

    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    enable_tracing: bool = Field(default=False)
    otel_endpoint: Optional[str] = Field(default=None)

    # Model Training
    retrain_schedule: str = Field(default="0 2 * * *")  # Daily at 2 AM
    min_training_samples: int = Field(default=1000)
    test_split_ratio: float = Field(default=0.2)

    # API Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_period: int = Field(default=60)  # seconds

    # Response Configuration
    max_recommendations: int = Field(default=100)
    default_recommendation_count: int = Field(default=20)
    response_cache_ttl: int = Field(default=300)  # 5 minutes

    # A/B Testing
    enable_ab_testing: bool = Field(default=True)
    ab_test_percentage: float = Field(default=0.5)

    # Security
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    api_key_required: bool = Field(default=False)
    valid_api_keys: List[str] = Field(default=[])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    @validator("model_registry_path", "data_path", "movielens_path", "snapshot_path", pre=True)
    def resolve_paths(cls, v):
        """Resolve paths relative to project root"""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            # Get the backend directory
            backend_dir = Path(__file__).parent.parent
            v = backend_dir / v
        return v

    @validator("kafka_bootstrap_servers")
    def validate_kafka_servers(cls, v, values):
        """Add security protocol to Kafka configuration"""
        if values.get("kafka_security_protocol") == "SASL_SSL":
            if not values.get("kafka_api_key") or not values.get("kafka_api_secret"):
                raise ValueError("Kafka API key and secret required for SASL_SSL")
        return v

    @property
    def kafka_config(self) -> Dict[str, any]:
        """Get Kafka consumer/producer configuration"""
        config = {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "group.id": self.kafka_group_id,
            "auto.offset.reset": self.kafka_auto_offset_reset,
            "security.protocol": self.kafka_security_protocol,
        }

        if self.kafka_security_protocol == "SASL_SSL":
            config.update({
                "sasl.mechanisms": self.kafka_sasl_mechanism or "PLAIN",
                "sasl.username": self.kafka_api_key,
                "sasl.password": self.kafka_api_secret,
            })

        return config

    @property
    def is_cloud_environment(self) -> bool:
        """Check if running in cloud environment"""
        return bool(
            os.getenv("K_SERVICE") or  # Cloud Run
            os.getenv("WEBSITE_INSTANCE_ID") or  # Azure
            os.getenv("AWS_EXECUTION_ENV")  # AWS
        )

    @property
    def redis_enabled(self) -> bool:
        """Check if Redis is configured and enabled"""
        return bool(self.redis_url and self.enable_redis_cache)

    def get_storage_client(self):
        """Get appropriate storage client based on configuration"""
        if self.storage_provider == "gcs" and self.gcs_bucket:
            from google.cloud import storage
            return storage.Client()
        elif self.storage_provider == "s3" and self.aws_s3_bucket:
            import boto3
            return boto3.client("s3")
        else:
            return None  # Local storage

# Singleton instance
_settings = None

def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
