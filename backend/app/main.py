"""
Main FastAPI Application
Following the pattern from insurance-rag-app
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from config.settings import get_settings
from app.state import AppState
from routers import recommendation_router, model_router, monitoring_router, interaction_router, ab_testing_router, provenance_router
from services.model_service import ModelService
from services.kafka_service import get_kafka_service
from services.movie_metadata_service import MovieMetadataService
from services.rating_statistics_service import RatingStatisticsService
from services.user_demographics_service import UserDemographicsService
from stream.consumer import start_consumer
from app.middleware import (
    PrometheusMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware
)

# Get settings
settings = get_settings()

# Application state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    startup_start = time.time()

    # Initialize services
    print("Starting MovieLens Recommender API...")

    # Download data from GCS if needed
    try:
        from utils.download_data import download_movielens_data
        download_movielens_data()
    except Exception as e:
        print(f"Warning: Could not download data: {e}")

    # Download models from GCS if needed
    try:
        from utils.download_models import download_model_registry
        download_model_registry()
    except Exception as e:
        print(f"Warning: Could not download models: {e}")

    try:
        # Initialize model service
        print("Initializing model service...")
        app_state.model_service = ModelService()
        await app_state.model_service.initialize()

        # Initialize movie metadata service
        print("Initializing movie metadata service...")
        app_state.movie_metadata_service = MovieMetadataService()
        await app_state.movie_metadata_service.initialize()

        # Initialize rating statistics service
        print("Initializing rating statistics service...")
        app_state.rating_statistics_service = RatingStatisticsService()
        await app_state.rating_statistics_service.initialize()

        # Initialize user demographics service
        print("Initializing user demographics service...")
        app_state.user_demographics_service = UserDemographicsService()
        await app_state.user_demographics_service.initialize()

        # Initialize AB switch service
        print("Initializing A/B switch service...")
        from services.ab_switch_service import get_ab_switch_service
        app_state.ab_switch_service = get_ab_switch_service()

        # Initialize Kafka if enabled
        if settings.kafka_bootstrap_servers != "localhost:9092":
            print("Initializing Kafka service...")
            app_state.kafka_service = get_kafka_service()
            await app_state.kafka_service.initialize()

            # Start Kafka consumer in background
            app_state.consumer_task = asyncio.create_task(
                start_consumer(app_state.kafka_service)
            )

        # Load default model
        print(f"Loading default model: {settings.default_model}")
        await app_state.model_service.load_model(settings.default_model)

        startup_time = time.time() - startup_start
        print(f"Application started in {startup_time:.2f}s")

    except Exception as e:
        print(f"Startup failed: {str(e)}")
        raise

    yield

    # Shutdown
    print("Shutting down...")

    # Cancel consumer task
    if hasattr(app_state, 'consumer_task') and app_state.consumer_task:
        app_state.consumer_task.cancel()
        try:
            await app_state.consumer_task
        except asyncio.CancelledError:
            pass

    # Cleanup services
    if app_state.kafka_service:
        await app_state.kafka_service.close()

    if app_state.model_service:
        await app_state.model_service.cleanup()

    print("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",  # Always enable docs for debugging
    redoc_url="/redoc",  # Always enable redoc for debugging
)

# Add custom middleware FIRST (executed last)
if settings.enable_metrics:
    app.add_middleware(PrometheusMiddleware)
if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware LAST (executed first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(
    recommendation_router.router,
    prefix=f"{settings.api_prefix}",
    tags=["recommendations"]
)
app.include_router(
    model_router.router,
    prefix=f"{settings.api_prefix}",
    tags=["models"]
)
app.include_router(
    monitoring_router.router,
    prefix=f"{settings.api_prefix}",
    tags=["monitoring"]
)
app.include_router(
    interaction_router.router,
    prefix=f"{settings.api_prefix}",
    tags=["interactions"]
)
app.include_router(
    ab_testing_router.router,
    prefix=f"{settings.api_prefix}/ab",
    tags=["ab_testing"]
)
app.include_router(
    provenance_router.router,
    prefix=f"{settings.api_prefix}/provenance",
    tags=["provenance"]
)

# Mount metrics endpoint
if settings.enable_metrics:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "running",
        "docs": "/docs" if settings.debug else None,
    }


# Health check
@app.get("/healthz", tags=["monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    health_status = await app_state.get_health_status()

    if not health_status["healthy"]:
        return JSONResponse(
            status_code=503,
            content=health_status
        )

    return health_status


# Debug endpoint
@app.get("/debug/cors", include_in_schema=False)
async def debug_cors() -> Dict[str, Any]:
    """Debug CORS settings"""
    return {
        "allowed_origins": settings.allowed_origins,
        "allowed_origins_type": str(type(settings.allowed_origins)),
        "allowed_origins_items": [str(origin) for origin in settings.allowed_origins] if settings.allowed_origins else []
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    import traceback

    # Log the full traceback in development
    if settings.debug:
        traceback.print_exc()

    # Log all errors with method and path
    print(f"ERROR: {request.method} {request.url.path} - {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.debug else "An error occurred",
        }
    )


# Startup message
@app.on_event("startup")
async def startup_message():
    """Print startup configuration"""
    print(f"""
    ============================================
    🎬 MovieLens Recommender API
    ============================================
    Version: {settings.version}
    Environment: {'Cloud' if settings.is_cloud_environment else 'Local'}
    Debug: {settings.debug}
    Auth: {'Enabled' if settings.enable_auth else 'Disabled'}
    Kafka: {'Connected' if settings.kafka_bootstrap_servers != "localhost:9092" else 'Disabled'}
    Redis: {'Enabled' if settings.redis_enabled else 'Disabled'}
    Models: {settings.model_registry_path}
    ============================================
    """)
