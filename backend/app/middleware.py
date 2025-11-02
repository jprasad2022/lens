"""
Custom Middleware
Following the pattern from insurance-rag-app
"""

import time
import json
from typing import Callable
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge

# Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        # Track active requests
        active_requests.inc()

        # Start timing
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Update metrics
        endpoint = f"{request.method} {request.url.path}"
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)

        # Decrement active requests
        active_requests.dec()

        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/healthz", "/metrics", "/"]:
            return await call_next(request)

        # Get client identifier (IP address or user ID)
        client_id = request.client.host if request.client else "unknown"

        # Clean old requests
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute"
                }),
                status_code=429,
                headers={"Retry-After": "60"},
                media_type="application/json"
            )

        # Record request
        self.requests[client_id].append(now)

        # Process request
        response = await call_next(request)

        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(time.time()))

        # Log request
        start_time = time.time()

        # Add request ID to request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}"

        # Log in development
        from config.settings import get_settings
        if get_settings().debug:
            print(f"[{request_id}] {request.method} {request.url.path} "
                  f"-> {response.status_code} ({duration:.3f}s)")

        return response

class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware for cache control headers"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add cache headers based on endpoint
        if request.url.path.startswith("/recommend"):
            # Cache recommendations for 5 minutes
            response.headers["Cache-Control"] = "public, max-age=300"
        elif request.url.path.startswith("/popular"):
            # Cache popular items for 1 hour
            response.headers["Cache-Control"] = "public, max-age=3600"
        elif request.url.path in ["/healthz", "/metrics"]:
            # Don't cache monitoring endpoints
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

        return response
