#!/usr/bin/env python3
"""
Simplified runner for debugging Cloud Run issues
"""
import os
import sys
import logging

# Configure logging to stdout for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

logger.info("=== SIMPLE STARTUP TEST ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}")

# Test basic imports
try:
    logger.info("Testing basic imports...")
    import uvicorn
    logger.info("✓ uvicorn imported")
    import fastapi
    logger.info("✓ fastapi imported")
except ImportError as e:
    logger.error(f"Failed to import basic dependencies: {e}")
    sys.exit(1)

# Try to start a minimal FastAPI app
try:
    from fastapi import FastAPI
    
    # Create minimal app
    app = FastAPI()
    
    @app.get("/healthz")
    async def health():
        return {"status": "ok", "service": "lens-api-debug"}
    
    @app.get("/")
    async def root():
        return {"message": "Minimal app is running"}
    
    # Start server
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting minimal server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
    
except Exception as e:
    logger.error(f"Failed to start minimal server: {e}", exc_info=True)
    sys.exit(1)