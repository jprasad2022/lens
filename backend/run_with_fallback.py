#!/usr/bin/env python3
"""
Production runner with fallback to minimal app if main app fails
"""
import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Log environment information
logger.info("=== STARTUP ENVIRONMENT ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PORT: {os.environ.get('PORT', 'Not set')}")
logger.info(f"K_SERVICE: {os.environ.get('K_SERVICE', 'Not set')}")

# Try to import the main app
app = None
app_type = None

try:
    logger.info("Attempting to import main application...")
    from app.main import app
    app_type = "main"
    logger.info("✅ Main application loaded successfully")
except Exception as e:
    logger.error(f"Failed to load main app: {e}")
    logger.info("Falling back to minimal app...")
    
    try:
        from fastapi import FastAPI
        app = FastAPI(title="LENS API - Fallback Mode")
        
        @app.get("/")
        async def root():
            return {
                "status": "running", 
                "mode": "fallback",
                "error": "Main app failed to load",
                "port": os.environ.get("PORT", 8000)
            }
        
        @app.get("/healthz")
        async def health():
            return {"healthy": True, "mode": "fallback"}
        
        @app.get("/debug/error")
        async def debug_error():
            return {"error": str(e), "type": type(e).__name__}
        
        app_type = "fallback"
        logger.info("✅ Fallback app created")
        
    except Exception as fallback_error:
        logger.error(f"Even fallback failed: {fallback_error}")
        sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting {app_type} server on port {port}")
    
    try:
        # Use string import for main app, direct object for fallback
        if app_type == "main":
            uvicorn.run(
                "app.main:app",
                host="0.0.0.0",
                port=port,
                log_level="info",
                access_log=True,
                reload=False
            )
        else:
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                access_log=True,
                reload=False
            )
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        sys.exit(1)