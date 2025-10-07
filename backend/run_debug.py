#!/usr/bin/env python3
"""
Debug runner script with extra logging for Cloud Run troubleshooting
"""
import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log environment information
logger.info("=== STARTUP ENVIRONMENT ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")
logger.info(f"K_SERVICE (Cloud Run): {os.environ.get('K_SERVICE', 'Not set')}")
logger.info(f"K_REVISION (Cloud Run): {os.environ.get('K_REVISION', 'Not set')}")

# Check for critical directories
critical_paths = [
    "app",
    "config", 
    "services",
    "routers",
    "data",
    "model_registry"
]

logger.info("=== DIRECTORY CHECK ===")
for path in critical_paths:
    exists = os.path.exists(path)
    logger.info(f"{path}: {'EXISTS' if exists else 'MISSING'}")

# Import app after environment logging
try:
    logger.info("=== IMPORTING APPLICATION ===")
    from app.main import app
    logger.info("Application imported successfully")
except Exception as e:
    logger.error(f"Failed to import application: {e}", exc_info=True)
    sys.exit(1)

if __name__ == "__main__":
    # Cloud Run sets the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        sys.exit(1)