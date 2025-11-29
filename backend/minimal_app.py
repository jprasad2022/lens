#!/usr/bin/env python3
"""
Minimal FastAPI app for debugging Cloud Run deployment
This will help us verify the container can start at all
"""
import os
import sys
import logging

# Setup logging to stdout (important for Cloud Run)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

logger.info("=== MINIMAL APP STARTING ===")
logger.info(f"Python: {sys.version}")
logger.info(f"PORT: {os.environ.get('PORT', 'NOT SET')}")

try:
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="LENS Minimal Test")
    
    @app.get("/")
    async def root():
        return {"status": "minimal app running", "port": os.environ.get('PORT', 8000)}
    
    @app.get("/healthz")
    async def health():
        return {"healthy": True, "service": "lens-api-minimal"}
    
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting minimal server on 0.0.0.0:{port}")
        
        # Use simple server config
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            # Important: no workers, no reload for Cloud Run
            workers=1,
            reload=False,
            access_log=True
        )
        
except Exception as e:
    logger.error(f"Failed to start: {e}", exc_info=True)
    sys.exit(1)