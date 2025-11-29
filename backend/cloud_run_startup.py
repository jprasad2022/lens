"""
Cloud Run optimized startup script
Handles missing data gracefully and provides better error messages
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        Path("/app/data"),
        Path("/app/model_registry"),
        Path("/app/logs")
    ]
    
    for dir_path in dirs:
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory exists: {dir_path}")

def create_dummy_data():
    """Create minimal dummy data files for startup"""
    data_dir = Path("/app/data")
    
    # Create empty marker files if real data doesn't exist
    files = ["movies.dat", "ratings.dat", "users.dat"]
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning(f"Creating dummy {filename} for startup")
            filepath.write_text("# Dummy file for Cloud Run startup\n")

def main():
    """Main startup function"""
    logger.info("=== Cloud Run Startup Script ===")
    logger.info(f"PORT: {os.environ.get('PORT', 'Not set')}")
    logger.info(f"K_SERVICE: {os.environ.get('K_SERVICE', 'Not set')}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Create dummy data if needed
    if os.environ.get('K_SERVICE'):  # Running in Cloud Run
        create_dummy_data()
    
    # Set environment variable to skip downloads during startup
    os.environ['SKIP_STARTUP_DOWNLOADS'] = 'true'
    
    # Import and run the main application
    logger.info("Starting FastAPI application...")
    try:
        import uvicorn
        from app.main import app
        
        port = int(os.environ.get("PORT", 8000))
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=30,
            limit_concurrency=1000,
            limit_max_requests=10000
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()