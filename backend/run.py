#!/usr/bin/env python3
"""
Explicit runner script for the application
"""
import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    # Cloud Run sets the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )