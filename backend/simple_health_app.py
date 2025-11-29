"""
Simple health check app that always works
Used to verify Cloud Run deployment
"""
from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "lens-api",
        "port": os.environ.get("PORT", "unknown"),
        "message": "Simple health app is running"
    }

@app.get("/healthz")
async def health():
    return {"healthy": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)