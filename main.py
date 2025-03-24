"""Main application entry point."""
import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import traceback

from app.api.routes import router
from app.core.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inconsistency-detector")

# Load settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Inconsistency Detector API",
    description="API for detecting logical inconsistencies in text prompts",
    version="1.0.0",
)

# Configure CORS
if settings.allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files for visualizations
visualizations_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
os.makedirs(visualizations_dir, exist_ok=True)
app.mount("/visualizations", StaticFiles(directory=visualizations_dir), name="visualizations")

# Include API routes
app.include_router(router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Inconsistency Detector API")
    logger.info(f"API running on {settings.api_host}:{settings.api_port}")
    logger.info(f"Debug mode: {settings.debug}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Inconsistency Detector API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )