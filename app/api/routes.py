"""API routes for the inconsistency detection service."""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import glob
from typing import List, Dict, Any, Optional

from app.core.schemas import AnalyzePromptRequest, AnalyzePromptResponse, ErrorResponse
from app.services.detector import InconsistencyDetector
from app.services.openai_service import OpenAIService

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

# Initialize services
openai_service = OpenAIService()
detector = InconsistencyDetector(openai_service)

@router.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "ok"}

@router.post("/analyze", response_model=AnalyzePromptResponse)
async def analyze_prompt(
    request: AnalyzePromptRequest,
    background_tasks: BackgroundTasks
):
    """Analyze a prompt for inconsistencies."""
    try:
        result = await detector.analyze_prompt(
            prompt=request.prompt,
            generate_visualization=request.visualization
        )
        
        # Add cleanup of old visualizations as a background task
        background_tasks.add_task(cleanup_old_visualizations)
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing prompt: {str(e)}")

@router.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    """Get a visualization file."""
    visualization_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
    filepath = os.path.join(visualization_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(filepath)

def cleanup_old_visualizations():
    """Delete old visualization files, keeping only the 20 most recent."""
    try:
        visualization_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
        files = glob.glob(os.path.join(visualization_dir, "*.png"))
        
        # Sort files by creation time (newest first)
        files.sort(key=os.path.getctime, reverse=True)
        
        # Keep only the 20 most recent files
        for old_file in files[20:]:
            os.remove(old_file)
            logger.debug(f"Deleted old visualization: {old_file}")
    except Exception as e:
        logger.error(f"Error cleaning up visualizations: {str(e)}")