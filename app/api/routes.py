"""API routes for the inconsistency detection service."""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import glob
from typing import List, Dict, Any, Optional, Literal

from app.core.schemas import AnalyzePromptRequest, AnalyzePromptResponse, ErrorResponse
from app.services.detector import InconsistencyDetector
from app.services.openai_service import OpenAIService
from app.core.config import settings

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

# Initialize services
openai_service = OpenAIService()
# Use the configured API_HOST or server URL from environment variables
base_url = f"http://{settings.api_host}:{settings.api_port}" if settings.api_host else "http://localhost:8000"
detector = InconsistencyDetector(openai_service, base_url=base_url)

@router.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "ok"}

@router.post("/analyze", response_model=AnalyzePromptResponse)
async def analyze_prompt(
    request: AnalyzePromptRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze a prompt for inconsistencies.
    
    Args:
        request: The analysis request with prompt, visualization flag, and visualization type
        background_tasks: FastAPI background tasks
    
    Returns:
        Analysis results
    """
    try:
        # La clave aquí es pasar el visualization_type desde la solicitud
        result = await detector.analyze_prompt(
            prompt=request.prompt,
            generate_visualization=request.visualization,
            visualization_type=request.visualization_type  # Aseguramos que se pasa este parámetro
        )
        
        # Añadimos el tipo de visualización a la respuesta
        if request.visualization and "visualization_url" in result and result["visualization_url"]:
            result["visualization_type"] = request.visualization_type
        
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
        
        # Clean up PNG files
        png_files = glob.glob(os.path.join(visualization_dir, "*.png"))
        png_files.sort(key=os.path.getctime, reverse=True)
        for old_file in png_files[20:]:
            os.remove(old_file)
            logger.debug(f"Deleted old PNG visualization: {old_file}")
            
        # Clean up HTML files
        html_files = glob.glob(os.path.join(visualization_dir, "*.html"))
        html_files.sort(key=os.path.getctime, reverse=True)
        for old_file in html_files[20:]:
            os.remove(old_file)
            logger.debug(f"Deleted old HTML visualization: {old_file}")
    except Exception as e:
        logger.error(f"Error cleaning up visualizations: {str(e)}")