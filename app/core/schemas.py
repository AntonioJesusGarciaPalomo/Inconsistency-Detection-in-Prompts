"""Pydantic schemas for the API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AnalyzePromptRequest(BaseModel):
    """Request schema for prompt analysis."""
    prompt: str = Field(..., description="The text prompt to analyze for inconsistencies")
    visualization: bool = Field(False, description="Whether to generate a visualization")

# For backwards compatibility
PromptAnalysisRequest = AnalyzePromptRequest

class CycleDescription(BaseModel):
    """Description of an inconsistency cycle."""
    cycle: List[int] = Field(..., description="Indices of claims in the cycle")
    description: str = Field(..., description="Human-readable description of the cycle")

class AnalyzePromptResponse(BaseModel):
    """Response schema for prompt analysis."""
    consistency_score: float = Field(..., description="Overall consistency score from 0 to 10, 0 means inconsistent")
    claims: List[str] = Field(..., description="List of extracted claims from the prompt")
    cycles: List[List[int]] = Field(..., description="List of detected inconsistency cycles")
    inconsistent_pairs: List[CycleDescription] = Field(..., description="Descriptions of inconsistent cycles")
    visualization_url: Optional[str] = Field(None, description="URL to the generated visualization")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

# For backwards compatibility
PromptAnalysisResponse = AnalyzePromptResponse

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")