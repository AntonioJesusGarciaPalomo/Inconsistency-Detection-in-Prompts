"""
Service for detecting inconsistencies in prompts using sheaf theory.
"""
import logging
import re
from typing import List, Dict, Any, Tuple
import os
from uuid import uuid4
import asyncio

from app.services.openai_service import OpenAIService
from app.utils.sheaf import SheafAnalyzer

logger = logging.getLogger(__name__)

class InconsistencyDetector:
    """Detector class for finding inconsistencies in text prompts."""
    
    def __init__(self, openai_service: OpenAIService):
        """Initialize the detector with an OpenAI service for LLM analysis."""
        self.openai_service = openai_service
        self.sheaf_analyzer = SheafAnalyzer()
        
    async def extract_claims(self, prompt: str) -> List[str]:
        """Extract atomic claims from a prompt using LLM."""
        system_message = """
        Extract all atomic claims from the provided text. An atomic claim is a single, 
        self-contained factual statement.
        
        Extract these as a list, one claim per line, prefixed with "- ".
        Focus on statements that could potentially contradict each other.
        """
        
        try:
            response = await self.openai_service.get_completion(
                system_message=system_message,
                user_message=prompt
            )
            
            # Extract claims, removing bullet points
            claims = [claim.strip()[2:].strip() if claim.strip().startswith("- ") else claim.strip() 
                     for claim in response.split("\n") if claim.strip()]
            
            logger.info(f"Extracted {len(claims)} claims from prompt")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            # Fall back to a simple regex-based extraction
            sentences = re.split(r'[.!?]\s+', prompt)
            return [s.strip() + "." for s in sentences if s.strip()]
    
    async def evaluate_consistency(self, claims: List[str]) -> float:
        """Evaluate the consistency between pairs of claims."""
        if len(claims) <= 1:
            return 10.0  # A single claim is always consistent
        
        # Detect circular inconsistencies
        cycles, _ = self.sheaf_analyzer.detect_circular_inconsistencies(claims)
        
        # Compute global consistency score
        consistency_score = self.sheaf_analyzer.compute_global_consistency(claims, cycles)
        
        return consistency_score
    
    async def generate_visualization(self, claims: List[str], cycles: List = None) -> str:
        """Generate a visualization of the claims and their relationships."""
        try:
            vis_path = self.sheaf_analyzer.visualize_inconsistency_network(claims, cycles)
            return vis_path
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return None
    
    async def analyze_prompt(self, prompt: str, generate_visualization: bool = False) -> Dict[str, Any]:
        """Analyze a prompt for inconsistencies and return the results."""
        # Extract claims from the prompt
        claims = await self.extract_claims(prompt)
        
        if not claims:
            return {
                "error": "Could not extract any claims from the prompt",
                "consistency_score": None,
                "claims": [],
                "inconsistent_pairs": []
            }
        
        # Detect circular inconsistencies
        cycles, _ = self.sheaf_analyzer.detect_circular_inconsistencies(claims)
        
        # Compute global consistency score
        consistency_score = self.sheaf_analyzer.compute_global_consistency(claims, cycles)
        
        # Create meaningful descriptions of inconsistencies
        inconsistent_pairs = []
        for cycle in cycles:
            cycle_claims = [claims[i] for i in cycle]
            cycle_description = " → ".join(cycle_claims) + " → " + cycle_claims[0]
            inconsistent_pairs.append({
                "cycle": cycle,
                "description": cycle_description
            })
        
        result = {
            "consistency_score": consistency_score,
            "claims": claims,
            "cycles": cycles,
            "inconsistent_pairs": inconsistent_pairs
        }
        
        # Generate visualization if requested
        if generate_visualization:
            vis_path = await self.generate_visualization(claims, cycles)
            result["visualization_url"] = vis_path
        
        return result