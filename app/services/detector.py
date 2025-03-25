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
        try:
            # Usar directamente analyze_text del SheafAnalyzer que ya incluye todo el flujo
            analysis_result = self.sheaf_analyzer.analyze_text(prompt)
            
            # Si no se han extraído afirmaciones, hacerlo con la API de OpenAI
            if not analysis_result['claims']:
                claims = await self.extract_claims(prompt)
                if claims:
                    # Volver a analizar con las afirmaciones extraídas
                    analysis_result = self.sheaf_analyzer.analyze_text("\n".join(claims))
            
            # Formatear resultados
            result = {
                "consistency_score": analysis_result.get('consistency_score', 10.0),
                "claims": analysis_result.get('claims', []),
                "cycles": analysis_result.get('cycles', []),
                "inconsistent_pairs": []
            }
            
            # Crear descripciones significativas de las inconsistencias
            cycles = analysis_result.get('cycles', [])
            claims = analysis_result.get('claims', [])
            
            for cycle in cycles:
                cycle_claims = [claims[i] for i in cycle]
                cycle_description = " → ".join(cycle_claims) + " → " + cycle_claims[0]
                result["inconsistent_pairs"].append({
                    "cycle": cycle,
                    "description": cycle_description
                })
            
            # Agregar también pares inconsistentes directos
            inconsistent_pairs = analysis_result.get('inconsistent_pairs', [])
            for i, j in inconsistent_pairs:
                result["inconsistent_pairs"].append({
                    "pair": [i, j],
                    "description": f"Contradicción directa entre '{claims[i]}' y '{claims[j]}'"
                })
            
            # Añadir la visualización si se solicita
            if generate_visualization:
                result["visualization_url"] = analysis_result.get('visualization_path')
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {str(e)}")
            return {
                "error": f"Error analyzing prompt: {str(e)}",
                "consistency_score": None,
                "claims": [],
                "cycles": [],
                "inconsistent_pairs": []
            }