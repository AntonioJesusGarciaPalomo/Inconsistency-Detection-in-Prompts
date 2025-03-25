"""
Service for detecting inconsistencies in prompts using language models and graph theory.
"""
import logging
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import uuid
import json
import asyncio
import networkx as nx
import numpy as np
from itertools import combinations

from app.services.openai_service import OpenAIService
from app.utils.graph_analyzer import GraphAnalyzer

logger = logging.getLogger(__name__)

class InconsistencyDetector:
    """Detector class for finding inconsistencies in text prompts using LLM capabilities."""
    
    def __init__(self, openai_service: OpenAIService):
        """Initialize the detector with an OpenAI service for LLM analysis."""
        self.openai_service = openai_service
        self.graph_analyzer = GraphAnalyzer()
        
    async def extract_claims(self, prompt: str) -> List[str]:
        """Extract atomic claims from a prompt using LLM."""
        system_message = """
        Extract all atomic claims from the provided text. An atomic claim is a single, 
        self-contained factual statement.
        
        Extract these as a list, one claim per line, prefixed with "- ".
        Focus on statements that could potentially contradict each other.
        Focus on comparative statements (X is more than Y, X is less than Y, etc.)
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
            # Fall back to simple sentence splitting
            import re
            sentences = re.split(r'[.!?]\s+', prompt)
            return [s.strip() + "." for s in sentences if s.strip()]
    
    async def detect_inconsistencies(self, claims: List[str]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """Detect inconsistencies in a set of claims."""
        if len(claims) <= 1:
            return [], {"consistency_score": 10.0}
        
        # First, try the graph analyzer's detection
        cycles, G = self.graph_analyzer.detect_circular_inconsistencies(claims)
        self.g = G  # Store for later use
        
        if cycles:
            logger.info(f"Graph analyzer detected inconsistency cycles: {cycles}")
            
            # Calculate consistency score
            consistency_score = self.graph_analyzer.compute_global_consistency(claims, cycles)
            
            return cycles, {
                "consistency_score": consistency_score,
                "inconsistency_description": "Detected logically inconsistent relationships"
            }
        
        # If graph analyzer didn't find cycles, try LLM-based analysis
        system_message = """
        Analyze the following claims for logical inconsistencies, especially focusing on:
        1. Direct contradictions
        2. Circular inconsistencies (e.g., A > B > C > A)
        3. Transitive relationship conflicts (e.g., A is older than B, B is older than C, C is older than A)
        
        Return your analysis as a valid JSON object with these fields:
        {
          "inconsistencies_detected": true/false,
          "inconsistency_description": "Brief description of any inconsistencies found",
          "inconsistent_claim_indices": [[0,1,2,3,4,5]], // Arrays of claim indices that form inconsistent cycles
          "consistency_score": 0-10 // Overall consistency score where 0 is completely inconsistent and 10 is completely consistent
        }
        
        IMPORTANT: Return ONLY a valid JSON object without any additional text, markdown formatting, or code block markers.
        
        IMPORTANT FOR TRANSITIVE RELATIONSHIPS: Be sure to check for complete cycles in transitive relationships. 
        For example, if claim 0 says "A > B", claim 1 says "B > C", claim 2 says "C > D", claim 3 says "D > E", 
        claim 4 says "E > F", and claim 5 says "F > A", then this forms a cycle [0,1,2,3,4,5] 
        that should be detected as an inconsistency.
        """
        
        # Prepare the claims for analysis
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        user_message = f"Please analyze these claims for logical inconsistencies:\n\n{claims_text}"
        
        try:
            # Get LLM analysis
            response = await self.openai_service.get_completion(
                system_message=system_message,
                user_message=user_message
            )
            
            # Parse the JSON response
            try:
                # Clean up the response to handle Markdown formatting
                cleaned_response = response
                
                # Remove Markdown JSON code blocks if present
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
                elif "```" in cleaned_response:
                    cleaned_response = cleaned_response.replace("```", "")
                
                # Fix common JSON formatting issues
                # Remove any trailing commas before closing brackets
                cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
                
                # Try to parse the cleaned response
                analysis = json.loads(cleaned_response)
                
                # Add inconsistency information to the graph
                if analysis.get("inconsistencies_detected", False):
                    logger.info(f"LLM detected inconsistencies: {analysis.get('inconsistency_description')}")
                    
                    # Get inconsistent cycles
                    cycles = analysis.get("inconsistent_claim_indices", [])
                    
                    # Adjust to 0-based indexing if needed
                    if cycles and all(all(idx > 0 for idx in cycle) for cycle in cycles):
                        cycles = [[idx-1 for idx in cycle] for cycle in cycles]
                    
                    # Set consistency score
                    consistency_score = analysis.get("consistency_score", 5.0)
                    
                    # Evaluate pairwise consistency for all claims in potential inconsistency cycles
                    pairs_to_evaluate = []
                    for cycle in cycles:
                        for i in range(len(cycle)):
                            pairs_to_evaluate.append((cycle[i], cycle[(i+1) % len(cycle)]))
                    
                    # Add additional claim pairs
                    if len(claims) < 10:  # Only do all pairs for smaller sets
                        for i, j in combinations(range(len(claims)), 2):
                            if (i, j) not in pairs_to_evaluate and (j, i) not in pairs_to_evaluate:
                                pairs_to_evaluate.append((i, j))
                
                    # Evaluate consistency for all pairs
                    if pairs_to_evaluate:
                        claim_pairs = [[claims[i], claims[j]] for i, j in pairs_to_evaluate]
                        scores = await self.openai_service.batch_evaluate_consistency(claim_pairs)
                        
                        # Add edges with consistency information
                        for (i, j), score in zip(pairs_to_evaluate, scores):
                            G.add_edge(i, j, consistency=score, is_consistent=(score >= 5.0))
                            # Add reverse edge with same consistency
                            G.add_edge(j, i, consistency=score, is_consistent=(score >= 5.0))
                    
                    # Return cycles and consistency score
                    return cycles, {
                        "consistency_score": consistency_score,
                        "inconsistency_description": analysis.get("inconsistency_description", "")
                    }
                
                # If no inconsistencies detected
                return [], {"consistency_score": analysis.get("consistency_score", 10.0)}
                
            except json.JSONDecodeError:
                logger.error(f"Could not parse LLM response as JSON: {response}")
                # Default to no inconsistencies
                return [], {"consistency_score": 10.0}
                
        except Exception as e:
            logger.error(f"Error in LLM-based inconsistency detection: {str(e)}")
            # Default to no inconsistencies
            return [], {"consistency_score": 10.0}
    
    async def visualize_inconsistencies(self, claims: List[str], cycles: List[List[int]], G=None) -> str:
        """
        Generate a visualization of inconsistencies.
        
        Args:
            claims: List of claims to visualize
            cycles: List of inconsistency cycles
            G: Graph to visualize (optional)
            
        Returns:
            Path to visualization
        """
        # Use stored graph if available and none provided
        if G is None and hasattr(self, 'g'):
            G = self.g
            
        # Delegate to the graph analyzer
        return self.graph_analyzer.visualize_inconsistency_network(claims, cycles, G)
    
    async def analyze_prompt(self, prompt: str, generate_visualization: bool = False) -> Dict[str, Any]:
        """Analyze a prompt for inconsistencies."""
        try:
            # Extract claims
            claims = await self.extract_claims(prompt)
            
            if not claims:
                return {
                    "consistency_score": 10.0,
                    "claims": [],
                    "cycles": [],
                    "inconsistent_pairs": [],
                    "pairwise_consistency": {},
                    "error": None
                }
            
            # Detect inconsistencies
            cycles, analysis_info = await self.detect_inconsistencies(claims)
            
            # Store consistency scores if available
            consistency_scores = {}
            
            # Track all pairwise consistency scores during detection
            if hasattr(self, 'g') and isinstance(self.g, nx.DiGraph):
                for u, v, data in self.g.edges(data=True):
                    if 'consistency' in data:
                        consistency_scores[(u, v)] = data['consistency']
            
            # Prepare response
            result = {
                "consistency_score": analysis_info.get("consistency_score", 10.0),
                "claims": claims,
                "cycles": cycles,
                "inconsistent_pairs": [],
                "pairwise_consistency": {},
                "error": None
            }
            
            # Format pairwise consistency scores for the response
            for (i, j), score in consistency_scores.items():
                # Only include each pair once (with smaller index first)
                if isinstance(i, int) and isinstance(j, int) and i < j:  # Avoid duplicates and non-integer nodes
                    key = f"{i}-{j}"
                    result["pairwise_consistency"][key] = round(score, 2)
            
            # Format inconsistent pairs
            for cycle in cycles:
                cycle_claims = [claims[i] for i in cycle]
                cycle_description = f"{' → '.join(cycle_claims)} → {cycle_claims[0]}"
                
                result["inconsistent_pairs"].append({
                    "cycle": cycle,
                    "description": cycle_description
                })
            
            # Generate visualization if requested
            if generate_visualization:
                vis_path = await self.visualize_inconsistencies(claims, cycles, self.g if hasattr(self, 'g') else None)
                # Format with complete URL for easy copying
                result["visualization_url"] = f"http://localhost:8000{vis_path}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {str(e)}")
            # Always return a valid structure even on error
            return {
                "consistency_score": 5.0,  # Default to neutral
                "claims": [],
                "cycles": [],
                "inconsistent_pairs": [],
                "pairwise_consistency": {},
                "error": f"Error analyzing prompt: {str(e)}"
            }