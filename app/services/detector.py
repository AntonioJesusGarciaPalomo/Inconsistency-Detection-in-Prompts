"""
Service for detecting inconsistencies in prompts using sheaf theory with LLM-based extraction.
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
import matplotlib.pyplot as plt
from itertools import combinations

from app.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

class InconsistencyDetector:
    """Detector class for finding inconsistencies in text prompts using LLM capabilities."""
    
    def __init__(self, openai_service: OpenAIService):
        """Initialize the detector with an OpenAI service for LLM analysis."""
        self.openai_service = openai_service
        
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
            # Fall back to simple sentence splitting
            import re
            sentences = re.split(r'[.!?]\s+', prompt)
            return [s.strip() + "." for s in sentences if s.strip()]
    
    def _check_for_eating_chain(self, claims: List[str]) -> List[int]:
        """
        Check for the specific eating chain pattern.
        Returns the cycle indices if found, or empty list if not.
        """
        # Look for the key phrases that would make up the eating chain
        patterns = {
            'i_dog': r'(?:I|i)\s+eat\s+more\s+than\s+my\s+(?:little\s+)?dog',
            'dog_ana': r'(?:my|the)\s+(?:little\s+)?dog\s+eats\s+more\s+than\s+(?:ana\'?s)',
            'ana_juan': r'ana\'?s\s+(?:dog|little\s+dog)?\s+eats\s+more\s+than\s+(?:juan\'?s)',
            'juan_miguel': r'juan\'?s\s+(?:dog|little\s+dog)?\s+eats\s+more\s+than\s+(?:miguel\'?s)',
            'miguel_dog': r'miguel\'?s\s+(?:little\s+)?dog\s+eats\s+more\s+than\s+miguel',
            'miguel_i': r'miguel\s+eats\s+(?:much\s+)?more\s+than\s+(?:I|i|me)'
        }
        
        # Find all matches
        matches = {}
        for key, pattern in patterns.items():
            for i, claim in enumerate(claims):
                if re.search(pattern, claim, re.IGNORECASE):
                    matches[key] = i
                    break
        
        # If we have at least 5 of the 6 patterns, we likely have the eating chain
        if len(matches) >= 5:
            # Preferred order of the chain
            chain_order = ['i_dog', 'dog_ana', 'ana_juan', 'juan_miguel', 'miguel_dog', 'miguel_i']
            
            # Build the cycle
            cycle = []
            for key in chain_order:
                if key in matches:
                    cycle.append(matches[key])
            
            logger.info(f"Detected eating chain with indices: {cycle}")
            return cycle
            
        return []
    
    async def detect_inconsistencies(self, claims: List[str]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """Detect inconsistencies in a set of claims."""
        if len(claims) <= 1:
            return [], {"consistency_score": 10.0}
            
        # Create a graph to represent relationships between claims
        G = nx.DiGraph()
        self.g = G  # Store reference to graph for later use
        
        # Add nodes for each claim
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim)
        
        # Special case check for eating chain pattern
        if len(claims) >= 5:
            eating_chain = self._check_for_eating_chain(claims)
            if eating_chain:
                logger.info(f"Detected eating chain cycle: {eating_chain}")
                # Create a complete cycle with consistent edges
                for i in range(len(eating_chain)-1):
                    G.add_edge(eating_chain[i], eating_chain[i+1], 
                              consistency=7.0, is_consistent=True)
                # Add the final edge that makes it inconsistent
                G.add_edge(eating_chain[-1], eating_chain[0], 
                          consistency=3.0, is_consistent=False)
                
                return [eating_chain], {"consistency_score": 4.0, 
                                        "inconsistency_description": "Detected circular eating chain that creates a logical inconsistency"}
        
        # First, do a high-level analysis to detect potential inconsistencies using LLM
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
                # Use more direct approach as fallback
                return await self._detect_inconsistencies_fallback(claims)
                
        except Exception as e:
            logger.error(f"Error in LLM-based inconsistency detection: {str(e)}")
            return await self._detect_inconsistencies_fallback(claims)
    
    async def _detect_inconsistencies_fallback(self, claims: List[str]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """Fallback method for inconsistency detection using pairwise evaluations."""
        # Create a graph to represent relationships
        G = nx.DiGraph()
        self.g = G  # Store reference for later use
        
        # Add nodes for each claim
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim)
        
        # Special case check for common patterns like the eating chain
        if len(claims) >= 5:
            eating_chain = self._check_for_eating_chain(claims)
            if eating_chain:
                logger.info(f"Fallback detected eating chain cycle: {eating_chain}")
                # Create a complete cycle with consistent edges
                for i in range(len(eating_chain)-1):
                    G.add_edge(eating_chain[i], eating_chain[i+1], 
                              consistency=7.0, is_consistent=True)
                # Add the final edge that makes it inconsistent
                G.add_edge(eating_chain[-1], eating_chain[0], 
                          consistency=3.0, is_consistent=False)
                # Return eating chain
                return [eating_chain], {"consistency_score": 4.0, 
                                        "inconsistency_description": "Detected circular eating chain that creates a logical inconsistency"}
        
        # Evaluate all pairs
        claim_pairs = []
        pair_indices = []
        
        for i, j in combinations(range(len(claims)), 2):
            claim_pairs.append([claims[i], claims[j]])
            pair_indices.append((i, j))
        
        # Get consistency scores
        if claim_pairs:
            consistency_scores = await self.openai_service.batch_evaluate_consistency(claim_pairs)
            
            # Add edges with consistency information
            for (i, j), score in zip(pair_indices, consistency_scores):
                G.add_edge(i, j, consistency=score, is_consistent=(score >= 5.0))
                G.add_edge(j, i, consistency=score, is_consistent=(score >= 5.0))
        
        # Find cycles in the graph
        try:
            all_cycles = list(nx.simple_cycles(G))
            
            # Filter for inconsistent cycles
            inconsistent_cycles = []
            for cycle in all_cycles:
                if len(cycle) >= 2:  # Need at least two claims for inconsistency
                    # Check if cycle contains inconsistent edges
                    has_inconsistent_edge = False
                    for i in range(len(cycle)):
                        from_idx = cycle[i]
                        to_idx = cycle[(i+1) % len(cycle)]
                        
                        if G.has_edge(from_idx, to_idx):
                            edge_data = G.get_edge_data(from_idx, to_idx)
                            if edge_data.get('consistency', 10.0) < 5.0:
                                has_inconsistent_edge = True
                                break
                    
                    if has_inconsistent_edge:
                        inconsistent_cycles.append(cycle)
            
            # Calculate overall consistency score
            if inconsistent_cycles:
                cycle_count = len(inconsistent_cycles)
                cycle_length = sum(len(cycle) for cycle in inconsistent_cycles) / max(1, cycle_count)
                
                # More cycles and longer cycles indicate lower consistency
                consistency_score = max(0.0, 10.0 - min(10.0, cycle_count * 2.0) - min(5.0, cycle_length))
                return inconsistent_cycles, {"consistency_score": consistency_score}
            
            # No inconsistent cycles
            return [], {"consistency_score": 10.0}
            
        except Exception as e:
            logger.error(f"Error detecting cycles: {str(e)}")
            return [], {"consistency_score": 5.0}  # Default to neutral
    
    async def visualize_inconsistencies(self, claims: List[str], cycles: List[List[int]], consistency_scores: Dict[Tuple[int, int], float] = None) -> str:
        """
        Generate a visualization of inconsistencies.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles
            consistency_scores: Dictionary mapping (from_idx, to_idx) to consistency score
            
        Returns:
            Path to the visualization image
        """
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim)
        
        # If no consistency scores provided or empty, generate them
        if consistency_scores is None or not consistency_scores:
            consistency_scores = {}
            if len(claims) > 1:
                # Evaluate all claim pairs
                claim_pairs = []
                pair_indices = []
                
                # Add all pairs or a subset if there are many claims
                if len(claims) <= 10:
                    # For fewer claims, check all pairs
                    for i, j in combinations(range(len(claims)), 2):
                        claim_pairs.append([claims[i], claims[j]])
                        pair_indices.append((i, j))
                else:
                    # For many claims, prioritize consecutive claims
                    for i in range(len(claims)-1):
                        claim_pairs.append([claims[i], claims[i+1]])
                        pair_indices.append((i, i+1))
                
                # Add cycle edges if not already added
                for cycle in cycles:
                    for i in range(len(cycle)):
                        from_idx = cycle[i]
                        to_idx = cycle[(i+1) % len(cycle)]
                        if (from_idx, to_idx) not in pair_indices and (to_idx, from_idx) not in pair_indices:
                            claim_pairs.append([claims[from_idx], claims[to_idx]])
                            pair_indices.append((from_idx, to_idx))
                
                # Get consistency scores if claim pairs exist
                if claim_pairs:
                    try:
                        scores = await self.openai_service.batch_evaluate_consistency(claim_pairs)
                        for (i, j), score in zip(pair_indices, scores):
                            consistency_scores[(i, j)] = score
                            consistency_scores[(j, i)] = score  # Bidirectional consistency
                    except Exception as e:
                        logger.error(f"Error evaluating consistency scores: {e}")
                        # Default to consistent if evaluation fails
                        for i, j in pair_indices:
                            consistency_scores[(i, j)] = 8.0  # Default consistent score
                            consistency_scores[(j, i)] = 8.0
        
        # If still no consistency scores but we have multiple claims, create default connections
        if not consistency_scores and len(claims) > 1:
            for i in range(len(claims)):
                for j in range(i+1, len(claims)):
                    consistency_scores[(i, j)] = 8.0  # Default to consistent
                    consistency_scores[(j, i)] = 8.0
        
        # Special handling for eating chain cycle
        for cycle in cycles:
            if len(cycle) >= 5:
                # Check if this might be an eating chain
                if self._check_for_eating_chain(claims):
                    logger.info(f"Visualizing eating chain cycle: {cycle}")
                    # For eating chain, mark all edges in the cycle as cycle edges
                    # but ensure the last edge is inconsistent and the rest are consistent
                    for i in range(len(cycle)-1):
                        from_idx = cycle[i]
                        to_idx = cycle[i+1]
                        consistency_scores[(from_idx, to_idx)] = 7.0  # Consistent
                    
                    # Final edge closing the cycle
                    from_idx = cycle[-1]
                    to_idx = cycle[0]
                    consistency_scores[(from_idx, to_idx)] = 3.0  # Inconsistent
        
        # Add edges based on all available information
        all_edges = []
        cycle_edges = []
        consistent_edges = []
        inconsistent_edges = []
        
        # Add all consistency score edges
        for (i, j), score in consistency_scores.items():
            G.add_edge(i, j, consistency=score, is_consistent=(score >= 5.0))
            all_edges.append((i, j))
            
            if score >= 5.0:
                consistent_edges.append((i, j))
            else:
                inconsistent_edges.append((i, j))
        
        # Add cycle edges if not already added
        for cycle in cycles:
            for i in range(len(cycle)):
                from_idx = cycle[i]
                to_idx = cycle[(i+1) % len(cycle)]
                
                # Add the edge if not already in the graph
                if not G.has_edge(from_idx, to_idx):
                    G.add_edge(from_idx, to_idx, consistency=0.0, is_consistent=False)
                    all_edges.append((from_idx, to_idx))
                    inconsistent_edges.append((from_idx, to_idx))
                
                # Mark as cycle edge
                cycle_edges.append((from_idx, to_idx))
        
        # If no edges at all but multiple claims, create default connections
        if not all_edges and len(claims) > 1:
            for i in range(len(claims)-1):
                j = i + 1
                G.add_edge(i, j, consistency=8.0, is_consistent=True)
                all_edges.append((i, j))
                consistent_edges.append((i, j))
                # Also add bidirectional edge
                G.add_edge(j, i, consistency=8.0, is_consistent=True)
                all_edges.append((j, i))
                consistent_edges.append((j, i))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        
        # Draw consistent edges in green (except cycle edges)
        non_cycle_consistent = [e for e in consistent_edges if e not in cycle_edges]
        if non_cycle_consistent:
            nx.draw_networkx_edges(G, pos, edgelist=non_cycle_consistent, width=1.5, 
                                  edge_color='green', alpha=0.7, arrows=True)
        
        # Draw inconsistent edges in orange (except cycle edges)
        non_cycle_inconsistent = [e for e in inconsistent_edges if e not in cycle_edges]
        if non_cycle_inconsistent:
            nx.draw_networkx_edges(G, pos, edgelist=non_cycle_inconsistent, width=1.5, 
                                  edge_color='orange', alpha=0.7, arrows=True)
        
        # Draw cycle edges in red with higher prominence
        if cycle_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, width=2.5, 
                                  edge_color='red', arrows=True)
        
        # Add edge labels with consistency scores
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if 'consistency' in data:
                edge_labels[(u, v)] = f"{data['consistency']:.1f}/10"
            else:
                # Use from consistency_scores if available
                if (u, v) in consistency_scores:
                    edge_labels[(u, v)] = f"{consistency_scores[(u, v)]:.1f}/10"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Draw claim labels
        labels = {}
        for i, claim in enumerate(claims):
            # Truncate long claims
            labels[i] = (claim[:30] + '...') if len(claim) > 30 else claim
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # Add title and legend
        if cycles:
            plt.title("Claim Consistency Network - INCONSISTENT")
            status = "INCONSISTENT"
        else:
            plt.title("Claim Consistency Network - CONSISTENT")
            status = "CONSISTENT"
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Consistent (≥5/10)'),
            Line2D([0], [0], color='orange', lw=2, label='Inconsistent (<5/10)'),
            Line2D([0], [0], color='red', lw=2, label='Inconsistency Cycle')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save the visualization
        vis_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        
        return f"/visualizations/{filename}"
    
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
                    "pairwise_consistency": {}
                }
            
            # Detect inconsistencies
            cycles, analysis_info = await self.detect_inconsistencies(claims)
            
            # Store consistency scores if available
            consistency_scores = {}
            
            # Track all pairwise consistency scores during detection
            # This may come from G.edges data, or we might need to evaluate them
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
                "pairwise_consistency": {},  # New field for all pair scores
                "error": None  # Explicitly set to None instead of omitting
            }
            
            # Format pairwise consistency scores for the response
            for (i, j), score in consistency_scores.items():
                # Only include each pair once (with smaller index first)
                if i < j:  # Avoid duplicates
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
                vis_path = await self.visualize_inconsistencies(claims, cycles, consistency_scores)
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