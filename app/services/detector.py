"""
Service for detecting inconsistencies in prompts using language models.
"""
import logging
import re
import json
from typing import List, Dict, Any, Tuple, Optional, Literal
import networkx as nx

from app.services.openai_service import OpenAIService
from app.utils.graph_analyzer import GraphAnalyzer
from app.utils.interactive_graph_analyzer import InteractiveGraphAnalyzer

logger = logging.getLogger(__name__)

class InconsistencyDetector:
    """Detector class for finding inconsistencies in text prompts using LLM capabilities."""
    
    def __init__(self, openai_service: OpenAIService, base_url: str = "http://localhost:8000"):
        """
        Initialize the detector with an OpenAI service for LLM analysis.
        
        Args:
            openai_service: Service for interacting with OpenAI API
            base_url: Base URL for visualization links, defaults to localhost:8000
        """
        self.openai_service = openai_service
        self.graph_analyzer = GraphAnalyzer()
        self.interactive_graph_analyzer = InteractiveGraphAnalyzer()
        self.base_url = base_url
        
    async def extract_claims(self, prompt: str) -> List[str]:
        """Extract atomic claims from a prompt using LLM."""
        system_message = """
        Extract ALL atomic claims from the provided text. An atomic claim is a single, 
        self-contained factual statement.

        You MUST extract ALL separate facts, including statements about likes, dislikes, 
        preferences, and contradictions.

        Format your response as a list of claims, one per line, prefixed with "- ".
        
        IMPORTANT: Make sure to extract EVERY claim, even if they appear to contradict each other.
        Do not try to resolve contradictions - just extract all statements as they appear.
        """
        
        try:
            response = await self.openai_service.get_completion(
                system_message=system_message,
                user_message=prompt
            )
            
            # Extract claims, removing bullet points
            claims = []
            for line in response.split("\n"):
                line = line.strip()
                if line and (line.startswith("- ") or line.startswith("* ")):
                    claim = line[2:].strip()
                    claims.append(claim)
            
            # If no claims were found with bullet points, try splitting by sentences
            if not claims:
                sentences = re.split(r'[.!?]\s+', prompt)
                claims = [s.strip() + "." for s in sentences if s.strip()]
                
            logger.info(f"Extracted {len(claims)} claims from prompt")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            # Fall back to simple sentence splitting
            import re
            sentences = re.split(r'[.!?]\s+', prompt)
            return [s.strip() + "." for s in sentences if s.strip()]
    
    async def detect_inconsistencies(self, claims: List[str]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Detect inconsistencies in a set of claims.
        
        Args:
            claims: List of claim statements
            
        Returns:
            Tuple of (list of detected cycles, analysis info dictionary)
        """
        if len(claims) <= 1:
            return [], {"consistency_score": 10.0}
        
        # Create a directed graph to track claim relationships
        G = nx.DiGraph()
        self._add_claims_to_graph(G, claims)
        
        # Get LLM analysis of the claims
        analysis = await self._get_llm_analysis(claims)
        
        # Extract key information from the analysis
        inconsistencies_detected = analysis.get("inconsistencies_detected", False)
        consistency_score = analysis.get("consistency_score", 10.0)
        inconsistency_description = analysis.get("inconsistency_description", "")
        
        # Get and validate inconsistency cycles
        claim_cycles = analysis.get("inconsistent_claim_indices", [])
        valid_cycles = self._validate_cycles(claim_cycles, len(claims))
        
        # If inconsistencies are detected but no valid cycles, try additional detection methods
        if inconsistencies_detected and not valid_cycles:
            valid_cycles = await self._perform_additional_cycle_detection(claims)
        
        # If we still don't have cycles but inconsistencies were detected, use all claims as a cycle
        if inconsistencies_detected and not valid_cycles:
            valid_cycles = [[i for i in range(len(claims))]]
        
        # Add edges to the graph based on detected cycles
        self._add_cycle_edges_to_graph(G, valid_cycles)
        
        # Store the graph for visualization
        self.g = G
        
        # Return results
        return valid_cycles, {
            "consistency_score": consistency_score,
            "inconsistency_description": inconsistency_description
        }
    
    def _add_claims_to_graph(self, G: nx.DiGraph, claims: List[str]) -> None:
        """
        Add all claims as nodes to the graph.
        
        Args:
            G: NetworkX directed graph
            claims: List of claim statements
        """
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim)
    
    async def _get_llm_analysis(self, claims: List[str]) -> Dict[str, Any]:
        """
        Get LLM analysis of claims to detect inconsistencies.
        
        Args:
            claims: List of claim statements
            
        Returns:
            Analysis dictionary with inconsistency information
        """
        system_message = """
        You are an inconsistency detection expert. Analyze the claims and find logical inconsistencies.

        Pay special attention to these types of inconsistencies:
        1. Direct contradictions (e.g., "I like X" vs "I don't like X")
        2. Comparative contradictions (e.g., "A > B" and "B > A")
        3. Circular dependencies (e.g., "A > B > C > A")
        4. Transitive relationship violations (e.g., if "A > B" and "B > C" then "A > C" must be true)
        
        RESPONSE FORMAT INSTRUCTIONS:
        - Respond ONLY with a valid JSON object
        - Do NOT include any explanations, markdown formatting, or code blocks
        - Do NOT use backticks around your JSON
        - Ensure all JSON keys and values are properly quoted
        - Avoid trailing commas

        Your JSON object MUST have exactly these fields:
        {
          "inconsistencies_detected": true/false,
          "consistency_score": 0-10,
          "inconsistency_description": "Description of inconsistencies",
          "inconsistent_claim_indices": [[i,j,k...]]
        }
        
        For "inconsistent_claim_indices", include each cycle as a list of indices in the proper order that forms the cycle.
        For example, if claims 0, 2, and 4 form a cycle in that order, include [0,2,4].
        """

        claims_text = "\n".join([f"{i}. {claim}" for i, claim in enumerate(claims)])
        user_message = f"""
        Analyze these claims for logical inconsistencies, especially focusing on CONTRADICTIONS, 
        CIRCULAR DEPENDENCIES, and TRANSITIVE RELATIONSHIP VIOLATIONS:
        
        {claims_text}
        
        Look carefully for patterns where multiple claims together create a cycle or contradiction,
        even if individual pairs seem consistent. For example: "A > B", "B > C", "C > A" creates a cycle.
        
        Remember to ONLY return a valid JSON object with the exact format specified - no text, no markdown, no explanations.
        """
        
        try:
            # Get LLM analysis
            response = await self.openai_service.get_completion(
                system_message=system_message,
                user_message=user_message,
                temperature=0.0  # Use zero temperature for more deterministic responses
            )
            
            # Parse the JSON response
            return self._parse_llm_response(response, len(claims))
                
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {str(e)}")
            return {
                "inconsistencies_detected": False,
                "consistency_score": 5.0,
                "inconsistent_claim_indices": []
            }
    
    def _parse_llm_response(self, response: str, num_claims: int) -> Dict[str, Any]:
        """
        Parse the LLM response to extract analysis information with robust error handling.
        
        Args:
            response: String response from LLM
            num_claims: Number of claims (used for fallback)
            
        Returns:
            Parsed analysis dictionary
        """
        # Clean up the response
        cleaned_response = self._preprocess_json_response(response)
        
        # Try multiple parsing approaches in sequence
        try:
            # 1. First attempt: direct parsing
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {str(e)}")
            
            try:
                # 2. Second attempt: extract JSON using regex
                json_match = re.search(r'({[\s\S]*})', cleaned_response)
                if json_match:
                    extracted_json = json_match.group(1)
                    # Clean up extracted JSON
                    extracted_json = self._preprocess_json_response(extracted_json)
                    return json.loads(extracted_json)
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Regex JSON extraction failed: {str(e)}")
                
                try:
                    # 3. Third attempt: Fix common JSON issues and try again
                    fixed_json = self._fix_common_json_issues(cleaned_response)
                    return json.loads(fixed_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Fixed JSON parsing failed: {str(e)}")
                    
                    # 4. Final attempt: Extract key fields using regex patterns
                    return self._extract_analysis_with_regex(cleaned_response, num_claims)

    def _preprocess_json_response(self, text: str) -> str:
        """
        Preprocess a JSON response text to make it more likely to parse correctly.
        
        Args:
            text: Raw response text
            
        Returns:
            Preprocessed text
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Remove any text before the first '{' and after the last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx+1]
        
        return text.strip()

    def _fix_common_json_issues(self, text: str) -> str:
        """
        Fix common JSON formatting issues.
        
        Args:
            text: JSON text with potential issues
            
        Returns:
            Fixed JSON text
        """
        # Remove trailing commas before closing brackets
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix missing quotes around keys
        text = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
        
        # Fix single quotes used instead of double quotes (but only for keys and string values)
        # This regex handles the typical patterns without trying to parse the entire JSON
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)  # for keys
        text = re.sub(r':\s*\'([^\']*)\'([,}])', r': "\1"\2', text)  # for values
        
        # Fix unquoted true/false/null
        text = re.sub(r':\s*true([,}])', r': true\1', text)
        text = re.sub(r':\s*false([,}])', r': false\1', text)
        text = re.sub(r':\s*null([,}])', r': null\1', text)
        
        return text

    def _extract_analysis_with_regex(self, text: str, num_claims: int) -> Dict[str, Any]:
        """
        Extract key analysis fields using regex when JSON parsing fails.
        
        Args:
            text: Text containing analysis information
            num_claims: Number of claims
            
        Returns:
            Extracted analysis dictionary
        """
        logger.info("Falling back to regex extraction for analysis")
        analysis = {
            "inconsistencies_detected": False,
            "consistency_score": 5.0,
            "inconsistency_description": "",
            "inconsistent_claim_indices": []
        }
        
        # Extract if inconsistencies were detected
        inconsistency_match = re.search(
            r'"inconsistencies_detected"\s*:\s*(true|false)', 
            text, 
            re.IGNORECASE
        )
        if inconsistency_match:
            analysis["inconsistencies_detected"] = inconsistency_match.group(1).lower() == "true"
        
        # Extract consistency score
        score_match = re.search(
            r'"consistency_score"\s*:\s*(\d+(?:\.\d+)?)', 
            text
        )
        if score_match:
            try:
                analysis["consistency_score"] = float(score_match.group(1))
            except ValueError:
                pass
        
        # Extract inconsistency description
        description_match = re.search(
            r'"inconsistency_description"\s*:\s*"([^"]*)"', 
            text
        )
        if description_match:
            analysis["inconsistency_description"] = description_match.group(1)
        
        # Extract inconsistent claim indices
        # This is the most complex part as it's a nested structure
        indices_match = re.search(
            r'"inconsistent_claim_indices"\s*:\s*(\[.*?\])', 
            text, 
            re.DOTALL
        )
        
        if indices_match:
            indices_str = indices_match.group(1)
            try:
                # Try to parse just this part as JSON
                indices = json.loads(indices_str)
                if isinstance(indices, list):
                    analysis["inconsistent_claim_indices"] = indices
            except json.JSONDecodeError:
                # If that fails, try to extract individual cycles with regex
                cycle_matches = re.findall(r'\[([\d\s,]+)\]', indices_str)
                for cycle_match in cycle_matches:
                    try:
                        cycle = [int(i.strip()) for i in cycle_match.split(',') if i.strip()]
                        if len(cycle) >= 2:
                            analysis["inconsistent_claim_indices"].append(cycle)
                    except ValueError:
                        continue
        
        # If inconsistencies were detected but no cycles were found, create a default cycle
        if analysis["inconsistencies_detected"] and not analysis["inconsistent_claim_indices"]:
            if text.lower().find("cycle") != -1 or text.lower().find("circular") != -1:
                analysis["inconsistent_claim_indices"] = [[i for i in range(num_claims)]]
        
        return analysis
    
    def _validate_cycles(self, claim_cycles: List[List[int]], num_claims: int) -> List[List[int]]:
        """
        Validate and normalize cycles to ensure they contain valid claim indices.
        
        Args:
            claim_cycles: List of cycles (each a list of claim indices)
            num_claims: Total number of claims
            
        Returns:
            List of valid cycles
        """
        valid_cycles = []
        for cycle in claim_cycles:
            # Verify all items are integers and within range
            valid_cycle = []
            for idx in cycle:
                try:
                    idx_int = int(idx)
                    if 0 <= idx_int < num_claims:
                        valid_cycle.append(idx_int)
                except (ValueError, TypeError):
                    continue
            
            # Only include cycles with at least 2 claims
            if len(valid_cycle) >= 2:
                valid_cycles.append(valid_cycle)
        
        return valid_cycles
    
    async def _perform_additional_cycle_detection(self, claims: List[str]) -> List[List[int]]:
        """
        Perform additional cycle detection using pattern analysis when LLM doesn't provide specific cycles.
        
        Args:
            claims: List of claim statements
            
        Returns:
            List of detected cycles
        """
        try:
            # Find relationships that likely form cycles
            relation_patterns = self._detect_relationship_patterns(claims)
            if relation_patterns:
                # Use the patterns to identify potential cycles
                potential_cycles = self._find_cycles_from_patterns(relation_patterns, len(claims))
                if potential_cycles:
                    return potential_cycles
        except Exception as e:
            logger.error(f"Error in additional cycle detection: {str(e)}")
        
        # If no cycles found, return empty list
        return []
    
    def _add_cycle_edges_to_graph(self, G: nx.DiGraph, cycles: List[List[int]]) -> None:
        """
        Add edges to the graph based on detected inconsistency cycles.
        
        Args:
            G: NetworkX directed graph
            cycles: List of inconsistency cycles
        """
        num_nodes = len(G.nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Check if this edge is part of a cycle
                    in_cycle = False
                    for cycle in cycles:
                        # Check if both i and j are in the cycle
                        if i in cycle and j in cycle:
                            cycle_i = cycle.index(i)
                            cycle_j = cycle.index(j)
                            # Check if j directly follows i in the cycle
                            if cycle_j == (cycle_i + 1) % len(cycle):
                                in_cycle = True
                                break
                    
                    is_consistent = not in_cycle
                    consistency = 8.0 if is_consistent else 2.0
                    
                    G.add_edge(i, j, 
                              consistency=consistency,
                              is_consistent=is_consistent,
                              is_cycle_edge=in_cycle)
    
    def _detect_relationship_patterns(self, claims: List[str]) -> List[Tuple[int, int]]:
        """
        Detect potential relationship patterns in claims that might form cycles.
        
        Args:
            claims: List of claim strings
            
        Returns:
            List of (from_idx, to_idx) tuples representing potential relationships
        """
        # Keywords that often indicate comparative relationships
        comparative_terms = ["more than", "less than", "greater", "bigger", "smaller", 
                             "older", "younger", "taller", "shorter", "faster", "slower",
                             "heavier", "lighter", "stronger", "weaker", "better", "worse",
                             "higher", "lower", "larger", "exceeds", "precedes"]
        
        relationships = []
        
        # Look for comparative relationships
        for i, claim1 in enumerate(claims):
            claim1_lower = claim1.lower()
            
            # Skip claims that don't contain comparative terms
            if not any(term in claim1_lower for term in comparative_terms):
                continue
                
            for j, claim2 in enumerate(claims):
                if i == j:
                    continue
                    
                # Look for potential connections between claims
                # This is a heuristic approach - the LLM should still be the primary detector
                for term in comparative_terms:
                    if term in claim1_lower:
                        # Extract potential subjects on both sides of the comparison
                        parts = claim1_lower.split(term, 1)
                        if len(parts) == 2:
                            subject_a = parts[0].strip().split()[-1] if parts[0].strip() else ""
                            subject_b = parts[1].strip().split()[0] if parts[1].strip() else ""
                            
                            # Check if either subject appears in claim2
                            if (subject_a and subject_a in claim2.lower()) or (subject_b and subject_b in claim2.lower()):
                                relationships.append((i, j))
                                break
        
        return relationships

    def _find_cycles_from_patterns(self, relationships: List[Tuple[int, int]], num_claims: int) -> List[List[int]]:
        """
        Find potential cycles from the detected relationship patterns.
        
        Args:
            relationships: List of (from_idx, to_idx) tuples
            num_claims: Total number of claims
            
        Returns:
            List of cycles where each cycle is a list of claim indices
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all claims as nodes
        for i in range(num_claims):
            G.add_node(i)
        
        # Add relationships as edges
        for from_idx, to_idx in relationships:
            G.add_edge(from_idx, to_idx)
        
        # Find simple cycles
        try:
            cycles = list(nx.simple_cycles(G))
            return [cycle for cycle in cycles if len(cycle) >= 2]
        except nx.NetworkXNoCycle:
            return []
        except Exception as e:
            logger.error(f"Error finding cycles: {str(e)}")
            return []
    
    async def visualize_inconsistencies(self, claims: List[str], cycles: List[List[int]]) -> str:
        """
        Generate a static visualization of inconsistencies.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles
            
        Returns:
            Path to saved visualization image
        """
        # Use the graph created during detection
        if hasattr(self, 'g'):
            return await self.graph_analyzer.visualize_inconsistency_network(claims, cycles, self.g)
        else:
            return await self.graph_analyzer.visualize_inconsistency_network(claims, cycles)
    
    async def visualize_inconsistencies_interactive(self, claims: List[str], cycles: List[List[int]]) -> str:
        """
        Generate an interactive HTML visualization of inconsistencies.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles
            
        Returns:
            Path to saved HTML visualization
        """
        # Use the graph created during detection if available
        if hasattr(self, 'g'):
            return await self.interactive_graph_analyzer.visualize_inconsistency_network(claims, cycles, self.g)
        else:
            return await self.interactive_graph_analyzer.visualize_inconsistency_network(claims, cycles)
    
    async def analyze_prompt(self, 
                            prompt: str, 
                            generate_visualization: bool = False,
                            visualization_type: Literal["static", "interactive"] = "static") -> Dict[str, Any]:
        """
        Analyze a prompt for inconsistencies.
        
        Args:
            prompt: Text prompt to analyze
            generate_visualization: Whether to generate a visualization
            visualization_type: Type of visualization to generate ('static' or 'interactive')
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract claims
            claims = await self.extract_claims(prompt)
            
            if not claims:
                return self._create_empty_result()
            
            # Detect inconsistencies
            cycles, analysis_info = await self.detect_inconsistencies(claims)
            
            # Create and format the result
            result = self._format_analysis_result(claims, cycles, analysis_info)
            
            # Generate visualization if requested
            if generate_visualization:
                try:
                    if visualization_type == "interactive":
                        # Generate interactive HTML visualization
                        vis_path = await self.visualize_inconsistencies_interactive(claims, cycles)
                    else:
                        # Generate static image visualization
                        vis_path = await self.visualize_inconsistencies(claims, cycles)
                    
                    if vis_path:
                        result["visualization_url"] = f"{self.base_url}{vis_path}"
                        # Add the visualization type to the result
                        result["visualization_type"] = visualization_type
                except Exception as e:
                    logger.error(f"Error generating visualization: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {str(e)}")
            return self._create_error_result(str(e))
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result when no claims are found."""
        return {
            "consistency_score": 10.0,
            "claims": [],
            "cycles": [],
            "inconsistent_pairs": [],
            "visualization_url": None,
            "visualization_type": None,
            "error": None
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result when exception occurs."""
        return {
            "consistency_score": 5.0,
            "claims": [],
            "cycles": [],
            "inconsistent_pairs": [],
            "visualization_url": None,
            "visualization_type": None,
            "error": f"Error analyzing prompt: {error_message}"
        }
    
    def _format_analysis_result(self, claims: List[str], cycles: List[List[int]], 
                               analysis_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format the analysis result dictionary."""
        result = {
            "consistency_score": analysis_info.get("consistency_score", 10.0),
            "claims": claims,
            "cycles": cycles,
            "inconsistent_pairs": [],
            "visualization_url": None,
            "visualization_type": None,
            "error": None
        }
        
        # Format inconsistent pairs
        for cycle in cycles:
            claim_texts = [claims[idx] for idx in cycle if 0 <= idx < len(claims)]
            if claim_texts:
                cycle_description = " → ".join(claim_texts)
                if len(claim_texts) > 0:
                    cycle_description += f" → {claim_texts[0]}"
                
                result["inconsistent_pairs"].append({
                    "cycle": cycle,
                    "description": cycle_description
                })
        
        return result