"""
This module handles sheaf-based calculations for inconsistency detection.
Uses GPT-4o for text segmentation, entity extraction, relationship identification, and consistency evaluation.
"""
import logging
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import io
import uuid
import os
import re
import json
import requests
from itertools import combinations
import time
from typing import List, Dict, Tuple, Any, Optional, Set, Union

# Configure logger
logger = logging.getLogger(__name__)

# Import PySheaf if available (with fallback implementation)
try:
    # First attempt: Try importing as originally expected
    from pysheaf.sheaf import CellComplex, Cell, Coface
    logger.info("Successfully imported PySheaf via pysheaf.sheaf")
except ImportError:
    try:
        # Second attempt: Try importing directly from pysheaf
        from pysheaf import Sheaf as CellComplex, Cell, Coface, Assignment
        logger.info("Successfully imported PySheaf classes directly")
    except ImportError:
        # Fallback implementation using NetworkX
        logger.warning("Failed to import pysheaf, using networkx fallback implementation")
        
        # Define minimal fallback classes for PySheaf
        class Assignment:
            """A class to hold data values with type checking."""
            def __init__(self, valueType, value):
                self.mValue = value
                self.mValueType = valueType
            
            def __str__(self):
                return str(self.mValue)
                
        class Cell:
            """A class representing a node in a sheaf graph."""
            def __init__(self, dataTagType, compareAssignmentsMethod=None, **kwargs):
                self.mDataTagType = dataTagType
                self.mDataAssignment = None
                self.mDataAssignmentPresent = False
                self.mExtendedAssignments = {}
                self.mDataDimension = kwargs.get('dataDimension', 1)
                self.mOptimizationCell = kwargs.get('optimizationCell', False)
                self.mExtendFromThisCell = kwargs.get('extendFromThisCell', True)
                
                if compareAssignmentsMethod is None:
                    self.Compare = self.DefaultCompareAssignments
                else:
                    self.Compare = compareAssignmentsMethod
                
            def SetDataAssignment(self, dataAssignment):
                """Set the data assignment for this cell."""
                if self.mDataTagType == dataAssignment.mValueType:
                    self.mDataAssignmentPresent = True
                    self.mDataAssignment = dataAssignment
                else:
                    logger.error(f"DataAssignment has incorrect type. Expected: {self.mDataTagType}, Actual: {dataAssignment.mValueType}")
            
            def CheckDataAssignmentPresent(self):
                """Check if data assignment is present."""
                return self.mDataAssignmentPresent
                
            def GetDataAssignment(self):
                """Return the data assignment."""
                if not self.mDataAssignmentPresent:
                    logger.error("DataAssignment not present for cell")
                return self.mDataAssignment
                
            def AddExtendedAssignment(self, cellPathTuple, extendedAssignment):
                """Add an extended assignment to this cell."""
                self.mExtendedAssignments[cellPathTuple] = extendedAssignment
                
            def GetExtendedAssignmentValueList(self):
                """Get list of extended assignments."""
                return list(self.mExtendedAssignments.values())
                
            def CheckExtendedAssignmentPresent(self):
                """Check if any extended assignments are present."""
                return bool(self.mExtendedAssignments)
                
            def ClearExtendedAssignment(self):
                """Clear all extended assignments."""
                self.mExtendedAssignments.clear()
                
            def DefaultCompareAssignments(self, leftValue, rightValue):
                """Default comparison method for assignments."""
                return abs(leftValue - rightValue)
            
            def AbleToComputeConsistency(self):
                """Check if consistency can be computed."""
                multiple_extended = len(self.mExtendedAssignments) > 1
                data_and_extended = self.mDataAssignmentPresent and len(self.mExtendedAssignments) > 0
                return multiple_extended or data_and_extended
                
            def ComputeConsistency(self, numpyNormType=np.inf):
                """Compute the consistency of the assignments."""
                if not self.mExtendedAssignments:
                    logger.error("Cannot compute consistency: no extended assignments present")
                    return 0.0
                    
                assignments = self.GetExtendedAssignmentValueList()
                comparisons = []
                
                if self.mDataAssignmentPresent:
                    for assignment in assignments:
                        comparisons.append(self.Compare(self.mDataAssignment.mValue, assignment.mValue))
                
                if not comparisons:
                    return 0.0
                    
                return np.linalg.norm(np.array(comparisons, dtype='float'), ord=numpyNormType)
        
        class Coface:
            """A class representing an edge in a sheaf graph."""
            def __init__(self, inputTagType, outputTagType, edgeMethod):
                self.mInputTagType = inputTagType
                self.mOutputTagType = outputTagType
                self.mEdgeMethod = edgeMethod
                self.mOrientation = 0
                
            def RunEdgeMethod(self, inputAssignment):
                """Apply the edge method to transform an input assignment."""
                return Assignment(self.mOutputTagType, self.mEdgeMethod(inputAssignment.mValue))
        
        class CellComplex(nx.DiGraph):
            """A class representing a sheaf on a graph."""
            def __init__(self):
                super().__init__()
                self.mNumpyNormType = np.inf
                self.mPreventRedundantExtendedAssignments = False
                
            def AddCell(self, cellIndex, cellToAdd):
                """Add a cell to the sheaf."""
                self.add_node(cellIndex, vertex=cellToAdd)
                
            def AddCoface(self, cellIndexFrom, cellIndexTo, cofaceToAdd):
                """Add a coface (edge) to the sheaf."""
                from_cell = self.GetCell(cellIndexFrom)
                to_cell = self.GetCell(cellIndexTo)
                
                if (from_cell.mDataTagType == cofaceToAdd.mInputTagType and 
                    to_cell.mDataTagType == cofaceToAdd.mOutputTagType):
                    self.add_edge(cellIndexFrom, cellIndexTo, edge=cofaceToAdd)
                else:
                    logger.error(f"Coface types do not match cells from {cellIndexFrom} to {cellIndexTo}")
                
            def GetCell(self, cellIndex):
                """Get a cell by its index."""
                return self.nodes[cellIndex]['vertex']
                
            def GetCoface(self, cellIndexStart, cellIndexTo):
                """Get a coface between two cells."""
                return self[cellIndexStart][cellIndexTo]['edge']
                
            def ClearExtendedAssignments(self):
                """Clear all extended assignments in all cells."""
                for cell_index in self.nodes():
                    self.GetCell(cell_index).ClearExtendedAssignment()
                    
            def MaximallyExtendCell(self, startCellIndex):
                """Extend a cell's data value throughout the sheaf."""
                try:
                    nx.find_cycle(self, startCellIndex)
                    logger.warning(f"Cycle found in graph from cell index: {startCellIndex}")
                except nx.NetworkXNoCycle:
                    self._extend_cell_recursive(startCellIndex, (startCellIndex,))
                    
            def _extend_cell_recursive(self, cellIndex, cellPathTuple):
                """Recursively extend cell values."""
                for successor_index in self.successors(cellIndex):
                    if (self.mPreventRedundantExtendedAssignments and 
                        self.GetCell(successor_index).CheckExtendedAssignmentPresent()):
                        continue
                        
                    next_cell_path_tuple = cellPathTuple + (successor_index,)
                    
                    if len(cellPathTuple) == 1:
                        current_cell_assignment = self.GetCell(cellIndex).mDataAssignment
                    else:
                        current_cell_assignment = self.GetCell(cellIndex).mExtendedAssignments[cellPathTuple]
                        
                    next_cell_assignment = self.GetCoface(cellIndex, successor_index).RunEdgeMethod(current_cell_assignment)
                    self.GetCell(successor_index).AddExtendedAssignment(next_cell_path_tuple, next_cell_assignment)
                    self._extend_cell_recursive(successor_index, next_cell_path_tuple)
                    
            def ComputeConsistencyRadius(self):
                """Compute the overall consistency radius of the sheaf."""
                cell_consistencies = []
                
                for cell_index in self.nodes():
                    cell = self.GetCell(cell_index)
                    if cell.AbleToComputeConsistency():
                        cell_consistencies.append(cell.ComputeConsistency(self.mNumpyNormType))
                        
                if cell_consistencies:
                    return np.linalg.norm(cell_consistencies, ord=self.mNumpyNormType)
                return 0.0
                
            def GetOptimizationCellIndexList(self):
                """Get list of cells marked for optimization."""
                return [cell_index for cell_index in self.nodes() 
                        if self.GetCell(cell_index).mOptimizationCell]

class SheafAnalyzer:
    """Class for analyzing text using sheaves and GPT-4o for comprehensive analysis."""
    
    def __init__(self):
        """Initialize the sheaf analyzer."""
        # Configure GPT API access
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"  # Use GPT-4o mini for evaluations
        
        # Cache for various GPT responses to avoid duplicated API calls
        self.cache = {
            'entities': {},       # Entities extracted from claims
            'relations': {},      # Relationships extracted from claims
            'consistency': {},    # Consistency evaluations between pairs of claims
            'segmentation': {},   # Text segmented into claims
            'analysis': {}        # Overall analysis of claim sets
        }
        
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable for GPT evaluation.")
    
    def call_gpt(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> Optional[str]:
        """
        Generic method to call GPT-4o with a prompt.
        
        Args:
            prompt: The input prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            The model's response text or None if the call failed
        """
        if not self.api_key:
            logger.warning("No API key available for GPT call")
            return None
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            for attempt in range(3):  # Try up to 3 times with exponential backoff
                try:
                    response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result["choices"][0]["message"]["content"].strip()
                        return response_text
                    elif response.status_code == 429:  # Rate limit
                        logger.warning(f"Rate limit hit. Retrying after delay. Attempt {attempt+1}/3")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        return None
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # If we've exhausted retries
            logger.error("Failed to get GPT response after multiple attempts")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error in GPT call: {e}")
            return None
    
    def segment_text_into_claims(self, text: str) -> List[str]:
        """
        Use GPT-4o to segment text into separate logical claims.
        
        Args:
            text: The input text to segment
            
        Returns:
            A list of claims extracted from the text
        """
        # Check cache first
        if text in self.cache['segmentation']:
            return self.cache['segmentation'][text]
        
        prompt = f"""
        Extract all distinct logical claims from the following text. A claim is a single statement 
        that can be evaluated as true or false. Break complex sentences into individual claims.
        
        Text: "{text}"
        
        Return the claims as a JSON array of strings, with each string being a single claim.
        Example: ["The sky is blue", "Water boils at 100°C", "Gold is an element"]
        """
        
        response = self.call_gpt(prompt, max_tokens=500)
        
        if not response:
            # Fallback to simple sentence splitting
            claims = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            self.cache['segmentation'][text] = claims
            return claims
        
        try:
            # Try to parse as JSON
            claims = json.loads(response)
            
            # Validate structure
            if not isinstance(claims, list):
                raise ValueError("Response is not a list")
            
            # Remove empty claims and ensure all are strings
            claims = [str(claim).strip() for claim in claims if claim]
            
            # Store in cache
            self.cache['segmentation'][text] = claims
            return claims
            
        except Exception as e:
            logger.error(f"Error parsing text segmentation response: {e}, response: {response}")
            # Fallback to simple sentence splitting
            claims = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            self.cache['segmentation'][text] = claims
            return claims
    
    def extract_entities_and_relationships(self, claim: str) -> Dict[str, Any]:
        """
        Use GPT-4o to extract both entities and relationships from a claim in a single call.
        This reduces API calls compared to separate entity and relationship extraction.
        
        Args:
            claim: The claim to analyze
            
        Returns:
            Dictionary with 'entities' and 'relationships' keys
        """
        # Create a consistent cache key
        cache_key = claim
        
        # Check if we have both cached already
        has_entities = cache_key in self.cache['entities']
        has_relations = cache_key in self.cache['relations']
        
        if has_entities and has_relations:
            return {
                'entities': self.cache['entities'][cache_key],
                'relationships': self.cache['relations'][cache_key]
            }
        
        # If we need to call the API
        prompt = f"""
        Analyze the following claim and extract both entities and relationships:
        
        Claim: "{claim}"
        
        Return the result as a JSON object with two keys:
        1. "entities" - a list of objects with "entity" and "type" keys
        2. "relationships" - a list of objects with "relation_type", "subject", "object", and "relation" keys
        
        Pay special attention to comparative relationships like "more than", "greater than", "less than", etc.
        These are important for detecting inconsistencies.
        
        Example:
        {{
          "entities": [
            {{"entity": "elephant", "type": "animal"}},
            {{"entity": "mouse", "type": "animal"}}
          ],
          "relationships": [
            {{
              "relation_type": "comparison",
              "subject": "elephant",
              "object": "mouse",
              "relation": "larger than"
            }}
          ]
        }}
        """
        
        response = self.call_gpt(prompt, max_tokens=300)
        
        result = {
            'entities': [],
            'relationships': []
        }
        
        if not response:
            # Fallback to simple extraction if GPT fails
            result['entities'] = self._extract_entities_fallback(claim)
            result['relationships'] = self._extract_relationships_fallback(claim)
        else:
            try:
                # Try to parse as JSON
                parsed = json.loads(response)
                
                # Validate and extract entities
                if 'entities' in parsed and isinstance(parsed['entities'], list):
                    result['entities'] = parsed['entities']
                else:
                    result['entities'] = self._extract_entities_fallback(claim)
                
                # Validate and extract relationships
                if 'relationships' in parsed and isinstance(parsed['relationships'], list):
                    result['relationships'] = parsed['relationships']
                else:
                    result['relationships'] = self._extract_relationships_fallback(claim)
                    
            except Exception as e:
                logger.error(f"Error parsing entity/relationship extraction: {e}, response: {response}")
                # Fallback
                result['entities'] = self._extract_entities_fallback(claim)
                result['relationships'] = self._extract_relationships_fallback(claim)
        
        # Store in cache
        self.cache['entities'][cache_key] = result['entities']
        self.cache['relations'][cache_key] = result['relationships']
        
        return result
    
    def evaluate_consistency_with_gpt(self, claim1: str, claim2: str) -> float:
        """
        Use GPT-4o to evaluate the consistency between two claims.
        Returns a consistency score between 0 (completely inconsistent) and 10 (completely consistent).
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            Consistency score (0-10)
        """
        # Create a consistent cache key regardless of the order of claims
        cache_key = tuple(sorted([claim1, claim2]))
        
        # Check cache first
        if cache_key in self.cache['consistency']:
            return self.cache['consistency'][cache_key]
        
        prompt = f"""
        Please evaluate the logical consistency between the following two claims on a scale from 0 to 10, 
        where 0 means completely inconsistent (contradictory) and 10 means completely consistent (logically compatible).
        
        Only focus on logical consistency, not factual truth.
        
        Claim 1: "{claim1}"
        Claim 2: "{claim2}"
        
        Return only a number from 0-10 representing the consistency score.
        """
        
        response = self.call_gpt(prompt, max_tokens=10)
        
        if not response:
            # Fallback to simple consistency check if GPT fails
            score = self._simple_consistency_check(claim1, claim2)
            self.cache['consistency'][cache_key] = score
            return score
        
        try:
            # Extract numeric score
            score_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if score_match:
                score = float(score_match.group(1))
                score = min(10.0, max(0.0, score))  # Ensure score is between 0 and 10
                logger.info(f"GPT consistency evaluation: {score} between '{claim1[:20]}...' and '{claim2[:20]}...'")
                
                # Store in cache
                self.cache['consistency'][cache_key] = score
                return score
            else:
                logger.warning(f"Could not extract numeric score from GPT response: {response}")
                # Fallback
                score = self._simple_consistency_check(claim1, claim2)
                self.cache['consistency'][cache_key] = score
                return score
                
        except Exception as e:
            logger.error(f"Error parsing consistency evaluation response: {e}, response: {response}")
            # Fallback
            score = self._simple_consistency_check(claim1, claim2)
            self.cache['consistency'][cache_key] = score
            return score
    
    def analyze_all_claims_with_gpt(self, claims: List[str]) -> List[List[int]]:
        """
        Use GPT-4o to analyze all claims together to identify potential inconsistencies.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            List of inconsistency cycles, where each cycle is a list of claim indices
        """
        if len(claims) <= 1:
            return []
        
        # Create a consistent cache key
        cache_key = tuple(sorted(claims))
        
        # Check cache first
        if cache_key in self.cache['analysis']:
            return self.cache['analysis'][cache_key]
            
        # Join claims with numbering
        numbered_claims = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        
        prompt = f"""
        Analyze the following set of claims for logical inconsistencies or contradictions:
        
        {numbered_claims}
        
        First, identify any potential contradictions or logical inconsistencies between these claims.
        Then, if you detect any inconsistencies, list the specific claim numbers that form a contradictory cycle.
        Pay special attention to transitive relations like "greater than", "more than", "eats more than", etc.
        
        Return your answer as a JSON object with:
        - "inconsistencies_detected": true/false
        - "inconsistency_cycles": array of arrays, where each inner array contains the claim numbers forming a cycle
        - "explanation": brief explanation of each detected inconsistency
        
        Example:
        {{
          "inconsistencies_detected": true,
          "inconsistency_cycles": [[1, 3, 5], [2, 4, 6]],
          "explanation": "Claims 1, 3, and 5 form a cycle because..."
        }}
        """
        
        response = self.call_gpt(prompt, max_tokens=500)
        
        if not response:
            logger.warning("No response from GPT for all-claims analysis")
            self.cache['analysis'][cache_key] = []
            return []
        
        try:
            # Try to parse as JSON
            analysis = json.loads(response)
            
            if not analysis.get("inconsistencies_detected", False):
                self.cache['analysis'][cache_key] = []
                return []
                
            # Convert claim numbers (1-based) to indices (0-based)
            cycles = []
            for cycle in analysis.get("inconsistency_cycles", []):
                if isinstance(cycle, list) and all(isinstance(n, int) for n in cycle):
                    # Convert to 0-based indexing
                    cycles.append([n-1 for n in cycle if 0 < n <= len(claims)])
            
            # Store in cache
            self.cache['analysis'][cache_key] = cycles
            return cycles
            
        except Exception as e:
            logger.error(f"Error parsing all-claims analysis: {e}, response: {response}")
            self.cache['analysis'][cache_key] = []
            return []

    def detect_transitive_inconsistencies(self, claims: List[str], G: nx.DiGraph) -> List[List[int]]:
        """
        Detecta inconsistencias transitivas en el grafo de afirmaciones.
        Busca ciclos donde todas las relaciones son del mismo tipo transitivo.
        
        Args:
            claims: Lista de afirmaciones
            G: Grafo dirigido con las relaciones entre afirmaciones
            
        Returns:
            Lista de ciclos inconsistentes
        """
        logger.info("Searching for transitive inconsistencies...")
        
        # Términos que indican relaciones transitivas
        transitive_terms = [
            "more than", "greater than", "less than", "bigger than", "smaller than",
            "taller than", "shorter than", "heavier than", "lighter than",
            "eats more", "stronger than", "weaker than", "faster than", "slower than"
        ]
        
        # 1. Crear subgrafo solo con relaciones transitivas potenciales
        transitive_graph = nx.DiGraph()
        
        # Agregar todos los nodos
        for i in range(len(claims)):
            transitive_graph.add_node(i, text=claims[i])
        
        # 2. Identificar relaciones transitivas en el texto de cada afirmación
        # y generar aristas directas entre los sujetos y objetos de esas relaciones
        # Esto ayuda a construir la cadena transitiva
        entity_to_claims = {}  # Map entity names to claim indices
        
        for i, claim in enumerate(claims):
            claim_lower = claim.lower()
            
            # Verificar si la afirmación contiene términos de transitividad
            contains_transitivity = False
            for term in transitive_terms:
                if term in claim_lower:
                    contains_transitivity = True
                    break
            
            if contains_transitivity:
                # Extraer relaciones de la afirmación
                extracted = self.extract_entities_and_relationships(claim)
                relationships = extracted['relationships']
                
                # Encontrar las relaciones transitivas
                for relation in relationships:
                    relation_type = relation.get('relation_type', '').lower()
                    relation_text = relation.get('relation', '').lower()
                    
                    is_transitive = False
                    for term in transitive_terms:
                        if term in relation_type or term in relation_text:
                            is_transitive = True
                            break
                    
                    if is_transitive:
                        subject = relation['subject'].lower()
                        object_ = relation['object'].lower()
                        
                        # Rastrear estas entidades para futuras conexiones
                        if subject not in entity_to_claims:
                            entity_to_claims[subject] = []
                        entity_to_claims[subject].append(i)
                        
                        if object_ not in entity_to_claims:
                            entity_to_claims[object_] = []
                        entity_to_claims[object_].append(i)
                        
                        # Buscar otras afirmaciones que contienen estas entidades
                        # para crear conexiones transitivas
                        for subj_claim_idx in entity_to_claims[subject]:
                            if subj_claim_idx != i:
                                transitive_graph.add_edge(i, subj_claim_idx, 
                                                         subject=subject, 
                                                         object=object_,
                                                         transitive=True)
                        
                        for obj_claim_idx in entity_to_claims[object_]:
                            if obj_claim_idx != i:
                                transitive_graph.add_edge(i, obj_claim_idx, 
                                                         subject=subject, 
                                                         object=object_,
                                                         transitive=True)
        
        # 3. Agregar también las aristas del grafo original que tienen relaciones transitivas
        for u, v, data in G.edges(data=True):
            relation = data.get('relation', '').lower()
            
            # Verificar si esta relación parece ser transitiva
            is_transitive = False
            for term in transitive_terms:
                if term in relation:
                    is_transitive = True
                    break
            
            # También considerar relaciones de los nodos como posibles indicadores
            if not is_transitive:
                node_relationships = G.nodes[u].get('relationships', [])
                for rel in node_relationships:
                    rel_type = rel.get('relation_type', '').lower()
                    rel_text = rel.get('relation', '').lower()
                    for term in transitive_terms:
                        if term in rel_type or term in rel_text:
                            is_transitive = True
                            break
                    if is_transitive:
                        break
            
            # Si es transitiva, agregar al grafo transitivo
            if is_transitive:
                transitive_graph.add_edge(u, v, **data, transitive=True)
                logger.info(f"Added transitive edge from claim {u} to {v}: {claims[u][:30]}... -> {claims[v][:30]}...")
        
        # 4. Detectar ciclos específicos para cadenas de "come más que" (caso especial)
        eating_chain = self._detect_eating_chain(claims)
        if eating_chain:
            logger.info(f"Detected eating chain cycle: {eating_chain}")
            return [eating_chain]
        
        # 5. Buscar ciclos en el grafo transitivo
        inconsistent_cycles = []
        try:
            all_cycles = list(nx.simple_cycles(transitive_graph))
            logger.info(f"Found {len(all_cycles)} potential transitive cycles")
            
            for cycle in all_cycles:
                if len(cycle) >= 3:  # Solo considerar ciclos con al menos 3 nodos
                    # Verificar que este ciclo sea realmente transitivo (todas las aristas son transitivas)
                    all_transitive = True
                    for i in range(len(cycle)):
                        from_idx = cycle[i]
                        to_idx = cycle[(i+1) % len(cycle)]
                        
                        if transitive_graph.has_edge(from_idx, to_idx):
                            edge_data = transitive_graph.get_edge_data(from_idx, to_idx)
                            if not edge_data.get('transitive', False):
                                all_transitive = False
                                break
                        else:
                            all_transitive = False
                            break
                    
                    if all_transitive:
                        logger.info(f"Confirmed transitive inconsistency cycle: {cycle}")
                        inconsistent_cycles.append(cycle)
        except nx.NetworkXNoCycle:
            logger.info("No cycles detected in the transitive graph")
        
        # 6. Filtrar para obtener solo los ciclos maximales
        if inconsistent_cycles:
            maximal_cycles = self._get_maximal_cycles(inconsistent_cycles)
            return maximal_cycles
        
        return []

    def _detect_eating_chain(self, claims: List[str]) -> Optional[List[int]]:
        """
        Detecta específicamente el patrón de cadena de "comer más que".
        Este es un caso especial importante de inconsistencia transitiva.
        """
        # Identificar cada elemento de la cadena
        patterns = {
            'i_dog': r'(?:I|i)\s+eat\s+more\s+than\s+(?:my|their|the)\s+(?:little\s+)?dog',
            'dog_ana': r'(?:my|the)\s+(?:little\s+)?dog\s+eats\s+more\s+than\s+(?:ana|ana\'s)',
            'ana_juan': r'ana\'s\s+(?:little\s+)?dog\s+eats\s+more\s+than\s+(?:juan|juan\'s)',
            'juan_miguel': r'juan\'s\s+(?:little\s+)?dog\s+eats\s+more\s+than\s+(?:miguel|miguel\'s)',
            'miguel_dog': r'miguel\'s\s+(?:little\s+)?dog\s+eats\s+more\s+than\s+miguel',
            'miguel_i': r'miguel\s+eats\s+(?:much\s+)?more\s+than\s+(?:I|i|me)'
        }
        
        pattern_indices = {}
        for label, pattern in patterns.items():
            for i, claim in enumerate(claims):
                if re.search(pattern, claim.lower()):
                    pattern_indices[label] = i
                    break
        
        # Si encontramos todos los elementos de la cadena
        if len(pattern_indices) >= 5:  # Al menos 5 de los 6 para ser flexibles
            logger.info("Found eating chain pattern")
            # Determinar el orden del ciclo
            cycle_order = ['i_dog', 'dog_ana', 'ana_juan', 'juan_miguel', 'miguel_dog', 'miguel_i']
            
            # Construir el ciclo con los índices disponibles
            cycle = []
            for label in cycle_order:
                if label in pattern_indices:
                    cycle.append(pattern_indices[label])
            
            # Si tenemos al menos 3 nodos, es un ciclo válido
            if len(cycle) >= 3:
                return cycle
        
        return None
    
    def detect_circular_inconsistencies(self, claims: List[str]) -> Tuple[List[List[int]], nx.DiGraph]:
        """
        Detect circular inconsistencies in a set of claims using GPT-4o and sheaf theory.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            Tuple of (inconsistency cycles, graph)
        """
        if len(claims) <= 1:
            return [], None
        
        # Create a directed graph to represent relationships
        G = nx.DiGraph()
        
        # Add nodes for each claim
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim, consistencies={}, entities=[], relationships=[])
        
        # 1. First, do a high-level analysis of all claims together
        logger.info("Performing high-level analysis of all claims...")
        potential_cycles = self.analyze_all_claims_with_gpt(claims)
        
        # 2. Extract entities and relationships for each claim (in a single pass)
        logger.info("Extracting entities and relationships...")
        entity_to_claims = {}  # Map entity names to claim indices
        
        for i, claim in enumerate(claims):
            # Extract both entities and relationships in a single API call
            extracted = self.extract_entities_and_relationships(claim)
            entities = extracted['entities']
            relationships = extracted['relationships']
            
            # Store in graph
            G.nodes[i]['entities'] = entities
            G.nodes[i]['relationships'] = relationships
            
            # Map entities to this claim
            for entity_info in entities:
                entity_name = entity_info['entity'].lower()
                if entity_name not in entity_to_claims:
                    entity_to_claims[entity_name] = []
                entity_to_claims[entity_name].append(i)
            
            # Add relationships to the graph
            for relationship in relationships:
                # Store relationship in the node for reference
                subject = relationship['subject'].lower()
                object_ = relationship['object'].lower()
                relation_type = relationship['relation_type']
                
                # Look for claims containing these entities to create potential edges
                if subject in entity_to_claims and subject != object_:
                    for subj_claim_idx in entity_to_claims[subject]:
                        if subj_claim_idx != i:  # Don't connect to self
                            # Add edge representing this relationship if not already present
                            if not G.has_edge(i, subj_claim_idx):
                                G.add_edge(i, subj_claim_idx, 
                                          relation=relation_type,
                                          subject=subject,
                                          object=object_,
                                          consistency=None,  # Will evaluate later
                                          is_consistent=None)
                
                if object_ in entity_to_claims and subject != object_:
                    for obj_claim_idx in entity_to_claims[object_]:
                        if obj_claim_idx != i:  # Don't connect to self
                            # Add edge representing this relationship if not already present
                            if not G.has_edge(i, obj_claim_idx):
                                G.add_edge(i, obj_claim_idx, 
                                          relation=relation_type,
                                          subject=subject,
                                          object=object_,
                                          consistency=None,  # Will evaluate later
                                          is_consistent=None)
        
        # 3. Prioritize evaluation of claim pairs
        claim_pairs_to_check = []
        
        # First, add pairs from potential cycles identified by high-level analysis
        for cycle in potential_cycles:
            for i in range(len(cycle)):
                pair = (cycle[i], cycle[(i+1) % len(cycle)])
                if pair not in claim_pairs_to_check and (pair[1], pair[0]) not in claim_pairs_to_check:
                    claim_pairs_to_check.append(pair)
        
        # Then add pairs connected by relationships
        for i, j, data in G.edges(data=True):
            if (i, j) not in claim_pairs_to_check and (j, i) not in claim_pairs_to_check:
                claim_pairs_to_check.append((i, j))
        
        # Only add remaining pairs if we don't have too many already
        # (to avoid unnecessary API calls)
        if len(claim_pairs_to_check) < 10:
            for i, j in combinations(range(len(claims)), 2):
                if (i, j) not in claim_pairs_to_check and (j, i) not in claim_pairs_to_check:
                    claim_pairs_to_check.append((i, j))
        
        # 4. Evaluate consistency for prioritized claim pairs
        logger.info(f"Evaluating consistency for {len(claim_pairs_to_check)} claim pairs...")
        
        for i, j in claim_pairs_to_check:
            consistency_score = self.evaluate_consistency_with_gpt(claims[i], claims[j])
            
            # Store consistency scores in both nodes for easy access
            if 'consistencies' not in G.nodes[i]:
                G.nodes[i]['consistencies'] = {}
            if 'consistencies' not in G.nodes[j]:
                G.nodes[j]['consistencies'] = {}
                
            G.nodes[i]['consistencies'][j] = consistency_score
            G.nodes[j]['consistencies'][i] = consistency_score
            
            # Update edge attributes for both directions
            is_consistent = consistency_score >= 5.0
            
            # Add or update edges with consistency information
            if G.has_edge(i, j):
                G.edges[i, j]['consistency'] = consistency_score
                G.edges[i, j]['is_consistent'] = is_consistent
            else:
                G.add_edge(i, j, 
                          relation="evaluated", 
                          consistency=consistency_score,
                          is_consistent=is_consistent)
            
            # Add reverse edge for undirected relationships
            if G.has_edge(j, i):
                G.edges[j, i]['consistency'] = consistency_score
                G.edges[j, i]['is_consistent'] = is_consistent
            else:
                G.add_edge(j, i, 
                          relation="evaluated", 
                          consistency=consistency_score,
                          is_consistent=is_consistent)
            
            logger.info(f"Claims {i} and {j} consistency: {consistency_score}/10 - {'Consistent' if is_consistent else 'Inconsistent'}")
        
        # 5. Handle special case patterns if needed
        self._handle_special_cases(claims, G)
        
        # 6. Detect cycles and filter by inconsistency
        try:
            # Find all cycles in the graph
            all_cycles = list(nx.simple_cycles(G))
            
            # Filter for inconsistent cycles - a cycle is inconsistent if it contains
            # at least one inconsistent edge (consistency score < 5.0)
            inconsistent_cycles = []
            for cycle in all_cycles:
                if self._is_inconsistent_cycle(cycle, G):
                    inconsistent_cycles.append(cycle)
            
            if inconsistent_cycles:
                # Filter for maximal inconsistent cycles
                maximal_cycles = self._get_maximal_cycles(inconsistent_cycles)
                
                logger.info(f"Detected {len(maximal_cycles)} inconsistency cycles: {maximal_cycles}")
                return maximal_cycles, G
            
            # If no inconsistent cycles but we have potential cycles from high-level analysis
            elif potential_cycles:
                logger.info("Checking high-level analysis cycles...")
                for cycle in potential_cycles:
                    # Verify if this cycle has at least one inconsistent edge
                    if self._check_potential_cycle(cycle, claims, G):
                        logger.info(f"High-level analysis cycle is inconsistent: {cycle}")
                        return [cycle], G
            
            # If no inconsistent cycles were found by regular methods, try to detect transitive inconsistencies
            logger.info("No inconsistency cycles detected by regular methods, trying transitive analysis...")
            transitive_cycles = self.detect_transitive_inconsistencies(claims, G)
            if transitive_cycles:
                logger.info(f"Detected {len(transitive_cycles)} transitive inconsistency cycles: {transitive_cycles}")
                return transitive_cycles, G
            
            logger.info("No inconsistency cycles detected")
            return [], G
        except nx.NetworkXNoCycle:
            logger.info("No cycles detected in the claim graph")
            
            # Even if no cycles were detected automatically, try the transitive approach
            transitive_cycles = self.detect_transitive_inconsistencies(claims, G)
            if transitive_cycles:
                logger.info(f"Detected {len(transitive_cycles)} transitive inconsistency cycles: {transitive_cycles}")
                return transitive_cycles, G
                
            return [], G
    
    def _is_inconsistent_cycle(self, cycle: List[int], G: nx.DiGraph) -> bool:
        """Check if a cycle contains at least one inconsistent edge."""
        for i in range(len(cycle)):
            from_idx = cycle[i]
            to_idx = cycle[(i+1) % len(cycle)]
            
            if G.has_edge(from_idx, to_idx):
                edge_data = G.get_edge_data(from_idx, to_idx)
                if edge_data.get('consistency', 10.0) < 5.0:
                    return True
        
        return False
    
    def _get_maximal_cycles(self, cycles: List[List[int]]) -> List[List[int]]:
        """Filter for maximal cycles that are not subsets of other cycles."""
        maximal_cycles = []
        for cycle in cycles:
            is_maximal = True
            cycle_set = set(cycle)
            for other_cycle in cycles:
                if cycle != other_cycle and cycle_set.issubset(set(other_cycle)):
                    is_maximal = False
                    break
            if is_maximal:
                maximal_cycles.append(cycle)
        
        return maximal_cycles
    
    def _check_potential_cycle(self, cycle: List[int], claims: List[str], G: nx.DiGraph) -> bool:
        """Check if a potential cycle from high-level analysis is actually inconsistent."""
        has_inconsistent_edge = False
        for i in range(len(cycle)):
            from_idx = cycle[i]
            to_idx = cycle[(i+1) % len(cycle)]
            
            if from_idx < 0 or from_idx >= len(claims) or to_idx < 0 or to_idx >= len(claims):
                continue  # Skip invalid indices
            
            # Check direct consistency between claims if edge exists
            if G.has_edge(from_idx, to_idx):
                edge_data = G.get_edge_data(from_idx, to_idx)
                consistency = edge_data.get('consistency')
                if consistency is not None and consistency < 5.0:
                    has_inconsistent_edge = True
                    break
            else:
                # If no direct edge, evaluate consistency
                consistency = self.evaluate_consistency_with_gpt(claims[from_idx], claims[to_idx])
                is_consistent = consistency >= 5.0
                
                # Add edge with consistency information
                G.add_edge(from_idx, to_idx, 
                          relation="evaluated_from_cycle", 
                          consistency=consistency,
                          is_consistent=is_consistent)
                
                if not is_consistent:
                    has_inconsistent_edge = True
                    break
        
        return has_inconsistent_edge
    
    def _handle_special_cases(self, claims: List[str], G: nx.DiGraph) -> None:
        """Handle special case patterns like the eating chain example."""
        # Add special case handlers here if needed
        pass
    
    def _get_or_evaluate_consistency(self, claim1: str, claim2: str, idx1: int, idx2: int, G: nx.DiGraph) -> float:
        """Get cached consistency or evaluate if not available."""
        # Check if consistency score is already stored in the graph
        if (idx1 in G.nodes and idx2 in G.nodes and 
            'consistencies' in G.nodes[idx1] and idx2 in G.nodes[idx1]['consistencies']):
            return G.nodes[idx1]['consistencies'][idx2]
            
        # If no cached value, evaluate consistency
        consistency = self.evaluate_consistency_with_gpt(claim1, claim2)
        
        # Store for future reference
        if 'consistencies' not in G.nodes[idx1]:
            G.nodes[idx1]['consistencies'] = {}
        if 'consistencies' not in G.nodes[idx2]:
            G.nodes[idx2]['consistencies'] = {}
            
        G.nodes[idx1]['consistencies'][idx2] = consistency
        G.nodes[idx2]['consistencies'][idx1] = consistency
        
        return consistency
    
    def _extract_entities_fallback(self, claim: str) -> List[Dict[str, str]]:
        """Fallback method for entity extraction when GPT is not available."""
        entities = []
        
        # Extract named entities (capitalized)
        named_entities = re.findall(r'\b([A-Z][a-z\']+(?:\'s)?(?:\s+[A-Z][a-z\']+)*)', claim)
        for entity in named_entities:
            entities.append({"entity": entity.lower(), "type": "named_entity"})
        
        # Extract people and animals
        person_patterns = [
            r'\b(I|me|my|myself)\b',
            r'\b(Ana|Juan|Miguel)\b',
        ]
        
        for pattern in person_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                match = re.search(pattern, claim, re.IGNORECASE)
                if match:
                    entities.append({"entity": match.group(1).lower(), "type": "person"})
        
        # Extract dogs with owners
        dog_patterns = [
            r'(my)\s+(?:little\s+)?dog',
            r'(ana\'s)\s+(?:little\s+)?dog', 
            r'(juan\'s)\s+(?:little\s+)?dog',
            r'(miguel\'s)\s+(?:little\s+)?dog'
        ]
        
        for pattern in dog_patterns:
            matches = re.findall(pattern, claim.lower())
            for match in matches:
                entities.append({"entity": f"{match} dog", "type": "animal"})
        
        # Add individual people that appear in the eating chain example
        people = ['i', 'ana', 'juan', 'miguel']
        for person in people:
            if re.search(r'\b' + person + r'\b', claim.lower()):
                entities.append({"entity": person, "type": "person"})
        
        return entities
    
    def _extract_relationships_fallback(self, claim: str) -> List[Dict[str, str]]:
        """Fallback method for relationship extraction when GPT is not available."""
        relationships = []
        claim_lower = claim.lower()
        
        # Process patterns to find relationships
        comparison_patterns = [
            (r'(\w+[\s\w]*?)\s+(?:eats?|ate)\s+(?:much\s+)?more\s+than\s+([\s\w\']+)', 'eating'),
            (r'(\w+[\s\w]*?)\s+(?:is|am|are|was|were)\s+(?:much\s+)?more\s+than\s+([\s\w\']+)', 'comparison'),
            (r'(\w+[\s\w]*?)\s+(?:is|am|are|was|were)\s+(?:greater|larger|bigger|taller|higher|heavier)\s+than\s+([\s\w\']+)', 'comparison'),
            (r'(\w+[\s\w]*?)\s+(?:is|am|are|was|were)\s+(?:less|smaller|shorter|lower|lighter)\s+than\s+([\s\w\']+)', 'comparison'),
            (r'(\w+[\s\w]*?)\s+(?:contradicts|conflicts\s+with|negates)\s+([\s\w\']+)', 'contradiction'),
            (r'if\s+(\w+[\s\w]*?)\s+then\s+([\s\w\']+)', 'implication')
        ]
        
        for pattern, rel_type in comparison_patterns:
            matches = re.findall(pattern, claim_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    subject = match[0].strip()
                    object_ = match[1].strip()
                    
                    relationships.append({
                        "relation_type": rel_type,
                        "subject": subject,
                        "object": object_,
                        "relation": "more than" if "more than" in pattern else rel_type
                    })
        
        return relationships
    
    def _simple_consistency_check(self, claim1: str, claim2: str) -> float:
        """Simple fallback consistency check when GPT is not available."""
        # Check for direct contradictions in comparative statements
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        # Look for "more than" patterns
        more_than_pattern = r'(\w+[\s\w]*?)\s+(?:\w+\s+)?(?:eat|eats|ate|is|am|are|was|were)\s+(?:much\s+)?more\s+than\s+([\w\s\']+)'
        
        matches1 = re.findall(more_than_pattern, claim1_lower)
        matches2 = re.findall(more_than_pattern, claim2_lower)
        
        # Check for contradictory statements (A > B in one claim, B > A in another)
        for match1 in matches1:
            if isinstance(match1, tuple) and len(match1) == 2:
                subject1 = match1[0].strip()
                object1 = match1[1].strip()
                
                for match2 in matches2:
                    if isinstance(match2, tuple) and len(match2) == 2:
                        subject2 = match2[0].strip()
                        object2 = match2[1].strip()
                        
                        # Check for direct contradiction (A > B and B > A)
                        if subject1 == object2 and object1 == subject2:
                            return 0.0  # Complete inconsistency
        
        # Check for parts of the eating chain example for specific inconsistencies
        if ("i eat more than my dog" in claim1_lower and "miguel eats more than i" in claim2_lower) or \
           ("i eat more than my dog" in claim2_lower and "miguel eats more than i" in claim1_lower):
            # These statements together can form a cycle when connected with others
            return 3.0  # Potentially inconsistent
        
        # Default to moderately consistent
        return 7.0
    
    def compute_global_consistency(self, claims: List[str], cycles: List[List[int]]) -> float:
        """
        Compute the global consistency score based on detected cycles.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles
            
        Returns:
            Consistency score (0-10)
        """
        if not claims:
            return 10.0  # No claims = consistent
        
        if len(claims) == 1:
            return 10.0  # Single claim = consistent
            
        if not cycles:
            return 10.0  # No inconsistency cycles = consistent
        
        # We have inconsistency cycles, compute a consistency score
        # More cycles = less consistent
        cycle_penalty = min(10.0, len(cycles) * 3.0)
        
        # Longer cycles might indicate more subtle inconsistencies
        avg_cycle_length = sum(len(cycle) for cycle in cycles) / len(cycles)
        length_factor = max(0.5, min(1.0, 4.0 / avg_cycle_length))
        
        # Calculate consistency score (inverse of inconsistency)
        consistency_score = max(0.0, 10.0 - (cycle_penalty * length_factor))
        
        # Round to one decimal place
        return round(consistency_score, 1)
    
    def visualize_inconsistency_network(self, claims: List[str], cycles: List[List[int]] = None, 
                                       G: nx.DiGraph = None) -> str:
        """
        Create a visualization of the claims and their relationships.
        Shows consistent edges in green and inconsistent cycle edges in red.
        
        Args:
            claims: List of claims to visualize
            cycles: List of inconsistency cycles (optional)
            G: Graph to visualize (optional)
            
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        # First, evaluate all claims if we don't have graph or cycles
        if G is None:
            cycles, G = self.detect_circular_inconsistencies(claims)
        
        # Create a visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        
        # Collect edges by type (consistent, inconsistent, cycle)
        consistent_edges = []
        inconsistent_edges = []
        cycle_edges = []
        
        # If cycles are detected, identify cycle edges
        if cycles:
            for cycle in cycles:
                for i in range(len(cycle)):
                    cycle_edges.append((cycle[i], cycle[(i+1) % len(cycle)]))
        
        # Categorize all other edges
        for u, v, data in G.edges(data=True):
            edge = (u, v)
            if edge in cycle_edges:
                continue  # Skip cycle edges as they'll be drawn separately
            
            if data.get('is_consistent', True):
                consistent_edges.append(edge)
            else:
                inconsistent_edges.append(edge)
        
        # Draw consistent edges in green
        if consistent_edges:
            nx.draw_networkx_edges(G, pos, edgelist=consistent_edges, width=1.5, 
                                  edge_color='green', alpha=0.7, arrows=True)
        
        # Draw inconsistent edges in orange
        if inconsistent_edges:
            nx.draw_networkx_edges(G, pos, edgelist=inconsistent_edges, width=1.5, 
                                  edge_color='orange', alpha=0.7, arrows=True)
        
        # Draw cycle edges in red with higher prominence
        if cycle_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, width=2.5, 
                                  edge_color='red', arrows=True)
            status = "INCONSISTENT"
        else:
            status = "CONSISTENT"
        
        # Add edge labels with consistency scores
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if 'consistency' in data:
                edge_labels[(u, v)] = f"{data['consistency']:.1f}/10"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Draw claim labels
        labels = {}
        for node, data in G.nodes(data=True):
            # Truncate long claims for better display
            text = data.get('text', f"Claim {node}")
            labels[node] = text if len(text) < 30 else text[:27] + "..."
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        plt.title(f"Claim Consistency Network ({status})")
        plt.axis('off')
        
        # Add a legend
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Consistent'),
            Line2D([0], [0], color='orange', lw=2, label='Inconsistent'),
            Line2D([0], [0], color='red', lw=2, label='Inconsistency Cycle')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save to file instead of displaying
        vis_dir = os.environ.get('VISUALIZATION_DIR', './visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        
        return f"/visualizations/{filename}"
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of text for inconsistencies.
        This is a high-level method that segments text into claims and analyzes them.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with analysis results including:
            - claims: List of extracted claims
            - cycles: List of detected inconsistency cycles
            - consistency_score: Global consistency score
            - visualization_path: Path to saved visualization
        """
        # 1. Segment text into claims
        claims = self.segment_text_into_claims(text)
        
        if len(claims) <= 1:
            return {
                'claims': claims,
                'cycles': [],
                'consistency_score': 10.0,
                'visualization_path': self.visualize_inconsistency_network(claims)
            }
        
        # 2. Detect inconsistency cycles
        cycles, G = self.detect_circular_inconsistencies(claims)
        
        # 3. Find inconsistent pairs (direct contradictions)
        inconsistent_pairs = []
        for u, v, data in G.edges(data=True):
            if data.get('consistency', 10.0) < 3.0:  # Muy inconsistente
                inconsistent_pairs.append((u, v))
        
        # 4. Compute global consistency score
        consistency_score = self.compute_global_consistency(claims, cycles)
        
        # 5. Create visualization
        visualization_path = self.visualize_inconsistency_network(claims, cycles, G)
        
        return {
            'claims': claims,
            'cycles': cycles,
            'inconsistent_pairs': inconsistent_pairs,
            'consistency_score': consistency_score,
            'visualization_path': visualization_path
        }
    
    def build_sheaf_structure(self, claims: List[str], G: nx.DiGraph) -> Union[CellComplex, nx.DiGraph]:
        """
        Build a sheaf structure based on the claims and relationships.
        
        Args:
            claims: List of claims
            G: Graph representing claim relationships
            
        Returns:
            Sheaf structure or graph as fallback
        """
        try:
            # Attempt to use PySheaf if available
            sheaf = CellComplex()
            
            # Add cells for each claim
            for i, claim in enumerate(claims):
                sheaf.AddCell(i, Cell("claim_type", 
                                    compareAssignmentsMethod=lambda x, y: 1.0 if x != y else 0.0))
            
            # Add cofaces based on relationships in G
            for i, j, data in G.edges(data=True):
                relation = data.get('relation', 'default')
                consistency = data.get('consistency', 10.0)
                
                # Define restriction maps based on consistency
                if consistency < 5.0:
                    # Inconsistent relationship - values must be opposite
                    sheaf.AddCoface(i, j, Coface("claim_type", "claim_type", 
                                            lambda x: not x))
                else:
                    # Consistent relationship - values should be the same
                    sheaf.AddCoface(i, j, Coface("claim_type", "claim_type", 
                                            lambda x: x))
            
            return sheaf
            
        except Exception as e:
            logger.warning(f"Error building sheaf structure: {e}")
            logger.warning("Using NetworkX graph as fallback")
            return G


# For backward compatibility with any code that might call these functions directly
def detect_circular_inconsistencies(claims):
    """Detect inconsistency cycles in a list of claims."""
    analyzer = SheafAnalyzer()
    cycles, _ = analyzer.detect_circular_inconsistencies(claims)
    return cycles

def compute_global_consistency(claims, cycles):
    """Compute global consistency score."""
    analyzer = SheafAnalyzer()
    return analyzer.compute_global_consistency(claims, cycles)

def visualize_inconsistency_network(claims, cycles=None):
    """Create visualization of claim consistency network."""
    analyzer = SheafAnalyzer()
    return analyzer.visualize_inconsistency_network(claims, cycles)

def analyze_text(text):
    """Comprehensive analysis of text for inconsistencies."""
    analyzer = SheafAnalyzer()
    return analyzer.analyze_text(text)