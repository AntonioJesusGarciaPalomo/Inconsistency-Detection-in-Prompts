"""
This module handles graph-based calculations for inconsistency detection.
Uses NetworkX for graph representations and cycle detection.
"""
import logging
import networkx as nx
import re
from typing import List, Dict, Tuple, Any, Optional, Set, Union
import matplotlib.pyplot as plt
import os
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """
    Class for analyzing logical consistency using graph theory.
    """
    
    def __init__(self):
        """Initialize the graph analyzer."""
        # Cache for results
        self.cache = {
            'consistency': {},    # Consistency evaluations between pairs of claims
            'cycles': {},         # Detected inconsistency cycles
            'analysis': {}        # Overall analysis of claim sets
        }
        
        # Define patterns for detecting comparative relationships
        self.comparison_patterns = [
            # Pattern group 1: Explicit comparisons with "than"
            (r'(\b[\w\s]+\b)(?:\s+(?:is|are|was|were))?\s+(more|greater|higher|larger|stronger|faster|older|heavier)\s+than\s+(\b[\w\s]+\b)', 'greater_than'),
            
            # Pattern group 2: Implicit comparisons with "as...as"
            (r'(\b[\w\s]+\b)\s+as\s+(?:\w+\s+)?as\s+(\b[\w\s]+\b)', 'equal_to'),
            
            # Pattern group 3: Verb-based comparisons
            (r'(\b[\w\s]+\b)\s+(?:eats|runs|jumps|works|pays)\s+(?:much\s+)?more\s+than\s+(\b[\w\s]+\b)', 'action_greater'),
            
            # Pattern group 4: Negative comparisons
            (r'(\b[\w\s]+\b)\s+(?:is|are)\s+not\s+as\s+(\w+)\s+as\s+(\b[\w\s]+\b)', 'not_as'),
            
            # Pattern group 5: Quantitative comparisons
            (r'(\b[\w\s]+\b)\s+(?:has|have)\s+(\d+|\w+)\s+more\s+(\w+)\s+than\s+(\b[\w\s]+\b)', 'quantitative_more')
        ]
    
    def detect_circular_inconsistencies(self, claims: List[str]) -> Tuple[List[List[int]], nx.DiGraph]:
        """
        Detect circular inconsistencies in a set of claims.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            Tuple of (inconsistency cycles, graph)
        """
        if len(claims) < 2:
            return [], nx.DiGraph()

        G = nx.DiGraph()
        entities, relationships = self.extract_entities_and_relationships(claims)
        
        # Add nodes with metadata
        for idx, claim in enumerate(claims):
            G.add_node(idx, text=claim, entities=self._extract_claim_entities(claim))
        
        # Add edges with relationship metadata
        for rel in relationships:
            entity_a, relation, entity_b, claim_idx = rel
            G.add_edge(claim_idx, f"{entity_a}-{entity_b}",
                      relation=relation,
                      entity_a=entity_a,
                      entity_b=entity_b,
                      consistency=7.0 if 'greater' in relation else 3.0)

        # Detect cycles using improved algorithm
        cycles = self._detect_logical_cycles(G)
        return cycles, G
    
    def extract_entities_and_relationships(self, claims: List[str]) -> Tuple[Set[str], List[Tuple]]:
        """
        Extract entities and relationships from claims, focusing on comparative relationships.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            Tuple of (set of entities, list of relationships)
            Each relationship is (entity1, relation, entity2, claim_index)
        """
        entities = set()
        relationships = []
        
        for claim_idx, claim in enumerate(claims):
            for pattern, rel_type in self.comparison_patterns:
                matches = re.findall(pattern, claim, re.IGNORECASE)
                for match in matches:
                    # Handle different pattern structures
                    if rel_type == 'equal_to':
                        a, b = match[0].strip().lower(), match[1].strip().lower()
                        relationships.append((a, rel_type, b, claim_idx))
                        entities.update({a, b})
                    elif rel_type == 'not_as':
                        a, _, b = match[0].strip().lower(), match[1], match[2].strip().lower()
                        relationships.append((b, 'greater_than', a, claim_idx))  # Reverse for negation
                        entities.update({a, b})
                    else:
                        if isinstance(match, tuple):
                            if len(match) >= 3:
                                a, _, b = match[0].strip().lower(), match[1], match[2].strip().lower()
                                relationships.append((a, rel_type, b, claim_idx))
                                entities.update({a, b})
        
        return entities, relationships
    
    def _detect_logical_cycles(self, G: nx.DiGraph) -> List[List[int]]:
        """
        Improved cycle detection with claim sequence tracking.
        
        Args:
            G: Directed graph with claims as nodes
            
        Returns:
            List of cycles (each cycle is a list of claim indices)
        """
        cycles = []
        visited_claims = set()
        
        try:
            for cycle in nx.all_simple_cycles(G):
                claim_sequence = []
                for node in cycle:
                    if isinstance(node, int):  # Filter actual claim nodes
                        claim_sequence.append(node)
                
                # Validate cycle has at least 3 claims and forms a logical loop
                if len(claim_sequence) >= 3 and self._is_valid_cycle(G, claim_sequence):
                    cycles.append(claim_sequence)
        except nx.NetworkXNoCycle:
            # No cycles found
            return []
        
        return cycles
    
    def _is_valid_cycle(self, G: nx.DiGraph, sequence: List[int]) -> bool:
        """
        Validate if claim sequence forms a logical inconsistency.
        
        Args:
            G: Directed graph
            sequence: List of claim indices
            
        Returns:
            True if the sequence forms a valid inconsistency cycle
        """
        for i in range(len(sequence)):
            current = sequence[i]
            next_claim = sequence[(i+1) % len(sequence)]
            
            # Check if there's a logical contradiction between consecutive claims
            if not self._claims_are_contradictory(G, current, next_claim):
                return False
        
        return True
    
    def _claims_are_contradictory(self, G: nx.DiGraph, a: int, b: int) -> bool:
        """
        Check if two claims create a contradiction.
        
        Args:
            G: Directed graph
            a: First claim index
            b: Second claim index
            
        Returns:
            True if the claims potentially contradict each other
        """
        # Safety check for node existence
        if a not in G.nodes or b not in G.nodes:
            return False
            
        # Get entities from nodes if available, otherwise use empty set
        a_entities = G.nodes[a].get('entities', set())
        b_entities = G.nodes[b].get('entities', set())
        
        if isinstance(a_entities, list):
            a_entities = set(a_entities)
        if isinstance(b_entities, list):
            b_entities = set(b_entities)
        
        # Check for entity overlap and relationship inversion
        common_entities = a_entities & b_entities
        if len(common_entities) < 2:
            return False
        
        # Check relationship directions
        a_relations = [data.get('relation', '') for _, _, data in G.out_edges(a, data=True)]
        b_relations = [data.get('relation', '') for _, _, data in G.out_edges(b, data=True)]
        
        return any(r1 != r2 for r1 in a_relations for r2 in b_relations if r1 and r2)
    
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
    
    def visualize_inconsistency_network(self, claims: List[str], cycles: List[List[int]] = None, G: nx.DiGraph = None) -> str:
        """
        Create a visualization of the claims and their relationships.
        
        Args:
            claims: List of claims to visualize
            cycles: List of inconsistency cycles (optional)
            G: Graph to visualize (optional)
            
        Returns:
            Path to saved visualization
        """
        if cycles is None:
            cycles = []
            
        # Create a graph if none provided
        if G is None:
            # If we don't have a graph, try to create one from the claims and cycles
            G = nx.DiGraph()
            
            # Add nodes for each claim
            for i, claim in enumerate(claims):
                G.add_node(i, text=claim)
                
            # Add edges for cycles
            for cycle in cycles:
                for i in range(len(cycle)):
                    from_idx = cycle[i]
                    to_idx = cycle[(i+1) % len(cycle)]
                    
                    # Add edge (last edge is inconsistent)
                    if i == len(cycle) - 1:
                        G.add_edge(from_idx, to_idx, consistency=3.0, is_consistent=False)
                    else:
                        G.add_edge(from_idx, to_idx, consistency=7.0, is_consistent=True)
        
        plt.figure(figsize=(14, 9))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Draw nodes with different colors
        # Draw claim nodes
        claim_nodes = [n for n in G.nodes if isinstance(n, int)]
        if claim_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=claim_nodes,
                                  node_color='lightblue',
                                  node_size=800,
                                  label='Claims')
        
        # Draw entity nodes
        entity_nodes = [n for n in G.nodes if not isinstance(n, int)]
        if entity_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=entity_nodes,
                                  node_color='lightgreen',
                                  node_size=500,
                                  label='Entities')

        # Draw edges with different styles
        consistent_edges = []
        inconsistent_edges = []
        
        # Collect cycle edges
        cycle_edges = []
        for cycle in cycles:
            for i in range(len(cycle)):
                cycle_edges.append((cycle[i], cycle[(i+1) % len(cycle)]))
        
        # Categorize edges
        for u, v, data in G.edges(data=True):
            edge = (u, v)
            # Prioritize cycle edges
            if edge in cycle_edges:
                inconsistent_edges.append(edge)
            # Otherwise check consistency data
            elif data.get('is_consistent', True) or data.get('consistency', 7.0) >= 5.0:
                consistent_edges.append(edge)
            else:
                inconsistent_edges.append(edge)
        
        # Draw consistent edges in green
        if consistent_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=consistent_edges,
                                  edge_color='green',
                                  arrows=True,
                                  alpha=0.6)
        
        # Draw inconsistent edges in red
        if inconsistent_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=inconsistent_edges,
                                  edge_color='red',
                                  arrows=True,
                                  style='dashed',
                                  width=2)

        # Add labels - short label for nodes, full text in annotation
        claim_labels = {}
        for i, claim in enumerate(claims):
            if i in G.nodes:  # Only add labels for nodes that exist
                short_text = f"Claim {i}"
                claim_labels[i] = short_text
                
                # Add full text as annotation
                plt.annotate(f"{i}: {claim[:50]}...", 
                           xy=pos[i],
                           xytext=(0, -20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                           size=8,
                           ha='center')
        
        # Add entity labels if they exist
        entity_labels = {n: n for n in entity_nodes}
        
        # Draw labels
        if claim_labels:
            nx.draw_networkx_labels(G, pos, labels=claim_labels, font_size=9)
        if entity_labels:
            nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=8, font_color='darkgreen')

        plt.title("Inconsistency Detection Network", fontsize=14)
        plt.legend(loc='upper right')
        plt.axis('off')  # Turn off axis
        
        # Save visualization
        vis_dir = os.environ.get('VISUALIZATION_DIR', './visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"inconsistency_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return path in format expected by detector
        return f"/visualizations/{os.path.basename(filepath)}"
    
    def _extract_claim_entities(self, claim: str) -> Set[str]:
        """
        Extract entities mentioned in a claim.
        
        Args:
            claim: Claim text to analyze
            
        Returns:
            Set of entities mentioned in the claim
        """
        entities = set()
        for pattern, _ in self.comparison_patterns:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Add all non-empty string parts of the match
                    entities.update([m.strip().lower() for m in match if isinstance(m, str) and m.strip()])
                else:
                    entities.add(match.strip().lower())
        return entities