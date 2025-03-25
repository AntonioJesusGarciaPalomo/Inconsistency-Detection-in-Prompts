"""
This module handles graph-based calculations for inconsistency detection.
Uses NetworkX for graph representations and cycle detection.
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
from itertools import combinations
from typing import List, Dict, Tuple, Any, Optional, Set, Union

# Configure logger
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
    
    def detect_circular_inconsistencies(self, claims: List[str]) -> Tuple[List[List[int]], nx.DiGraph]:
        """
        Detect circular inconsistencies in a set of claims.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            Tuple of (inconsistency cycles, graph)
        """
        if len(claims) <= 1:
            return [], None
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each claim
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim)
        
        # First, detect transitive inconsistencies
        transitive_cycles = self.detect_transitive_inconsistencies(claims)
        if transitive_cycles:
            logger.info(f"Detected transitive inconsistency cycles: {transitive_cycles}")
            
            # Add these to the graph
            for cycle in transitive_cycles:
                for i in range(len(cycle)):
                    from_idx = cycle[i]
                    to_idx = cycle[(i+1) % len(cycle)]
                    # Add direct edges between claims in the cycle
                    G.add_edge(from_idx, to_idx, consistency=7.0, is_consistent=True)
                
                # Mark the last edge that completes the cycle as inconsistent
                G.add_edge(cycle[-1], cycle[0], consistency=3.0, is_consistent=False)
            
            return transitive_cycles, G
        
        return [], G
    
    def extract_entities_and_relationships(self, claims: List[str]) -> Tuple[Set[str], List[Tuple[str, str, str, int]]]:
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
        
        # Common comparative markers
        comparative_markers = [
            # "More than" relationships
            (r'(\S+(?:\s+\S+){0,5})\s+(?:is|are|was|were)\s+(?:\S+\s+){0,3}(more|greater|higher|bigger|larger|stronger|faster|older|taller|heavier)\s+than\s+(\S+(?:\s+\S+){0,5})', 'greater_than'),
            
            # "Less than" relationships
            (r'(\S+(?:\s+\S+){0,5})\s+(?:is|are|was|were)\s+(?:\S+\s+){0,3}(less|smaller|lower|weaker|slower|younger|shorter|lighter)\s+than\s+(\S+(?:\s+\S+){0,5})', 'less_than'),
            
            # Verb + "more than"
            (r'(\S+(?:\s+\S+){0,5})\s+((?:eat|eats|run|runs|jump|jumps|think|thinks|work|works|play|plays|pay|pays|spend|spends|earn|earns)\s+(?:much\s+)?more)\s+than\s+(\S+(?:\s+\S+){0,5})', 'does_more_than')
        ]
        
        # Process each claim
        for i, claim in enumerate(claims):
            for pattern, relation_type in comparative_markers:
                matches = re.finditer(pattern, claim, re.IGNORECASE)
                
                for match in matches:
                    if relation_type in ['greater_than', 'does_more_than']:
                        # For "more than" patterns
                        if len(match.groups()) >= 3:
                            entity1 = match.group(1).strip().lower()
                            relation = match.group(2).strip().lower()
                            entity2 = match.group(3).strip().lower()
                        else:
                            entity1 = match.group(1).strip().lower()
                            relation = "more_than"
                            entity2 = match.group(3).strip().lower()
                    elif relation_type == 'less_than':
                        # For "less than" patterns (reverse the direction for consistency)
                        entity2 = match.group(1).strip().lower()  # Note the reversal
                        relation = "more_than"  # Normalize to "more_than"
                        entity1 = match.group(3).strip().lower()  # Note the reversal
                    
                    # Clean up entities (remove trailing punctuation, etc.)
                    entity1 = re.sub(r'[.,;:!?]$', '', entity1)
                    entity2 = re.sub(r'[.,;:!?]$', '', entity2)
                    
                    # Add to collections
                    entities.add(entity1)
                    entities.add(entity2)
                    relationships.append((entity1, relation, entity2, i))
        
        return entities, relationships
    
    def build_transitive_graph(self, entities: Set[str], relationships: List[Tuple[str, str, str, int]]) -> nx.DiGraph:
        """
        Build a directed graph representing transitive relationships between entities.
        
        Args:
            entities: Set of entities
            relationships: List of relationships (entity1, relation, entity2, claim_index)
            
        Returns:
            Directed graph with entities as nodes and relationships as edges
        """
        G = nx.DiGraph()
        
        # Add nodes for entities
        for entity in entities:
            G.add_node(entity)
        
        # Add edges for relationships
        for entity1, relation, entity2, claim_idx in relationships:
            if relation in ["more_than", "greater_than", "does_more_than"]:
                # Direction is from "more" to "less"
                G.add_edge(entity1, entity2, relation=relation, claim_idx=claim_idx)
        
        return G
    
    def find_transitive_cycles(self, G: nx.DiGraph) -> List[List[int]]:
        """
        Find cycles in the transitive relationship graph and convert to claim cycles.
        
        Args:
            G: Directed graph of transitive relationships
            
        Returns:
            List of claim index cycles
        """
        # Find all cycles in the graph
        try:
            entity_cycles = list(nx.simple_cycles(G))
        except nx.NetworkXNoCycle:
            return []
        
        # Convert entity cycles to claim index cycles
        claim_cycles = []
        for cycle in entity_cycles:
            if len(cycle) < 2:
                continue
                
            # Extract claim indices from the cycle
            claims_in_cycle = []
            for i in range(len(cycle)):
                entity1 = cycle[i]
                entity2 = cycle[(i+1) % len(cycle)]
                
                if G.has_edge(entity1, entity2):
                    claim_idx = G.edges[entity1, entity2].get('claim_idx')
                    if claim_idx is not None:
                        claims_in_cycle.append(claim_idx)
            
            # Only include cycles with unique claims
            if len(claims_in_cycle) > 1 and len(set(claims_in_cycle)) == len(claims_in_cycle):
                claim_cycles.append(claims_in_cycle)
        
        return claim_cycles
    
    def detect_transitive_inconsistencies(self, claims: List[str]) -> List[List[int]]:
        """
        Detect inconsistencies in transitive relationships.
        This identifies circular chains of comparative relationships that create logical contradictions.
        
        Args:
            claims: List of claims to analyze
        
        Returns:
            List of inconsistency cycles (each cycle is a list of claim indices)
        """
        if len(claims) <= 2:
            return []
        
        # Extract entities and relationships
        entities, relationships = self.extract_entities_and_relationships(claims)
        if not relationships:
            return []
            
        # Build the transitive graph
        G = self.build_transitive_graph(entities, relationships)
        
        # Find cycles in the transitive graph
        cycles = self.find_transitive_cycles(G)
        
        # Sort cycles by length (longer cycles are usually more significant)
        cycles.sort(key=len, reverse=True)
        
        return cycles
    
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
            G = nx.DiGraph()
            # Add nodes for each claim
            for i, claim in enumerate(claims):
                G.add_node(i, text=claim)
            
            # Add edges for cycles
            for cycle in cycles:
                for i in range(len(cycle)):
                    from_idx = cycle[i]
                    to_idx = cycle[(i+1) % len(cycle)]
                    
                    # Last edge is inconsistent, others are consistent
                    if i == len(cycle) - 1:
                        G.add_edge(from_idx, to_idx, consistency=3.0, is_consistent=False)
                    else:
                        G.add_edge(from_idx, to_idx, consistency=7.0, is_consistent=True)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Collect edges by type
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
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        
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
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Consistent'),
            Line2D([0], [0], color='orange', lw=2, label='Inconsistent'),
            Line2D([0], [0], color='red', lw=2, label='Inconsistency Cycle')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save to file
        vis_dir = os.environ.get('VISUALIZATION_DIR', './visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        
        return f"/visualizations/{filename}"
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for logical inconsistencies.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Simple sentence segmentation to extract claims
        claims = []
        for sentence in re.split(r'[.!?]\s+', text):
            if sentence.strip():
                claims.append(sentence.strip() + ".")
        
        # Detect inconsistency cycles
        cycles, G = self.detect_circular_inconsistencies(claims)
        
        # Calculate consistency score
        consistency_score = self.compute_global_consistency(claims, cycles)
        
        # Create visualization
        visualization_path = self.visualize_inconsistency_network(claims, cycles, G)
        
        return {
            'claims': claims,
            'cycles': cycles,
            'consistency_score': consistency_score,
            'visualization_path': visualization_path
        }


# For backward compatibility
def detect_circular_inconsistencies(claims):
    """Detect inconsistency cycles in a list of claims."""
    analyzer = GraphAnalyzer()
    cycles, _ = analyzer.detect_circular_inconsistencies(claims)
    return cycles

def compute_global_consistency(claims, cycles):
    """Compute global consistency score."""
    analyzer = GraphAnalyzer()
    return analyzer.compute_global_consistency(claims, cycles)

def visualize_inconsistency_network(claims, cycles=None):
    """Create visualization of claim consistency network."""
    analyzer = GraphAnalyzer()
    return analyzer.visualize_inconsistency_network(claims, cycles)

def analyze_text(text):
    """Analyze text for logical inconsistencies."""
    analyzer = GraphAnalyzer()
    return analyzer.analyze_text(text)