"""
This module handles visualization of inconsistency networks.
"""
import logging
import networkx as nx
import matplotlib.pyplot as plt
import os
import uuid

logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """Class for visualizing logical consistency networks."""
    
    def __init__(self):
        """Initialize the graph analyzer."""
        pass
    
    async def visualize_inconsistency_network(self, claims: list, cycles: list = None, G: nx.DiGraph = None) -> str:
        """
        Create a visualization of the inconsistency network.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles (lists of claim indices)
            G: Graph to visualize (optional)
            
        Returns:
            Path to saved visualization
        """
        if cycles is None:
            cycles = []
        
        # If no graph provided, create one from claims and cycles
        if G is None:
            G = nx.DiGraph()
            
            # Add all claims as nodes
            for i, claim in enumerate(claims):
                G.add_node(i, text=claim)
            
            # Add edges from cycles
            for cycle in cycles:
                for i in range(len(cycle)):
                    idx1 = cycle[i]
                    idx2 = cycle[(i + 1) % len(cycle)]
                    if idx1 < len(claims) and idx2 < len(claims):
                        G.add_edge(idx1, idx2, 
                                is_consistent=False,
                                is_cycle_edge=True)
        
        plt.figure(figsize=(14, 10))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='skyblue',
                             node_size=1000,
                             alpha=0.8)
        
        # Gather all cycle edges
        cycle_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                idx1 = cycle[i]
                idx2 = cycle[(i + 1) % len(cycle)]
                cycle_edges.add((idx1, idx2))
        
        # Categorize edges
        green_edges = []
        red_edges = []
        
        for u, v, data in G.edges(data=True):
            # Check if this edge is in a cycle or marked as inconsistent
            in_cycle = (u, v) in cycle_edges
            is_inconsistent = not data.get('is_consistent', True) or data.get('is_cycle_edge', False)
            
            if in_cycle or is_inconsistent:
                red_edges.append((u, v))
            else:
                green_edges.append((u, v))
        
        # Draw consistent edges in green
        if green_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=green_edges,
                                  edge_color='green',
                                  arrows=True,
                                  width=1.5)
        
        # Draw inconsistent edges in red
        if red_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=red_edges,
                                  edge_color='red',
                                  arrows=True,
                                  width=2,
                                  style='dashed')
        
        # Add edge labels with consistency scores
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            label = ""
            if 'consistency' in data:
                label = f"{data['consistency']:.1f}"
            
            # Mark cycle edges
            if (u, v) in cycle_edges:
                label += "†"  # Add a symbol to mark cycle edges
                
            if label:
                edge_labels[(u, v)] = label
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        
        # Add node labels
        node_labels = {}
        for node in G.nodes():
            if isinstance(node, int) and 0 <= node < len(claims):
                node_labels[node] = f"{node}"
            else:
                node_labels[node] = str(node)
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Add title
        if cycles:
            plt.title(f"Inconsistency Network - {len(cycles)} cycles detected", fontsize=14)
        else:
            plt.title("Consistency Network - No inconsistencies detected", fontsize=14)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Consistent Relationship'),
            Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Inconsistent Relationship')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Turn off axis
        plt.axis('off')
        
        # Add text for claims
        y_pos = -0.05
        for i, claim in enumerate(claims):
            plt.figtext(0.05, y_pos, f"{i}. {claim}", fontsize=9, 
                     bbox=dict(facecolor='white', alpha=0.8))
            y_pos -= 0.05
        
        # Save visualization
        vis_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"inconsistency_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return path to visualization
        return f"/visualizations/{filename}"