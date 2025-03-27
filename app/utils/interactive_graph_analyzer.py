"""
This module provides interactive visualization of inconsistency networks.
"""
import logging
import networkx as nx
import os
import uuid
from typing import List, Optional, Tuple, Dict, Any, Set

logger = logging.getLogger(__name__)

class InteractiveGraphAnalyzer:
    """Class for creating interactive visualizations of logical consistency networks."""
    
    def __init__(self, height: str = "800px", width: str = "100%", bgcolor: str = "#ffffff", 
                font_color: str = "#000000"):
        """
        Initialize the interactive graph analyzer.
        
        Args:
            height: Height of the visualization canvas
            width: Width of the visualization canvas
            bgcolor: Background color
            font_color: Font color
        """
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color
    
    async def visualize_inconsistency_network(self, 
                                    claims: List[str], 
                                    cycles: List[List[int]] = None, 
                                    G: Optional[nx.DiGraph] = None) -> str:
        """
        Create an interactive visualization of the inconsistency network.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles (lists of claim indices)
            G: Graph to visualize (optional)
            
        Returns:
            Path to saved HTML visualization
        """
        try:
            # Importamos aquí para manejar posibles errores de importación
            from pyvis.network import Network
            
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
            
            # Create a new undirected graph for visualization
            undirected_G = nx.Graph()
            
            # Add all nodes
            for node, attrs in G.nodes(data=True):
                undirected_G.add_node(node, **attrs)
            
            # Gather all cycle edges
            cycle_edges = set()
            for cycle in cycles:
                for i in range(len(cycle)):
                    idx1 = cycle[i]
                    idx2 = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add((idx1, idx2))
                    cycle_edges.add((idx2, idx1))  # Add both directions
            
            # Process the edges - only add each pair once but keep track of edge attributes
            processed_pairs = set()
            
            for u, v, data in G.edges(data=True):
                # Create a canonical representation of the edge (smaller index first)
                edge_pair = tuple(sorted([u, v]))
                
                # Skip if we've already processed this pair
                if edge_pair in processed_pairs:
                    continue
                
                # Check if this edge is part of a cycle
                is_cycle_edge = (u, v) in cycle_edges or (v, u) in cycle_edges
                
                # Check consistency
                is_consistent = data.get('is_consistent', True) and not is_cycle_edge
                
                # Add the edge with appropriate attributes
                undirected_G.add_edge(u, v, 
                                     is_consistent=is_consistent,
                                     is_cycle_edge=is_cycle_edge,
                                     consistency=data.get('consistency', 8.0 if is_consistent else 2.0))
                
                # Mark as processed
                processed_pairs.add(edge_pair)
            
            # Create a Pyvis network
            net = Network(height=self.height, width=self.width, bgcolor=self.bgcolor, 
                        font_color=self.font_color, directed=False, notebook=False)
            
            # Configure physics for better node separation
            net.barnes_hut(spring_length=200, spring_strength=0.05, damping=0.09)
            
            # Add nodes to the network
            for node_id in undirected_G.nodes():
                node_attrs = undirected_G.nodes[node_id]
                
                # Get claim text if available
                if isinstance(node_id, int) and 0 <= node_id < len(claims):
                    claim_text = claims[node_id]
                    
                    # For better readability in tooltips, we'll use the full text
                    # But for node labels, we'll keep it shorter
                    if len(claim_text) > 30:
                        label_text = f"{node_id}: {claim_text[:27]}..."
                    else:
                        label_text = f"{node_id}: {claim_text}"
                    
                    # Add the node with custom settings for better visibility
                    net.add_node(
                        n_id=node_id,
                        label=label_text,
                        title=f"Claim {node_id}: {claim_text}",  # Tooltip with full text
                        color="#6BAED6",  # Light blue
                        shape="box",
                        borderWidth=2,
                        font={"size": 16},
                        size=25  # Slightly larger nodes
                    )
                else:
                    # Fallback for non-claim nodes
                    net.add_node(
                        n_id=node_id,
                        label=str(node_id),
                        color="#C0C0C0"  # Grey for non-claim nodes
                    )
            
            # Add edges to the network
            for u, v, data in undirected_G.edges(data=True):
                # Determine if this is a cycle edge
                is_cycle_edge = data.get('is_cycle_edge', False)
                is_consistent = data.get('is_consistent', True)
                
                edge_color = "red" if is_cycle_edge or not is_consistent else "green"
                edge_style = "dashed" if is_cycle_edge or not is_consistent else "solid"
                
                # Create label with consistency score if available
                if 'consistency' in data:
                    label = f"{data['consistency']:.1f}"
                    if is_cycle_edge:
                        label += "†"  # Add a symbol to mark cycle edges
                else:
                    label = "†" if is_cycle_edge else ""
                
                # Add edge with appropriate styling - no arrows for any edge
                net.add_edge(
                    source=u,
                    to=v,
                    color=edge_color,
                    label=label,
                    width=2.0 if is_cycle_edge or not is_consistent else 1.5,
                    dashes=True if edge_style == "dashed" else False,
                    arrows=""  # No arrows for any edge
                )
            
            # Add custom HTML header to include explanation and title
            title_html = "Inconsistency Network" if cycles else "Consistency Network"
            cycles_info = f"{len(cycles)} cycles detected" if cycles else "No inconsistencies detected"
            
            # Prepare additional HTML to include before the graph
            custom_html = f"""
            <div style="margin-bottom: 20px; font-family: Arial, sans-serif;">
                <h2 style="color: #333;">{title_html} - {cycles_info}</h2>
                <div style="margin: 10px 0;">
                    <span style="color: green; font-weight: bold; margin-right: 20px;">
                        <span style="border-bottom: 2px solid green; padding-bottom: 2px;">
                            — Consistency Relationship
                        </span>
                    </span>
                    <span style="color: red; font-weight: bold;">
                        <span style="border-bottom: 2px dashed red; padding-bottom: 2px;">
                            - - - Inconsistency Relationship
                        </span>
                    </span>
                </div>
                <p style="color: #666;">
                    Click and drag nodes to rearrange. Hover over nodes for full claim text.
                    Double-click to focus on a node and its connections.
                </p>
            </div>
            """
            
            # If there are inconsistency cycles, add them to the HTML
            if cycles:
                custom_html += '<div style="margin-bottom: 20px; font-family: Arial, sans-serif;">'
                custom_html += '<h3 style="color: #333;">Detected Inconsistency Cycles:</h3>'
                custom_html += '<ul style="list-style-type: none; padding: 0;">'
                
                for i, cycle in enumerate(cycles):
                    cycle_text = " → ".join([f"{idx}: {claims[idx][:40]+'...' if len(claims[idx])>40 else claims[idx]}" 
                                            for idx in cycle if 0 <= idx < len(claims)])
                    if cycle_text:
                        custom_html += f'<li style="margin-bottom: 10px; color: #444;">'
                        custom_html += f'<span style="font-weight: bold;">Cycle {i+1}:</span> {cycle_text}'
                        custom_html += '</li>'
                
                custom_html += '</ul></div>'
            
            # Add detailed claim list
            custom_html += '<div style="margin-top: 30px; font-family: Arial, sans-serif;">'
            custom_html += '<h3 style="color: #333;">All Claims:</h3>'
            custom_html += '<ul style="padding-left: 20px;">'
            
            for i, claim in enumerate(claims):
                custom_html += f'<li style="margin-bottom: 5px;"><b>{i}:</b> {claim}</li>'
            
            custom_html += '</ul></div>'
            
            # Save visualization
            vis_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            filename = f"interactive_inconsistency_{uuid.uuid4().hex[:8]}.html"
            filepath = os.path.join(vis_dir, filename)
            
            # Save with custom HTML
            net.save_graph(filepath)
            
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Insert custom HTML after the body tag
            modified_html = html_content.replace('<body>', f'<body>\n{custom_html}')
            
            # Write back the modified file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_html)
            
            # Return path to visualization
            return f"/visualizations/{filename}"
            
        except ImportError as e:
            logger.error(f"Error importing Pyvis: {e}. Falling back to static visualization.")
            # Fallback to generating a static image with NetworkX and matplotlib
            return await self._fallback_visualization(claims, cycles, G)
    
    async def _fallback_visualization(self, 
                               claims: List[str], 
                               cycles: List[List[int]] = None, 
                               G: Optional[nx.DiGraph] = None) -> str:
        """
        Fallback method to create a static visualization if Pyvis is not available.
        
        Args:
            claims: List of claims
            cycles: List of inconsistency cycles
            G: Graph to visualize (optional)
            
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        
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
                                  arrows=False,  # No arrows
                                  width=1.5)
        
        # Draw inconsistent edges in red
        if red_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=red_edges,
                                  edge_color='red',
                                  arrows=False,  # No arrows
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
                # Create node label with claim text (truncated if too long)
                claim_text = claims[node]
                if len(claim_text) > 40:
                    label = f"{node}: {claim_text[:37]}..."
                else:
                    label = f"{node}: {claim_text}"
                node_labels[node] = label
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
        
        # Save visualization
        vis_dir = os.environ.get("VISUALIZATION_DIR", "./visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"fallback_inconsistency_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.warning(f"Generated fallback static visualization instead of interactive one")
        
        # Return path to visualization
        return f"/visualizations/{filename}"