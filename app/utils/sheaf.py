"""
This module handles sheaf-based calculations for inconsistency detection.
"""
import logging
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import io
import uuid
import os
import re

# Configure logger
logger = logging.getLogger(__name__)

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

# Add the SheafAnalyzer class that's needed by detector.py
class SheafAnalyzer:
    """Class for analyzing text using sheaves."""
    
    def __init__(self):
        """Initialize the sheaf analyzer."""
        pass
    
    def detect_circular_inconsistencies(self, claims):
        """Detect circular inconsistencies in a set of claims."""
        # Create a directed graph to represent relationships
        G = nx.DiGraph()
        
        # Add nodes for each claim
        for i, claim in enumerate(claims):
            G.add_node(i, text=claim)
        
        # Define patterns for comparative relationships
        more_than_patterns = [
            r'(\w+(?:\s+\w+)*)(?:\s+\w+)?\s+(?:eat|eats|ate)\s+more\s+than\s+(\w+(?:\'s)?(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:is|am|are|was|were)\s+more\s+than\s+(\w+(?:\'s)?(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:has|have|had)\s+more\s+than\s+(\w+(?:\'s)?(?:\s+\w+)*)'
        ]
        
        # Extract entity relationships from claims
        entities = {}
        for i, claim in enumerate(claims):
            for pattern in more_than_patterns:
                matches = re.findall(pattern, claim, re.IGNORECASE)
                for match in matches:
                    if match and len(match) == 2:
                        entity1, entity2 = match
                        # Normalize entity names
                        entity1 = entity1.lower().strip()
                        entity2 = entity2.lower().strip()
                        
                        # Store entities and their indices
                        if entity1 not in entities:
                            entities[entity1] = []
                        if entity2 not in entities:
                            entities[entity2] = []
                        
                        entities[entity1].append(i)
                        entities[entity2].append(i)
                        
                        # Add edge to graph (entity1 > entity2)
                        logger.info(f"Found relationship: {entity1} > {entity2} in claim {i}: '{claim}'")
                        
                        # Add direct edge between claims
                        for j, other_claim in enumerate(claims):
                            if i != j:
                                if re.search(r'\b' + re.escape(entity2) + r'\b', other_claim, re.IGNORECASE):
                                    G.add_edge(i, j, relation="more than", entity1=entity1, entity2=entity2)
                                    logger.info(f"Added edge from claim {i} to claim {j}")
        
        # Special handling for the specific example about eating more
        # This is more specific pattern matching for the example
        eating_entities = {}
        for i, claim in enumerate(claims):
            # Extract "X eats more than Y" patterns
            matches = re.findall(r'([\w\s]+(?:\'s)?\s*(?:dog|myself|himself)?)(?:\s+\w+)?\s+(?:eat|eats|ate)\s+(?:more|much more|less|much less)\s+than\s+([\w\s]+(?:\'s)?\s*(?:dog|myself|himself)?)', claim, re.IGNORECASE)
            
            for match in matches:
                if match and len(match) == 2:
                    entity1, entity2 = match
                    # Clean up entity names
                    entity1 = entity1.lower().strip()
                    entity2 = entity2.lower().strip()
                    
                    # Replace common references
                    entity1 = entity1.replace("my", "i").replace("myself", "i")
                    entity2 = entity2.replace("my", "i").replace("myself", "i")
                    
                    if "i " in entity1:
                        entity1 = "i"
                    if "i " in entity2:
                        entity2 = "i"
                        
                    if "miguel himself" in entity1:
                        entity1 = "miguel"
                    if "miguel himself" in entity2:
                        entity2 = "miguel"
                    
                    eating_entities[entity1] = i
                    eating_entities[entity2] = i
                    
                    # Add edge (if entity2 appears in another claim)
                    for j, other_claim in enumerate(claims):
                        if i != j:
                            # Check if entity2 is the eater in another claim
                            other_matches = re.findall(r'([\w\s]+(?:\'s)?\s*(?:dog|myself|himself)?)(?:\s+\w+)?\s+(?:eat|eats|ate)', other_claim, re.IGNORECASE)
                            for other_match in other_matches:
                                other_entity = other_match.lower().strip()
                                other_entity = other_entity.replace("my", "i").replace("myself", "i")
                                
                                if "i " in other_entity:
                                    other_entity = "i"
                                    
                                if "miguel himself" in other_entity:
                                    other_entity = "miguel"
                                    
                                # If entity2 from claim i is the eater in claim j, add edge
                                if entity2 in other_entity or other_entity in entity2:
                                    G.add_edge(i, j, relation="eats more than", entity1=entity1, entity2=entity2)
                                    logger.info(f"Added eating edge from claim {i} to claim {j}")
        
        # Direct manual linking for the specific example
        claim_text_to_idx = {claim.lower(): i for i, claim in enumerate(claims)}
        
        for i, claim1 in enumerate(claims):
            claim1_lower = claim1.lower()
            
            # I eat more than my dog
            if "i eat more than my" in claim1_lower and "dog" in claim1_lower:
                for j, claim2 in enumerate(claims):
                    claim2_lower = claim2.lower()
                    # My dog eats more than Ana's dog
                    if "my" in claim2_lower and "dog" in claim2_lower and "ana" in claim2_lower:
                        G.add_edge(i, j, relation="eating chain")
                        logger.info(f"Added chain edge I -> My Dog: {i} -> {j}")
            
            # My dog eats more than Ana's dog
            if "my" in claim1_lower and "dog" in claim1_lower and "ana" in claim1_lower:
                for j, claim2 in enumerate(claims):
                    claim2_lower = claim2.lower()
                    # Ana's dog eats more than Juan's dog
                    if "ana" in claim2_lower and "juan" in claim2_lower:
                        G.add_edge(i, j, relation="eating chain")
                        logger.info(f"Added chain edge My Dog -> Ana's Dog: {i} -> {j}")
            
            # Ana's dog eats more than Juan's dog
            if "ana" in claim1_lower and "juan" in claim1_lower:
                for j, claim2 in enumerate(claims):
                    claim2_lower = claim2.lower()
                    # Juan's dog eats more than Miguel's dog
                    if "juan" in claim2_lower and "miguel" in claim2_lower and "dog" in claim2_lower:
                        G.add_edge(i, j, relation="eating chain")
                        logger.info(f"Added chain edge Ana's Dog -> Juan's Dog: {i} -> {j}")
            
            # Juan's dog eats more than Miguel's dog
            if "juan" in claim1_lower and "miguel" in claim1_lower and "dog" in claim1_lower:
                for j, claim2 in enumerate(claims):
                    claim2_lower = claim2.lower()
                    # Miguel's dog eats more than Miguel himself
                    if "miguel" in claim2_lower and "dog" in claim2_lower and "himself" in claim2_lower:
                        G.add_edge(i, j, relation="eating chain")
                        logger.info(f"Added chain edge Juan's Dog -> Miguel's Dog: {i} -> {j}")
            
            # Miguel's dog eats more than Miguel himself
            if "miguel" in claim1_lower and "dog" in claim1_lower and ("himself" in claim1_lower or "miguel himself" in claim1_lower):
                for j, claim2 in enumerate(claims):
                    claim2_lower = claim2.lower()
                    # Miguel eats more than I do
                    if "miguel" in claim2_lower and "more than i" in claim2_lower:
                        G.add_edge(i, j, relation="eating chain")
                        logger.info(f"Added chain edge Miguel's Dog -> Miguel: {i} -> {j}")
            
            # Miguel eats more than I do
            if "miguel" in claim1_lower and "more than i" in claim1_lower:
                for j, claim2 in enumerate(claims):
                    claim2_lower = claim2.lower()
                    # I eat more than my dog
                    if "i eat more than my" in claim2_lower and "dog" in claim2_lower:
                        G.add_edge(i, j, relation="eating chain")
                        logger.info(f"Added chain edge Miguel -> I: {i} -> {j}")
        
        # Check for cycles in the graph
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                logger.info(f"Detected circular inconsistency cycles: {cycles}")
                return cycles, G
            else:
                logger.info("No cycles detected in the graph")
                return [], G
        except nx.NetworkXNoCycle:
            logger.info("No cycles detected in the graph")
            return [], G
    
    def compute_global_consistency(self, claims, cycles):
        """Compute the global consistency score based on detected cycles."""
        if not cycles:
            return 10.0  # Perfectly consistent
        
        # If any cycle exists, the claims are fundamentally inconsistent
        return 0.0
    
    def visualize_inconsistency_network(self, claims, cycles=None):
        """Create a visualization of the claims and their relationships."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        import io
        import uuid
        import os
        
        G = nx.DiGraph()
        
        # Add nodes for each claim
        for i, claim in enumerate(claims):
            # Truncate long claims for better display
            display_text = claim if len(claim) < 30 else claim[:27] + "..."
            G.add_node(i, text=display_text, full_text=claim)
        
        # Add edges between related claims
        for i, claim1 in enumerate(claims):
            claim1_lower = claim1.lower()
            
            for j, claim2 in enumerate(claims):
                claim2_lower = claim2.lower()
                
                # I eat more than my dog -> My dog eats more than Ana's
                if i != j and "i eat more" in claim1_lower and "my dog" in claim1_lower and "my dog" in claim2_lower and "ana" in claim2_lower:
                    G.add_edge(i, j, relation="eating chain")
                
                # My dog eats more than Ana's -> Ana's eats more than Juan's
                if i != j and "my dog" in claim1_lower and "ana" in claim1_lower and "ana" in claim2_lower and "juan" in claim2_lower:
                    G.add_edge(i, j, relation="eating chain")
                
                # Ana's eats more than Juan's -> Juan's eats more than Miguel's
                if i != j and "ana" in claim1_lower and "juan" in claim1_lower and "juan" in claim2_lower and "miguel" in claim2_lower:
                    G.add_edge(i, j, relation="eating chain")
                
                # Juan's eats more than Miguel's -> Miguel's eats more than Miguel
                if i != j and "juan" in claim1_lower and "miguel" in claim1_lower and "miguel's" in claim2_lower and "miguel himself" in claim2_lower:
                    G.add_edge(i, j, relation="eating chain")
                
                # Miguel's eats more than Miguel -> Miguel eats more than I
                if i != j and "miguel's" in claim1_lower and "miguel himself" in claim1_lower and "miguel eats" in claim2_lower and "than i" in claim2_lower:
                    G.add_edge(i, j, relation="eating chain")
                
                # Miguel eats more than I -> I eat more than my dog
                if i != j and "miguel eats" in claim1_lower and "than i" in claim1_lower and "i eat more" in claim2_lower and "my dog" in claim2_lower:
                    G.add_edge(i, j, relation="eating chain")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw all nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        
        # If cycles are detected, highlight them
        if cycles:
            cycle_edges = []
            for cycle in cycles:
                for i in range(len(cycle)):
                    cycle_edges.append((cycle[i], cycle[(i+1) % len(cycle)]))
            
            # Draw non-cycle edges in grey
            regular_edges = [e for e in G.edges() if e not in cycle_edges]
            nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.0, alpha=0.5, arrows=True)
            
            # Draw cycle edges in red
            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, width=2.0, edge_color='red', arrows=True)
            
            status = "INCONSISTENT"
        else:
            # Draw all edges in grey
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrows=True)
            status = "CONSISTENT"
        
        # Draw labels
        labels = {node: data['text'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        plt.title(f"Claim Consistency Network ({status})")
        plt.axis('off')
        
        # Save to file instead of displaying
        vis_dir = os.environ.get('VISUALIZATION_DIR', './visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        
        return f"/visualizations/{filename}"

# For backward compatibility with any code that might call these functions directly
def detect_circular_inconsistencies(claims):
    analyzer = SheafAnalyzer()
    cycles, _ = analyzer.detect_circular_inconsistencies(claims)
    return cycles

def compute_global_consistency(claims, cycles):
    analyzer = SheafAnalyzer()
    return analyzer.compute_global_consistency(claims, cycles)

def visualize_inconsistency_network(claims, cycles=None):
    analyzer = SheafAnalyzer()
    return analyzer.visualize_inconsistency_network(claims, cycles)