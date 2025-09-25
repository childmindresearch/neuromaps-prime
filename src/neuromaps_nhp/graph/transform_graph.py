import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Set, List, Optional, Tuple, Union, Dict

class BrainAtlasTransformGraph:
    """
    A class to manage brain atlas transformation graphs with human and non-human primate (NHP) nodes.
    Supports directed graphs for asymmetric transformations.
    """
    
    def __init__(self, 
                 human_nodes: Optional[Set[str]] = None, 
                 nhp_nodes: Optional[Set[str]] = None,
                 cross_species_edges: Optional[List[Tuple[str, str]]] = None,
                 directed: bool = True):
        """
        Initialize the brain atlas transformation graph.
        
        Parameters:
        -----------
        human_nodes : set, optional
            Set of human atlas node names
        nhp_nodes : set, optional  
            Set of NHP atlas node names
        cross_species_edges : list of tuples, optional
            List of (human_node, nhp_node) pairs for cross-species connections
        directed : bool, optional
            Whether to create a directed graph (default: True)
        """
        # Default nodes if none provided
        self.human_nodes = human_nodes or {"civet", "fsaverage", "fslr", "s1200"}
        self.nhp_nodes = nhp_nodes or {"civetnmt", "yerkes19", "d99", "mebrains", "nmt2sym"}
        
        # Sanitize node names
        self.human_nodes = self._sanitize_nodes(self.human_nodes)
        self.nhp_nodes = self._sanitize_nodes(self.nhp_nodes)
        
        # Initialize graph (directed or undirected)
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        
        # Default cross-species connection
        self.cross_species_edges = cross_species_edges or [("s1200", "yerkes19")]
        
        # Dictionary to store transformation files for each edge
        self.transform_files: Dict[Tuple[str, str], str] = {}
        
        # Build the graph
        self._build_graph()
    
    def _sanitize_nodes(self, nodes: Union[Set[str], List[str]]) -> Set[str]:
        """
        Sanitize node names by removing whitespace and converting to lowercase.
        
        Parameters:
        -----------
        nodes : set or list
            Node names to sanitize
            
        Returns:
        --------
        set
            Sanitized node names
        """
        if isinstance(nodes, list):
            nodes = set(nodes)
        
        return {node.strip().lower() for node in nodes if node.strip()}
    
    def _build_graph(self):
        """Build the complete graph with intra-species and cross-species connections."""
        # Add all nodes
        self.graph.add_nodes_from(self.human_nodes)
        self.graph.add_nodes_from(self.nhp_nodes)
        
        # Create complete subgraph for human nodes
        self._add_complete_subgraph(self.human_nodes)
        
        # Create complete subgraph for NHP nodes  
        self._add_complete_subgraph(self.nhp_nodes)
        
        # Add cross-species connections
        self._add_cross_species_connections()
    
    def _add_complete_subgraph(self, nodes: Set[str]):
        """Add edges to make a complete subgraph among the given nodes."""
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    self.graph.add_edge(node1, node2)
                    if self.directed:
                        self.graph.add_edge(node2, node1)  # Add reverse direction
    
    def _add_cross_species_connections(self):
        """Add cross-species edges."""
        for human_node, nhp_node in self.cross_species_edges:
            if human_node in self.human_nodes and nhp_node in self.nhp_nodes:
                self.graph.add_edge(human_node, nhp_node)
                if self.directed:
                    self.graph.add_edge(nhp_node, human_node)  # Add reverse direction
            else:
                print(f"Warning: Cross-species edge ({human_node}, {nhp_node}) contains invalid nodes")
    
    def add_human_node(self, node_name: str):
        """
        Add a new human node and connect it to all existing human nodes.
        
        Parameters:
        -----------
        node_name : str
            Name of the human node to add
        """
        node_name = node_name.strip().lower()
        if node_name not in self.human_nodes:
            self.human_nodes.add(node_name)
            self.graph.add_node(node_name)
            
            # Connect to all existing human nodes
            for existing_node in self.human_nodes:
                if existing_node != node_name:
                    self.graph.add_edge(node_name, existing_node)
                    if self.directed:
                        self.graph.add_edge(existing_node, node_name)
    
    def add_nhp_node(self, node_name: str):
        """
        Add a new NHP node and connect it to all existing NHP nodes.
        
        Parameters:
        -----------
        node_name : str
            Name of the NHP node to add
        """
        node_name = node_name.strip().lower()
        if node_name not in self.nhp_nodes:
            self.nhp_nodes.add(node_name)
            self.graph.add_node(node_name)
            
            # Connect to all existing NHP nodes
            for existing_node in self.nhp_nodes:
                if existing_node != node_name:
                    self.graph.add_edge(node_name, existing_node)
                    if self.directed:
                        self.graph.add_edge(existing_node, node_name)
    
    def add_cross_species_edge(self, human_node: str, nhp_node: str):
        """
        Add a cross-species edge between human and NHP nodes.
        
        Parameters:
        -----------
        human_node : str
            Human atlas node name
        nhp_node : str
            NHP atlas node name
        """
        human_node = human_node.strip().lower()
        nhp_node = nhp_node.strip().lower()
        
        if human_node in self.human_nodes and nhp_node in self.nhp_nodes:
            self.graph.add_edge(human_node, nhp_node)
            if self.directed:
                self.graph.add_edge(nhp_node, human_node)
            if (human_node, nhp_node) not in self.cross_species_edges:
                self.cross_species_edges.append((human_node, nhp_node))
        else:
            print(f"Error: Invalid nodes. {human_node} must be in human nodes, {nhp_node} must be in NHP nodes")
    
    def set_transform_file(self, source: str, target: str, file_path: str):
        """
        Set transformation file for a specific edge.
        
        Parameters:
        -----------
        source : str
            Source node name
        target : str
            Target node name
        file_path : str
            Path to transformation file
        """
        source = source.strip().lower()
        target = target.strip().lower()
        
        if self.graph.has_edge(source, target):
            self.transform_files[(source, target)] = file_path
        else:
            print(f"Error: No edge exists between '{source}' and '{target}'")
    
    def get_transform_file(self, source: str, target: str) -> Optional[str]:
        """
        Get transformation file for a specific edge.
        
        Parameters:
        -----------
        source : str
            Source node name
        target : str
            Target node name
            
        Returns:
        --------
        str or None
            Path to transformation file, None if not set
        """
        source = source.strip().lower()
        target = target.strip().lower()
        
        return self.transform_files.get((source, target))
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find the shortest path between two nodes.
        
        Parameters:
        -----------
        source : str
            Source node name
        target : str
            Target node name
            
        Returns:
        --------
        list or None
            Shortest path as list of nodes, None if no path exists
        """
        source = source.strip().lower()
        target = target.strip().lower()
        
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            print(f"No path exists between '{source}' and '{target}'")
            return None
        except nx.NodeNotFound as e:
            print(f"Node not found: {e}")
            return None
    
    def get_path_length(self, source: str, target: str) -> Optional[int]:
        """
        Get the length of shortest path between two nodes.
        
        Parameters:
        -----------
        source : str
            Source node name
        target : str
            Target node name
            
        Returns:
        --------
        int or None
            Path length, None if no path exists
        """
        source = source.strip().lower()
        target = target.strip().lower()
        
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            print(f"No path exists between '{source}' and '{target}'")
            return None
        except nx.NodeNotFound as e:
            print(f"Node not found: {e}")
            return None
    
    def _create_layout(self) -> dict:
        """Create custom layout with separated human and NHP clusters."""
        pos = {}
        
        human_nodes = list(self.human_nodes)
        nhp_nodes = list(self.nhp_nodes)
        
        # Human cluster (left side)
        human_center = (-2, 0)
        human_radius = 1
        for i, node in enumerate(human_nodes):
            angle = 2 * np.pi * i / len(human_nodes)
            pos[node] = (human_center[0] + human_radius * np.cos(angle),
                        human_center[1] + human_radius * np.sin(angle))
        
        # NHP cluster (right side)
        nhp_center = (2, 0)
        nhp_radius = 1.2
        for i, node in enumerate(nhp_nodes):
            angle = 2 * np.pi * i / len(nhp_nodes)
            pos[node] = (nhp_center[0] + nhp_radius * np.cos(angle),
                        nhp_center[1] + nhp_radius * np.sin(angle))
        
        return pos
    
    def plot_graph(self, 
                   figsize: Tuple[int, int] = (14, 8),
                   save_path: Optional[str] = None,
                   show_plot: bool = True,
                   title: Optional[str] = None):
        """
        Plot the transformation graph.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size as (width, height)
        save_path : str, optional
            Path to save the plot. If None, saves as 'transform_graph.png'
        show_plot : bool
            Whether to display the plot
        title : str, optional
            Plot title
        """
        if title is None:
            title = f"Human-NHP Brain Atlas Transformation Graph {'(Directed)' if self.directed else '(Undirected)'}"
            
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = self._create_layout()
        
        # Draw human nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.human_nodes,
                              node_color='lightblue', node_size=1500, label='Human')
        
        # Draw NHP nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.nhp_nodes,
                              node_color='lightcoral', node_size=1500, label='NHP')
        
        # Separate cross-species edges from others
        cross_species_edge_list = []
        for human_node, nhp_node in self.cross_species_edges:
            if self.graph.has_edge(human_node, nhp_node):
                cross_species_edge_list.append((human_node, nhp_node))
                if self.directed and self.graph.has_edge(nhp_node, human_node):
                    cross_species_edge_list.append((nhp_node, human_node))
        
        other_edges = [(u, v) for u, v in self.graph.edges() 
                       if (u, v) not in cross_species_edge_list]
        
        # Draw regular edges
        edge_params = {'alpha': 0.3, 'width': 1}
        if self.directed:
            edge_params.update({'arrows': True, 'arrowsize': 10, 'arrowstyle': '->'})
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=other_edges, **edge_params)
        
        # Draw cross-species edges with emphasis
        cross_edge_params = {'edge_color': 'red', 'width': 3, 'alpha': 0.8}
        if self.directed:
            cross_edge_params.update({'arrows': True, 'arrowsize': 15, 'arrowstyle': '->'})
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=cross_species_edge_list, 
                              **cross_edge_params)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')
        
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = 'transform_graph.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_graph_info(self) -> dict:
        """
        Get information about the graph.
        
        Returns:
        --------
        dict
            Graph statistics and information
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_human_nodes': len(self.human_nodes),
            'num_nhp_nodes': len(self.nhp_nodes),
            'human_nodes': sorted(list(self.human_nodes)),
            'nhp_nodes': sorted(list(self.nhp_nodes)),
            'cross_species_edges': self.cross_species_edges,
            'is_connected': nx.is_connected(self.graph.to_undirected()) if self.directed else nx.is_connected(self.graph),
            'directed': self.directed,
            'transform_files_count': len(self.transform_files)
        }
    
    def __str__(self) -> str:
        """String representation of the graph."""
        info = self.get_graph_info()
        graph_type = "Directed" if self.directed else "Undirected"
        return (f"BrainAtlasTransformGraph ({graph_type}): {info['num_nodes']} nodes, {info['num_edges']} edges\n"
                f"Human nodes ({info['num_human_nodes']}): {', '.join(info['human_nodes'])}\n"
                f"NHP nodes ({info['num_nhp_nodes']}): {', '.join(info['nhp_nodes'])}\n"
                f"Cross-species edges: {info['cross_species_edges']}\n"
                f"Transform files registered: {info['transform_files_count']}")


# Example usage
if __name__ == "__main__":
    # Create directed graph with default nodes
    brain_graph = BrainAtlasTransformGraph(directed=True)
    
    # Print graph info
    print(brain_graph)
    print()
    
    # Set transformation files
    brain_graph.set_transform_file("s1200", "yerkes19", "/path/to/s1200_to_yerkes19.h5")
    brain_graph.set_transform_file("yerkes19", "s1200", "/path/to/yerkes19_to_s1200.h5")
    
    # Test path finding
    path = brain_graph.find_shortest_path("fsaverage", "d99")
    print(f"Path from fsaverage to d99: {path}")
    
    # Plot the graph
    brain_graph.plot_graph(save_path='brain_atlas_graph.png')