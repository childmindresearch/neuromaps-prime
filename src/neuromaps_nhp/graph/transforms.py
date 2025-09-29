from neuromaps_nhp.resources.fetch_resource import resource_manager, search_resources
from neuromaps_nhp.graph.transform_graph import BrainAtlasTransformGraph
import matplotlib.pyplot as plt
import networkx as nx

brain_graph = BrainAtlasTransformGraph()

def fetch_transform(source: str, target: str, density: str, hemisphere: str, resource_name: str) -> str:
    # get the graph to find shortest path, if there are multiple vertices involved get all resources from resource_manager and concatenate the transforms then return the concatenated path
    path = brain_graph.find_shortest_path(source, target)
    if path is None:
        raise ValueError(f"No path found from {source} to {target}")
    if len(path) == 1:
        # direct transform
        filepath = resource_manager.get_filepath(
            resource_type="transform",
            source=source,
            target=target,
            density=density,
            hemisphere=hemisphere,
            resource_name=resource_name
        )
        if filepath is None:
            raise ValueError(f"No transform found from {source} to {target} with density {density}, hemisphere {hemisphere}, resource_name {resource_name}")
        return filepath
    else:
        # multiple transforms, concatenate them
        transform_paths = []
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            filepath = resource_manager.get_filepath(
                resource_type="transform",
                source=src,
                target=tgt,
                density=density,
                hemisphere=hemisphere,
                resource_name=resource_name
            )
            if filepath is None:
                raise ValueError(f"No transform found from {src} to {tgt} with density {density}, hemisphere {hemisphere}, resource_name {resource_name}")
            transform_paths.append(filepath)
        # concatenate paths with commas
        return ",".join(transform_paths)
    

def search_transforms(source: str = None, target: str = None, density: str = None, 
                     hemisphere: str = None, resource_name: str = None, 
                     show_graph: bool = True) -> dict:
    """
    Search for available transforms and show all possible paths.
    
    Args:
        source: Source atlas (optional filter)
        target: Target atlas (optional filter)  
        density: Density filter (optional)
        hemisphere: Hemisphere filter (optional)
        resource_name: Resource name filter (optional)
        show_graph: Whether to display the graph visualization
        
    Returns:
        Dictionary containing available transforms and possible paths
    """
    # Search for available transforms
    available_transforms = search_resources(
        resource_type="transform",
        source=source,
        target=target,
        density=density,
        hemisphere=hemisphere,
        resource_name=resource_name
    )
    
    # Get all unique atlases from available transforms
    atlases = set()
    direct_transforms = []
    
    for transform in available_transforms:
        atlases.add(transform.source)
        atlases.add(transform.target)
        direct_transforms.append((transform.source, transform.target))

    # Find all possible paths between atlases
    all_paths = {}
    atlases_list = list(atlases)
    
    for src_atlas in atlases_list:
        for tgt_atlas in atlases_list:
            if src_atlas != tgt_atlas:
                try:
                    path = brain_graph.find_shortest_path(src_atlas, tgt_atlas)
                    if path and len(path) > 1:
                        path_key = f"{src_atlas} -> {tgt_atlas}"
                        all_paths[path_key] = {
                            'path': path,
                            'length': len(path) - 1,
                            'direct': len(path) == 2
                        }
                except:
                    continue
    
    # Create visualization if requested
    if show_graph and atlases:
        plt.figure(figsize=(12, 8))
        
        # Create a networkx graph
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(atlases)
        
        # Add edges from direct transforms
        G.add_edges_from(direct_transforms)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1500, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title("Brain Atlas Transform Graph\n(Direct transforms shown as edges)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Prepare summary
    summary = {
        'available_transforms': available_transforms,
        'unique_atlases': sorted(list(atlases)),
        'direct_transforms': len(direct_transforms),
        'all_possible_paths': all_paths,
        'total_possible_paths': len(all_paths)
    }
    
    return summary

def print_transform_summary(summary: dict):
    """Print a formatted summary of transform search results."""
    print("=" * 60)
    print("BRAIN ATLAS TRANSFORM SEARCH RESULTS")
    print("=" * 60)
    
    print(f"\nUnique Atlases Found: {len(summary['unique_atlases'])}")
    print(", ".join(summary['unique_atlases']))
    
    print(f"\nDirect Transforms Available: {summary['direct_transforms']}")
    
    print(f"\nTotal Possible Paths: {summary['total_possible_paths']}")
    
    print("\nDirect Paths:")
    direct_paths = {k: v for k, v in summary['all_possible_paths'].items() if v['direct']}
    for path_name, path_info in direct_paths.items():
        print(f"  {path_name}")
    
    print("\nIndirect Paths (multi-step):")
    indirect_paths = {k: v for k, v in summary['all_possible_paths'].items() if not v['direct']}
    for path_name, path_info in indirect_paths.items():
        path_str = " -> ".join(path_info['path'])
        print(f"  {path_name} (steps: {path_info['length']})")
        print(f"    Route: {path_str}")
    
    print("\nAvailable Transform Files:")
    for i, transform in enumerate(summary['available_transforms'], 1):
        print(f"  {i}. {transform.source} -> {transform.target}")
        print(f"     Density: {transform.density}, "
              f"Hemisphere: {transform.hemisphere}, "
              f"Resource: {transform.resource_name}")

if __name__ == "__main__":
    # Search for specific density and hemisphere
    source="S1200"
    target="Yerkes19"
    density="10k" 
    hemisphere="L"
    show_graph=False
    
    specific_results = search_transforms(
        source=source,
        target=target,
        density=density,
        hemisphere=hemisphere,
        resource_name="sphere",
    )
    print(f"Found {len(specific_results['available_transforms'])} transforms for source {source}, target {target}, {density} density, {hemisphere} hemisphere")