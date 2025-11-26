from graph import Graph
from graph.simple_graph import SimpleGraph

def create_graph(graph_type: str) -> Graph:
    """Factory function to create graph instances"""
    graphs = {
        'simple': SimpleGraph,
    }
    
    if graph_type not in graphs:
        raise ValueError(f"Unknown graph type: {graph_type}. Available: {list(graphs.keys())}")
    
    return graphs[graph_type]()