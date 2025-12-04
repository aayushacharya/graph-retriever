from graph import Graph
from graph.freebase import Freebase

def create_graph(graph_type: str) -> Graph:
    """Factory function to create graph instances"""
    graphs = {
        'freebase': Freebase,
    }
    
    if graph_type not in graphs:
        raise ValueError(f"Unknown graph type: {graph_type}. Available: {list(graphs.keys())}")
    
    return graphs[graph_type]()