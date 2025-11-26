from . import Graph
from typing import List, Dict, Any, Tuple
import numpy as np


class SimpleGraph(Graph):
    """Simple adjacency list graph implementation"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.node_features = {}
    
    def build(self, data: Any) -> None:
        """Build graph from data dictionary"""
        if isinstance(data, dict):
            for node_id, features in data.get('nodes', {}).items():
                self.add_node(node_id, features)
            for edge in data.get('edges', []):
                self.add_edge(edge['source'], edge['target'], edge.get('weight', 1.0))
    
    def add_node(self, node_id: str, features: np.ndarray) -> None:
        self.nodes[node_id] = True
        self.node_features[node_id] = features
        if node_id not in self.edges:
            self.edges[node_id] = []
    
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append((target, weight))
    
    def get_neighbors(self, node_id: str) -> List[str]:
        return [target for target, _ in self.edges.get(node_id, [])]
    
    def get_node_features(self, node_id: str) -> np.ndarray:
        return self.node_features.get(node_id, np.array([]))

