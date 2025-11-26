import numpy as np
from typing import List, Dict, Any
from . import Dataset


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing"""
    
    def __init__(self, num_nodes: int = 100, feature_dim: int = 64, num_queries: int = 20):
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.data = None
        self.queries = None
        self.ground_truth = None
    
    def load(self) -> None:
        # Generate random node features
        nodes = {}
        for i in range(self.num_nodes):
            features = np.random.randn(self.feature_dim)
            features /= np.linalg.norm(features)
            nodes[f"node_{i}"] = features
        
        self.data = {'nodes': nodes, 'edges': []}
        
        # Generate queries (similar to some nodes)
        self.queries = []
        self.ground_truth = {}
        for i in range(self.num_queries):
            # Pick a random node and add noise
            base_node = np.random.randint(0, self.num_nodes)
            query = nodes[f"node_{base_node}"] + np.random.randn(self.feature_dim) * 0.1
            query /= np.linalg.norm(query)
            self.queries.append(query)
            self.ground_truth[i] = [f"node_{base_node}"]
    
    def get_graph_data(self) -> Any:
        return self.data
    
    def get_queries(self) -> List[np.ndarray]:
        return self.queries
    
    def get_ground_truth(self) -> Dict[int, List[str]]:
        return self.ground_truth