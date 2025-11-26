from . import Retriever
from graph import Graph
import numpy as np
from typing import List, Tuple


class CosineRetriever(Retriever):
    """Cosine similarity based retriever"""
    
    def __init__(self):
        self.graph = None
        self.node_list = []
    
    def initialize(self, graph: Graph) -> None:
        self.graph = graph
        self.node_list = list(graph.nodes.keys())
    
    def retrieve(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        scores = []
        for node_id in self.node_list:
            features = self.graph.get_node_features(node_id)
            if len(features) > 0:
                similarity = np.dot(query, features) / (
                    np.linalg.norm(query) * np.linalg.norm(features) + 1e-10
                )
                scores.append((node_id, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def train(self, training_data: List[Tuple[np.ndarray, str]]) -> None:
        # Cosine retriever doesn't require training
        pass