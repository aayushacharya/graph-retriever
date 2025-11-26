from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from graph import Graph
import numpy as np


class Retriever(ABC):
    """Abstract base class for retriever models"""
    
    @abstractmethod
    def initialize(self, graph: Graph) -> None:
        """Initialize the retriever with a graph"""
        pass
    
    @abstractmethod
    def retrieve(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k nodes for a given query
        
        Returns:
            List of (node_id, score) tuples
        """
        pass
    
    @abstractmethod
    def train(self, training_data: List[Tuple[np.ndarray, str]]) -> None:
        """Train the retriever model"""
        pass