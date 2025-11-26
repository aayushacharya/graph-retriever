
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
# ============================================================================
# Abstract Base Classes (Interfaces)
# ============================================================================

class Graph(ABC):
    """Abstract base class for graph implementations"""
    
    @abstractmethod
    def build(self, data: Any) -> None:
        """Build the graph structure from data"""
        pass
    
    @abstractmethod
    def add_node(self, node_id: str, features: np.ndarray) -> None:
        """Add a node to the graph"""
        pass
    
    @abstractmethod
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add an edge between two nodes"""
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        pass
    
    @abstractmethod
    def get_node_features(self, node_id: str) -> np.ndarray:
        """Get features of a node"""
        pass