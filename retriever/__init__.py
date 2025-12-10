from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from graph import Graph
import numpy as np


class Retriever(ABC):
    """Abstract base class for retriever models"""
    
    @abstractmethod
    def retrieve(self, query: str, graph: Graph) -> List[Any]:
        """Retrieve answer for a given query
        
        Returns:
            List of (node_id, score) tuples
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the name of the retriever model"""
        pass
    