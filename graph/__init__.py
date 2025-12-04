
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
    def execute_query(self, query: str) -> List[Any]:
        """Execute a query on the graph and return results"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass