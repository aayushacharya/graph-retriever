
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field
# ============================================================================
# Abstract Base Classes (Interfaces)
# ============================================================================

class Node(BaseModel):
    """Data model for a single node in the graph"""
    node_id: str = Field(description="Unique identifier for the node")
    node_type: Optional[str] = Field(description="Type of the node")
    attributes: Dict[str, Any] = Field(description="Attributes of the node")


class Neighbor(BaseModel):
    """Data model for a neighbor node with relation"""
    node: Node = Field(description="The neighbor node")
    relation: str = Field(description="The relation to the neighbor")


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
    def search_nodes(self, criteria: Dict[str, Any]) -> List[Any]:
        """Search for nodes in the graph based on given criteria"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its identifier"""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, hops: int, relations: Optional[List[str]]) -> List[Neighbor]:
        """Get neighbors of a given node"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass