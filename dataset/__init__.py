from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class Dataset(ABC):
    """Abstract base class for dataset implementations"""
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset"""
        pass
    
    @abstractmethod
    def get_graph_data(self) -> Any:
        """Get data for graph construction"""
        pass
    
    @abstractmethod
    def get_queries(self) -> List[np.ndarray]:
        """Get query data for retrieval"""
        pass
    
    @abstractmethod
    def get_ground_truth(self) -> Dict[int, List[str]]:
        """Get ground truth for evaluation"""
        pass


