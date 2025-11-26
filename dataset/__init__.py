

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataset.synthetic_dataset import SyntheticDataset
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


def create_dataset(dataset_type: str) -> Dataset:
    """Factory function to create dataset instances"""
    datasets = {
        'synthetic': SyntheticDataset,
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_type]()