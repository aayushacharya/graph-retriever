
from abc import ABC, abstractmethod
from typing import List, Dict
from graph import Graph
from retriever import Retriever
import numpy as np
from dataset import Dataset



class Experiment(ABC):
    """Abstract base class for experiment runners"""
    
    @abstractmethod
    def setup(self, graph: Graph, retriever: Retriever, dataset: Dataset) -> None:
        """Setup the experiment with components"""
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, float]:
        """Run the experiment and return metrics"""
        pass
    
    @abstractmethod
    def evaluate(self, predictions: List[List[str]], 
                 ground_truth: Dict[int, List[str]]) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        pass
