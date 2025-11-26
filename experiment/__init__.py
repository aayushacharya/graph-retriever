
from abc import ABC, abstractmethod
from typing import List, Dict
from graph import Graph
from retriever import Retriever
from experiment.standard_experiment import StandardExperiment
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


def create_experiment(experiment_type: str) -> Experiment:
    """Factory function to create experiment instances"""
    experiments = {
        'standard': StandardExperiment,
    }
    
    if experiment_type not in experiments:
        raise ValueError(f"Unknown experiment type: {experiment_type}. Available: {list(experiments.keys())}")
    
    return experiments[experiment_type]()