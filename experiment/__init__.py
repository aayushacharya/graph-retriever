
from abc import ABC, abstractmethod
from typing import List, Dict
from graph import Graph
from retriever import Retriever
from pydantic import BaseModel
import numpy as np
from dataset import Dataset, QAFormat
import time

class QAResult(QAFormat):
    """Data model for a single question answer result in the dataset"""
    predicted_answers: List[str] = []
    metrics: Dict[str, float] = {}

class Result(BaseModel):
    """Data model for a single result item"""
    qa_results: List[QAResult]
    aggregated_metrics: Dict[str, float]


class Experiment(ABC):
    """Abstract base class for experiment runners"""
    
    @abstractmethod
    def run(self,graph: Graph, retriever: Retriever, dataset: Dataset) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        pass

    @abstractmethod
    def dump_results(self, results: Result, output_path: str) -> None:
        """Dump experiment results to a file"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

def get_report_name(experiment: Experiment, graph: Graph, retriever: Retriever, dataset: Dataset) -> str:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        return f"{str(experiment)}_{str(graph)}_{str(retriever)}_{str(dataset)}_{current_time}"