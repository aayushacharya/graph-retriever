from typing import List, Dict, Any
import numpy as np
from experiment import Experiment
from graph import Graph
from retriever import Retriever
from dataset import Dataset

class StandardExperiment(Experiment):
    """Standard retrieval experiment"""
    
    def __init__(self, k: int = 10):
        self.k = k
        self.graph = None
        self.retriever = None
        self.dataset = None
    
    def setup(self, graph: Graph, retriever: Retriever, dataset: Dataset) -> None:
        self.graph = graph
        self.retriever = retriever
        self.dataset = dataset
        
        # Load dataset and build graph
        self.dataset.load()
        self.graph.build(self.dataset.get_graph_data())
        self.retriever.initialize(self.graph)
    
    def run(self) -> Dict[str, float]:
        queries = self.dataset.get_queries()
        ground_truth = self.dataset.get_ground_truth()
        
        predictions = []
        for query in queries:
            results = self.retriever.retrieve(query, self.k)
            pred_nodes = [node_id for node_id, _ in results]
            predictions.append(pred_nodes)
        
        metrics = self.evaluate(predictions, ground_truth)
        return metrics
    
    def evaluate(self, predictions: List[List[str]], 
                 ground_truth: Dict[int, List[str]]) -> Dict[str, float]:
        precision_scores = []
        recall_scores = []
        
        for i, pred in enumerate(predictions):
            if i in ground_truth:
                gt_set = set(ground_truth[i])
                pred_set = set(pred)
                
                if len(pred_set) > 0:
                    precision = len(gt_set & pred_set) / len(pred_set)
                    precision_scores.append(precision)
                
                if len(gt_set) > 0:
                    recall = len(gt_set & pred_set) / len(gt_set)
                    recall_scores.append(recall)
        
        metrics = {
            'precision': np.mean(precision_scores) if precision_scores else 0.0,
            'recall': np.mean(recall_scores) if recall_scores else 0.0,
        }
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
                metrics['precision'] + metrics['recall']
            )
        else:
            metrics['f1'] = 0.0
        
        return metrics