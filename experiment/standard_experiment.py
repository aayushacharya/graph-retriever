from typing import List, Dict, Any
import numpy as np
from experiment import Experiment
from graph import Graph
from retriever import Retriever
from dataset import Dataset
from experiment import Result, QAResult, get_report_name
import time
import json
import os

class StandardExperiment(Experiment):
    """Standard retrieval experiment"""
    
    def __init__(self, k: int = 10):
        self.experiment_name = "StandardExperiment"

    def __str__(self) -> str:
        return self.experiment_name


    def run(self, graph: Graph, retriever: Retriever, dataset: Dataset) -> Dict[str, float]:
        """Run the standard experiment"""
        # Get dev data
        dev_data = dataset.get_dev_data()
        aggregated_metrics: Dict[str, float] = {"ExactMatch": 0.0}
        exact_matches =[]
        qa_results: List[QAResult] = []
        result: Result = Result(qa_results=qa_results, aggregated_metrics=aggregated_metrics)
        for item in dev_data:
            question = item.question_text
            qa_result=QAResult(
                question_id=item.question_id,
                question_text=item.question_text,
                answers=item.answers,
                metadata=item.metadata,
                predicted_answers=[],
                metrics={}
            )
            # Retrieve generated_answers
            predictions = retriever.retrieve(question, graph)
            qa_result.predicted_answers = predictions
            exact_match = 1 if all(ans in predictions for ans in item.answers) else 0
            qa_result.metrics["ExactMatch"] = float(exact_match)
            qa_results.append(qa_result)
            exact_matches.append(exact_match)
        aggregated_metrics["ExactMatch"] = float(np.mean(exact_matches))
        result.qa_results = qa_results
        result.aggregated_metrics = aggregated_metrics
        # Dump results
        output_dir="output"
        report_name = get_report_name(self,graph, retriever, dataset)
        self.dump_results(result, os.path.join(output_dir, f"{report_name}.json"))
        return aggregated_metrics
    
    
    
    def dump_results(self, results: Result, output_path: str) -> None:
        """Dump experiment results to a file"""
        with open(output_path, 'w') as f:
            f.write(results.model_dump_json(indent=4))