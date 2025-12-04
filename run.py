"""
Graph Retriever Project using Abstract Base Classes
"""

import argparse
from graph import Graph
from graph.create_graph import create_graph
from retriever import Retriever
from retriever.create_retriever import create_retriever
from dataset import Dataset
from dataset.create_dataset import create_dataset
from experiment import Experiment
from experiment.create_experiment import create_experiment
from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Retriever Experiments')
    parser.add_argument('--graph', type=str, default='freebase',
                        choices=['freebase'],
                        help='Graph type to use')
    parser.add_argument('--retriever', type=str, default='gemini_baseline_retriever',
                        choices=['gemini_baseline_retriever'],
                        help='Retriever model to use')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic'],
                        help='Dataset to use')
    parser.add_argument('--experiment', type=str, default='standard',
                        choices=['standard'],
                        help='Experiment type to run')
    
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()    
    
    
    graph: Graph = create_graph(args.graph)
    retriever: Retriever = create_retriever(args.retriever)
    dataset: Dataset = create_dataset(args.dataset)
    experiment: Experiment = create_experiment(args.experiment)
    
    metrics = experiment.run(graph, retriever, dataset)
    print("Experiment Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()