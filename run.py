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

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Retriever Experiments')
    parser.add_argument('--graph', type=str, default='simple',
                        choices=['simple'],
                        help='Graph type to use')
    parser.add_argument('--retriever', type=str, default='cosine',
                        choices=['cosine'],
                        help='Retriever model to use')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic'],
                        help='Dataset to use')
    parser.add_argument('--experiment', type=str, default='standard',
                        choices=['standard'],
                        help='Experiment type to run')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of results to retrieve')
    
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()    
    
    
    graph: Graph = create_graph(args.graph)
    retriever: Retriever = create_retriever(args.retriever)
    dataset: Dataset = create_dataset(args.dataset)
    experiment: Experiment = create_experiment(args.experiment)
    
    experiment.setup(graph, retriever, dataset)
    
    metrics = experiment.run()
    print("Experiment Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()