# Graph Retriever

A flexible and extensible graph-based retrieval system built with Python's Abstract Base Classes, providing a clean interface-driven architecture similar to Go.

## Quickstart

### Prerequisites

- Python 3.13+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd graph-retriever

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### Basic Usage

Run a simple experiment with default settings:

```bash
python run.py
```

This will run a cosine similarity retriever on a simple graph with synthetic data.

#### Command Line Options

```bash
python run.py --graph <graph_type> --retriever <retriever_type> --dataset <dataset_type> --k <num_results>
```

**Available Options:**

- `--graph`: Graph implementation (`simple`)
- `--retriever`: Retrieval model (`cosine`)
- `--dataset`: Dataset to use (`synthetic`)
- `--experiment`: Experiment type (`standard`)
- `--k`: Number of results to retrieve (default: 10)

#### Example Commands

```bash

# Use simple graph with cosine retriever
python run.py --graph simple --retriever cosine --k 5


```

## Contributing

We welcome contributions! This project is designed to be easily extensible through its interface-based architecture.

### Project Structure

```
graph-retriever/
├── run.py                      # Main entry point
├── dataset/
│   ├── __init__.py             # Dataset abstract base class
│   ├── create_dataset.py       # Dataset factory function
│   └── custom_dataset.py       # Custom dataset implementation
├── experiment/
│   ├── __init__.py             # Experiment abstract base class
│   ├── create_experiment.py    # Experiment factory function
│   └── custom_experiment.py    # Custom experiment implementation
├── graph/
│   ├── __init__.py             # Graph abstract base class
│   ├── create_graph.py         # Graph factory function
│   └── custom_graph.py         # Custom graph implementation
├── retriever/
│   ├── __init__.py             # Retriever abstract base class
│   └── create_retriever.py     # Retriever factory function
├── .gitignore
└── README.md
```

### How to Contribute

If you want to add a new `Dataset`, `Experiment`, `Graph`, or a new `Retriever`. Just create a new file in the relevant directory. E.g. For adding a new experiment called `MyCustomExperiment`, create a new file in `experiment/my_custom_experiment.py`. Import the base class `Experiment` as shown below and implement the base functions.

```python
class MyCustomExperiment(Experiment):
    def __init__(self):
        # Initialize experiment parameters
        pass

    def setup(self, graph: Graph, retriever: Retriever, dataset: Dataset) -> None:
        # Setup experiment components
        pass

    def run(self) -> Dict[str, float]:
        # Run experiment and return metrics
        pass

    def evaluate(self, predictions: List[List[str]],
                 ground_truth: Dict[int, List[str]]) -> Dict[str, float]:
        # Evaluate predictions and return metrics
        pass
```

After you have implemented the new function, make sure to register it in `experiment/create_experiment.py`. Finally add the name of your experiment in the parser choices in `run.py`.

### Contribution Guidelines

1. **Follow the Interface**: Always implement all abstract methods defined in the base classes
2. **Type Hints**: Use type hints for all method signatures
3. **Documentation**: Add docstrings explaining your implementation
4. **Testing**: Test your implementation with the synthetic dataset before submitting
5. **Code Style**: Follow PEP 8 style guidelines
6. **Error Handling**: Add appropriate error handling for edge cases
7. **Commit conventions**: Use commit conventions described [here](https://www.conventionalcommits.org/en/v1.0.0/#specification).

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-new-feature`)
3. Implement your changes following the guidelines above
4. Test your implementation thoroughly
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/my-new-feature`)
7. Create a Pull Request with a clear description of your changes
