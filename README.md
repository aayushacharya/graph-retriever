# Graph Retriever

A flexible and extensible graph-based retrieval system built with Python's Abstract Base Classes, providing a clean interface-driven architecture similar to Go.

## Quickstart

### Prerequisites

- Python 3.13+
- Virtuoso server and Freebase setup according to [GrailQA](https://github.com/dki-lab/GrailQA/tree/main).

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

- `--graph`: Graph implementation (`freebase`)
- `--retriever`: Retrieval model (`gemini_baseline_retriever`)
- `--dataset`: Dataset to use (`synthetic`)
- `--experiment`: Experiment type (`standard`)

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
├── test_grailqa.py             # Test script for GrailQA
├── sample_dataset.json         # Sample dataset file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment variables example
├── .env                        # Environment variables (local)
├── data/                       # Data directory
├── dataset/                    # Dataset module
│   ├── __init__.py             # Dataset abstract base class
│   ├── create_dataset.py       # Dataset factory function
│   └── custom_dataset.py       # Custom dataset implementation
├── db/                         # Database directory
├── experiment/                 # Experiment module
│   ├── __init__.py             # Experiment abstract base class
│   ├── create_experiment.py    # Experiment factory function
│   └── custom_experiment.py    # Custom experiment implementation
├── graph/                      # Graph module
│   ├── __init__.py             # Graph abstract base class
│   ├── create_graph.py         # Graph factory function
│   └── custom_graph.py         # Custom graph implementation
├── ontology/                   # Ontology directory
├── output/                     # Output directory
├── retriever/                  # Retriever module
│   ├── __init__.py             # Retriever abstract base class
│   ├── create_retriever.py     # Retriever factory function
│   └── custom_retriever.py     # Custom retriever implementation
└── utils/                      # Utils directory
```

### How to Contribute

If you want to add a new `Dataset`, `Experiment`, `Graph`, or a new `Retriever`. Just create a new file in the relevant directory. E.g. For adding a new experiment called `MyCustomExperiment`, create a new file in `experiment/my_custom_experiment.py`. Import the base class `Experiment` as shown below and implement the base functions.

```python
class MyCustomExperiment(Experiment):
    """Custom experiment implementation"""
    
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
