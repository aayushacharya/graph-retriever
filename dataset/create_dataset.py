from dataset import Dataset
from dataset.synthetic_dataset import SyntheticDataset
from dataset.grailqa_dataset import GrailQADataset

def create_dataset(dataset_type: str) -> Dataset:
    """Factory function to create dataset instances"""
    datasets = {
        'synthetic': SyntheticDataset,
        'grailqa': GrailQADataset,
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_type]()