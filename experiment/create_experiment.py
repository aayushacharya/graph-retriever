from experiment import Experiment
from experiment.standard_experiment import StandardExperiment

def create_experiment(experiment_type: str) -> Experiment:
    """Factory function to create experiment instances"""
    experiments = {
        'standard': StandardExperiment,
    }
    
    if experiment_type not in experiments:
        raise ValueError(f"Unknown experiment type: {experiment_type}. Available: {list(experiments.keys())}")
    
    return experiments[experiment_type]()