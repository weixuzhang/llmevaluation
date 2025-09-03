import json
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")

def save_results(results: Dict[str, Any], save_dir: str, 
                experiment_name: Optional[str] = None) -> str:
    """
    Save experiment results to disk.
    
    Args:
        results: Dictionary containing results to save
        save_dir: Directory to save results
        experiment_name: Optional experiment name (timestamp used if None)
        
    Returns:
        Path to saved results file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Save as JSON for easy reading
    json_file = save_path / f"{experiment_name}.json"
    
    # Convert non-serializable objects to strings
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    with open(json_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save as pickle for complete object preservation
    pickle_file = save_path / f"{experiment_name}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {json_file} and {pickle_file}")
    return str(json_file)

def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from disk.
    
    Args:
        filepath: Path to results file (.json or .pkl)
        
    Returns:
        Dictionary containing loaded results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results loaded from {filepath}")
    return results

def create_experiment_config(task: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create experiment configuration dictionary.
    
    Args:
        task: LongLaMP task name
        model_name: HuggingFace model name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        'task': task,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'num_users': kwargs.get('num_users', -1),
        'max_new_tokens': kwargs.get('max_new_tokens', 256),
        'temperature': kwargs.get('temperature', 0.7),
        'top_p': kwargs.get('top_p', 0.9),
        'beta': kwargs.get('beta', 0.1),
        'embedding_model': kwargs.get('embedding_model', 'all-MiniLM-L6-v2'),
        'batch_size': kwargs.get('batch_size', 8),
        'seed': kwargs.get('seed', 42)
    }
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if key not in config:
            config[key] = value
    
    return config

def print_config(config: Dict[str, Any]) -> None:
    """Print experiment configuration in a formatted way."""
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    
    for key, value in config.items():
        print(f"{key:20}: {value}")
    
    print("="*50 + "\n")

def save_generated_samples(samples: Dict[str, Any], save_dir: str,
                          experiment_name: str, num_samples: int = 5) -> None:
    """
    Save example generated samples for inspection.
    
    Args:
        samples: Dictionary containing generated samples
        save_dir: Directory to save samples
        experiment_name: Experiment name
        num_samples: Number of samples to save
    """
    save_path = Path(save_dir) / f"{experiment_name}_samples.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Generated Samples - {experiment_name}\n")
        f.write("="*80 + "\n\n")
        
        for i in range(min(num_samples, len(samples.get('inputs', [])))):
            f.write(f"Sample {i+1}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Input: {samples['inputs'][i]}\n\n")
            f.write(f"Target: {samples['targets'][i]}\n\n")
            f.write(f"Baseline: {samples['baseline_outputs'][i]}\n\n")
            f.write(f"Personalized: {samples['personalized_outputs'][i]}\n\n")
            f.write("="*80 + "\n\n")
    
    logger.info(f"Sample outputs saved to {save_path}")
