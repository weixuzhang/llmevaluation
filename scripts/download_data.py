#!/usr/bin/env python3
"""
Script to download and verify the LongLaMP dataset.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging

logger = logging.getLogger(__name__)

def download_longlamp_dataset():
    """Download all LongLaMP datasets."""
    
    tasks = {
        'LongLaMP-2': 'abstract_generation_user',
        'LongLaMP-3': 'topic_writing_user', 
        'LongLaMP-4': 'product_review_user'
    }
    
    print("Downloading LongLaMP datasets...")
    print("This may take a few minutes...")
    
    for task_name, hf_name in tasks.items():
        try:
            logger.info(f"Downloading {task_name} ({hf_name})...")
            
            # Download the dataset
            dataset = load_dataset('LongLaMP/LongLaMP', name=hf_name)
            
            print(f"SUCCESS {task_name}: {len(dataset['test'])} test examples")
            
            # Print sample to verify
            sample = dataset['test'][0]
            print(f"   Sample input: {sample['input'][:100]}...")
            print(f"   Sample output: {sample['output'][:100]}...")
            print(f"   Number of profiles: {len(sample['profile'])}")
            print()
            
        except Exception as e:
            logger.error(f"Failed to download {task_name}: {e}")
            print(f"FAILED to download {task_name}")
            
    print("Dataset download completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run experiment: python main.py --task LongLaMP-2 --model_name microsoft/DialoGPT-small")

def verify_dataset_access():
    """Verify that we can access the datasets."""
    tasks = ['LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4']
    
    print("Verifying dataset access...")
    
    for task in tasks:
        try:
            # Try to load a small sample
            from src.data import load_longlamp_data
            from src.data.data_types import ExperimentConfig
            
            config = ExperimentConfig(
                task=task,
                model_name="dummy",
                num_users=1
            )
            
            dataset = load_longlamp_data(config, split='test')
            print(f"SUCCESS {task}: Successfully loaded {len(dataset)} examples")
            
        except Exception as e:
            print(f"FAILED {task}: Failed to load - {e}")

if __name__ == "__main__":
    setup_logging("INFO")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_dataset_access()
    else:
        download_longlamp_dataset()
