import os
import json
from typing import List, Optional, Iterator
import logging
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import random

from .data_types import UserData, Profile, ExperimentConfig

logger = logging.getLogger(__name__)

class LongLaMPDataset:
    """Dataset class for LongLaMP benchmark."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data: List[UserData] = []
        self.task_to_hf_name = {
            'LongLaMP-2': 'abstract_generation_user',
            'LongLaMP-3': 'topic_writing_user', 
            'LongLaMP-4': 'product_review_user'
        }
        
    def load_data(self, split: str = 'test') -> None:
        """Load data from HuggingFace datasets."""
        if self.config.task not in self.task_to_hf_name:
            raise ValueError(f"Unsupported task: {self.config.task}")
            
        logger.info(f"Loading {self.config.task} dataset, split: {split}")
        
        hf_name = self.task_to_hf_name[self.config.task]
        dataset = load_dataset('LongLaMP/LongLaMP', name=hf_name, split=split)
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Convert to our UserData format
        for i, row in enumerate(dataset):
            user_data = UserData(
                user_id=str(i),
                input_text=row['input'],
                profiles=row['profile'],
                target_output=row['output'],
                task=self.config.task
            )
            self.data.append(user_data)
            
        # Limit number of users if specified
        if self.config.num_users > 0 and len(self.data) > self.config.num_users:
            # Use random sampling to get diverse users
            random.seed(self.config.seed)
            self.data = random.sample(self.data, self.config.num_users)
            logger.info(f"Sampled {len(self.data)} users for experiment")
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> UserData:
        return self.data[idx]
        
    def __iter__(self) -> Iterator[UserData]:
        return iter(self.data)
        
    def get_user_profiles(self, user_id: str) -> List[Profile]:
        """Get profiles for a specific user."""
        for user_data in self.data:
            if user_data.user_id == user_id:
                return user_data.profiles
        raise ValueError(f"User {user_id} not found")
        
    def get_all_profile_texts(self) -> List[List[str]]:
        """Get all profile texts for all users."""
        return [user_data.get_profile_texts() for user_data in self.data]


def load_longlamp_data(config: ExperimentConfig, split: str = 'test') -> LongLaMPDataset:
    """
    Convenience function to load LongLaMP data.
    
    Args:
        config: Experiment configuration
        split: Dataset split ('test', 'train', 'dev')
        
    Returns:
        LongLaMPDataset instance with loaded data
    """
    dataset = LongLaMPDataset(config)
    dataset.load_data(split)
    return dataset


def create_dataloader(dataset: LongLaMPDataset, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    """Create a PyTorch DataLoader from LongLaMPDataset."""
    
    def collate_fn(batch: List[UserData]) -> dict:
        """Custom collate function for batching UserData objects."""
        return {
            'user_ids': [item.user_id for item in batch],
            'input_texts': [item.input_text for item in batch], 
            'profiles': [item.profiles for item in batch],
            'target_outputs': [item.target_output for item in batch],
            'tasks': [item.task for item in batch]
        }
    
    return DataLoader(
        dataset.data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
