"""
Configuration management for personalized LLM evaluation system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class LLMConfig:
    """Configuration for Language Model"""
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class RefinementConfig:
    """Configuration for iterative refinement loop"""
    max_iterations: int = 3
    convergence_threshold: float = 0.95  # BERTScore threshold for convergence
    min_edit_distance: int = 5  # Minimum edit distance to continue refinement
    use_meta_judge: bool = True
    judge_model: str = "gpt-4o-mini"
    meta_judge_model: str = "gpt-4o-mini"


@dataclass
class PreferenceConfig:
    """Configuration for preference learning and steering"""
    embedding_dim: int = 768
    encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    alpha: float = 0.1  # Logits steering strength
    use_contrastive_learning: bool = True
    similarity_threshold: float = 0.8


@dataclass
class DataConfig:
    """Configuration for dataset and evaluation"""
    dataset_name: str = "LongLaMP"
    num_examples: int = 100
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    max_sequence_length: int = 2048


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and protocols"""
    metrics: List[str] = None
    use_bertscore: bool = True
    use_edit_distance: bool = True
    use_rouge: bool = True
    use_bleu: bool = True
    compute_preference_alignment: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["bertscore", "edit_distance", "rouge", "bleu", "preference_alignment"]


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    experiment_name: str = "personalized_llm_eval"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    save_intermediate: bool = True
    use_wandb: bool = False
    wandb_project: str = "personalized-llm-eval"
    
    # Component configs
    llm_config: LLMConfig = None
    refinement_config: RefinementConfig = None
    preference_config: PreferenceConfig = None
    data_config: DataConfig = None
    evaluation_config: EvaluationConfig = None
    
    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = LLMConfig()
        if self.refinement_config is None:
            self.refinement_config = RefinementConfig()
        if self.preference_config is None:
            self.preference_config = PreferenceConfig()
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig()
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    def create_output_dir(self) -> Path:
        """Create output directory if it doesn't exist"""
        output_path = Path(self.output_dir) / self.experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path 