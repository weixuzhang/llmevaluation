from typing import Dict, List, Any, TypeAlias
from dataclasses import dataclass

# Type aliases for better readability
Profile: TypeAlias = Dict[str, str]

@dataclass
class UserData:
    """Container for user's data including input, profiles, and target output."""
    user_id: str
    input_text: str
    profiles: List[Profile]
    target_output: str
    task: str
    
    def get_profile_texts(self) -> List[str]:
        """Extract text content from profiles based on task type."""
        texts = []
        for profile in self.profiles:
            if self.task == 'LongLaMP-2':  # Abstract generation
                text = f"{profile.get('title', '')} {profile.get('abstract', '')}"
            elif self.task == 'LongLaMP-3':  # Topic writing
                text = f"{profile.get('content', '')} {profile.get('summary', '')}"
            elif self.task == 'LongLaMP-4':  # Product review
                text = f"{profile.get('overall', '')} {profile.get('summary', '')} {profile.get('description', '')} {profile.get('reviewText', '')}"
            else:
                # Default: concatenate all profile values
                text = ' '.join(str(v) for v in profile.values() if v)
            
            texts.append(text.strip())
        
        return [text for text in texts if text]  # Filter out empty texts

@dataclass
class ExperimentConfig:
    """Configuration for personalization experiments."""
    task: str
    model_name: str
    num_users: int = -1  # -1 means use all users
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0.1  # Personalization strength
    embedding_model: str = 'all-MiniLM-L6-v2'
    batch_size: int = 8
    seed: int = 42
