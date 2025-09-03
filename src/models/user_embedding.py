import torch
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

from ..data.data_types import UserData, Profile

logger = logging.getLogger(__name__)

class UserEmbedding:
    """Class for creating and managing user embeddings from profile data."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize the user embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Cache for computed embeddings
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        
    def compute_user_embedding(self, user_data: UserData) -> torch.Tensor:
        """
        Compute user embedding from their profile data.
        
        Args:
            user_data: UserData containing profiles
            
        Returns:
            User embedding tensor of shape (embedding_dim,)
        """
        cache_key = f"{user_data.user_id}_{user_data.task}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        profile_texts = user_data.get_profile_texts()
        
        if not profile_texts:
            logger.warning(f"No profile texts found for user {user_data.user_id}")
            # Return zero embedding if no profiles
            embedding_dim = self.model.get_sentence_embedding_dimension()
            embedding = torch.zeros(embedding_dim, dtype=torch.float32)
        else:
            # Encode all profile texts
            profile_embeddings = self.model.encode(
                profile_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Average the embeddings (simple strategy)
            embedding = torch.mean(profile_embeddings, dim=0)
            
        # Normalize the embedding
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        
        # Cache the result
        self.embedding_cache[cache_key] = embedding
        
        return embedding
        
    def compute_batch_embeddings(self, user_data_list: List[UserData]) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for a batch of users efficiently.
        
        Args:
            user_data_list: List of UserData objects
            
        Returns:
            Dictionary mapping user_id to embedding tensor
        """
        embeddings = {}
        
        logger.info(f"Computing embeddings for {len(user_data_list)} users")
        
        for user_data in tqdm(user_data_list, desc="Computing user embeddings"):
            embeddings[user_data.user_id] = self.compute_user_embedding(user_data)
            
        return embeddings
        
    def compute_text_embedding(self, text: str) -> torch.Tensor:
        """
        Compute embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding tensor
        """
        embedding = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        return torch.nn.functional.normalize(embedding, p=2, dim=0)
        
    def compute_token_similarities(self, user_embedding: torch.Tensor, 
                                 token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarities between user embedding and token embeddings.
        
        Args:
            user_embedding: User embedding tensor of shape (embedding_dim,)
            token_embeddings: Token embeddings tensor of shape (vocab_size, embedding_dim)
            
        Returns:
            Similarity scores tensor of shape (vocab_size,)
        """
        # Ensure user_embedding has the right shape
        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)  # (1, embedding_dim)
            
        # Compute cosine similarity
        similarities = torch.cosine_similarity(user_embedding, token_embeddings, dim=1)
        
        return similarities
        
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()
        
    def save_embeddings(self, embeddings: Dict[str, torch.Tensor], filepath: str) -> None:
        """Save computed embeddings to disk."""
        torch.save(embeddings, filepath)
        logger.info(f"Saved embeddings to {filepath}")
        
    def load_embeddings(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load embeddings from disk."""
        embeddings = torch.load(filepath, map_location=self.device)
        self.embedding_cache.update(embeddings)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
