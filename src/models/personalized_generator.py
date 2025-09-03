import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import logging
from tqdm import tqdm

from .user_embedding import UserEmbedding
from ..data.data_types import UserData, ExperimentConfig

logger = logging.getLogger(__name__)

class PersonalizedGenerator:
    """
    Personalized text generator that adjusts LLM logits based on user embeddings.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the personalized generator.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize user embedding model
        self.user_embedding = UserEmbedding(
            model_name=config.embedding_model,
            device=self.device
        )
        
                # Pre-compute token embeddings for efficiency
        self._precompute_token_embeddings()
        
    def _precompute_token_embeddings(self) -> None:
        """Pre-compute embeddings for all tokens in the vocabulary."""
        logger.info("Pre-computing token embeddings...")
        
        vocab_size = len(self.tokenizer)
        batch_size = 1000  # Process tokens in batches
        
        all_token_embeddings = []
        
        for i in tqdm(range(0, vocab_size, batch_size), desc="Computing token embeddings"):
            batch_end = min(i + batch_size, vocab_size)
            token_ids = list(range(i, batch_end))
            
            # Decode tokens to text
            token_texts = []
            for token_id in token_ids:
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    # Clean up the token text
                    token_text = token_text.strip()
                    if not token_text:
                        token_text = f"<token_{token_id}>"
                    token_texts.append(token_text)
                except:
                    token_texts.append(f"<token_{token_id}>")
            
            # Compute embeddings for this batch
            batch_embeddings = self.user_embedding.model.encode(
                token_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            all_token_embeddings.append(batch_embeddings)
            
        # Concatenate all embeddings
        self.token_embeddings = torch.cat(all_token_embeddings, dim=0)
        logger.info(f"Token embeddings shape: {self.token_embeddings.shape}")
        
    def generate_personalized(self, user_data: UserData, 
                            use_personalization: bool = True) -> str:
        """
        Generate personalized text for a user.
        
        Args:
            user_data: User data containing input and profiles
            use_personalization: Whether to apply personalization
            
        Returns:
            Generated text
        """
        # Prepare input
        input_text = user_data.input_text
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Compute user embedding if personalization is enabled
        user_embedding = None
        if use_personalization:
            user_embedding = self.user_embedding.compute_user_embedding(user_data)
            user_embedding = user_embedding.to(self.device)
            
        # Generate with custom logits processor
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Custom generation with logit modification
        generated_ids = self._generate_with_personalization(
            inputs['input_ids'],
            user_embedding,
            generation_config
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
        
    def _generate_with_personalization(self, input_ids: torch.Tensor,
                                     user_embedding: Optional[torch.Tensor],
                                     generation_config: GenerationConfig) -> torch.Tensor:
        """
        Generate text with personalized logit adjustment.
        
        Args:
            input_ids: Input token IDs
            user_embedding: User embedding (None for baseline)
            generation_config: Generation configuration
            
        Returns:
            Generated token IDs
        """
        generated_ids = input_ids.clone()
        
        for _ in range(generation_config.max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Apply personalization if user embedding is provided
                if user_embedding is not None:
                    personalized_logits = self._apply_personalization(
                        logits, user_embedding
                    )
                else:
                    personalized_logits = logits
                    
                # Apply temperature
                if generation_config.temperature != 1.0:
                    personalized_logits = personalized_logits / generation_config.temperature
                    
                # Apply top-p sampling
                if generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(personalized_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    personalized_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(personalized_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
        return generated_ids
        
    def _apply_personalization(self, logits: torch.Tensor, 
                             user_embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply personalization by adjusting logits based on user embedding similarity.
        
        Args:
            logits: Original logits from the model
            user_embedding: User embedding
            
        Returns:
            Personalized logits
        """
        # Compute similarities between user embedding and token embeddings
        similarities = self.user_embedding.compute_token_similarities(
            user_embedding, self.token_embeddings
        )
        
        # Ensure similarities are on the same device as logits
        similarities = similarities.to(logits.device)
        
        # Apply personalization: logits + beta * similarity
        personalized_logits = logits + self.config.beta * similarities
        
        return personalized_logits
        
    def generate_batch(self, user_data_list: List[UserData], 
                      use_personalization: bool = True) -> List[str]:
        """
        Generate text for a batch of users.
        
        Args:
            user_data_list: List of user data
            use_personalization: Whether to apply personalization
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for user_data in tqdm(user_data_list, desc="Generating texts"):
            generated_text = self.generate_personalized(
                user_data, use_personalization=use_personalization
            )
            generated_texts.append(generated_text)
            
        return generated_texts
