"""
Logits steering module for personalized text generation at decoding time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging


@dataclass
class SteeringParams:
    """Parameters for logits steering"""
    alpha: float = 0.1  # Steering strength
    beta: float = 0.05  # Decay factor
    min_alpha: float = 0.01  # Minimum steering strength
    adaptive: bool = True  # Whether to use adaptive steering
    token_level: bool = False  # Whether to apply token-level steering


class LogitsSteerer(nn.Module):
    """
    Module for steering LLM generation using preference embeddings.
    Implements decoding-time intervention by modifying logits.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 preference_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 device: str = "cpu"):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.preference_dim = preference_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Preference-to-logits projection network
        layers = []
        layers.append(nn.Linear(preference_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        self.preference_projector = nn.Sequential(*layers)
        
        # Adaptive steering parameters
        self.steering_controller = nn.Sequential(
            nn.Linear(preference_dim + vocab_size, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        logging.info(f"Initialized LogitsSteerer with vocab_size={vocab_size}")
    
    def compute_steering_vector(self, preference_embedding: torch.Tensor) -> torch.Tensor:
        """Compute steering vector from preference embedding"""
        if preference_embedding.dim() == 1:
            preference_embedding = preference_embedding.unsqueeze(0)
        
        steering_vector = self.preference_projector(preference_embedding)
        return steering_vector
    
    def adaptive_steering_strength(self, 
                                 preference_embedding: torch.Tensor,
                                 original_logits: torch.Tensor,
                                 base_alpha: float = 0.1) -> torch.Tensor:
        """Compute adaptive steering strength based on context"""
        # Concatenate preference and logits for context
        context = torch.cat([preference_embedding, original_logits], dim=-1)
        
        # Compute adaptive strength
        adaptive_alpha = self.steering_controller(context) * base_alpha
        
        return adaptive_alpha
    
    def steer_logits(self, 
                    original_logits: torch.Tensor,
                    preference_embedding: torch.Tensor,
                    params: SteeringParams = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply preference-based steering to logits
        
        Args:
            original_logits: Original model logits [batch_size, vocab_size]
            preference_embedding: User preference embedding [batch_size, preference_dim]
            params: Steering parameters
            
        Returns:
            Steered logits and metadata
        """
        if params is None:
            params = SteeringParams()
        
        batch_size = original_logits.shape[0]
        
        # Compute steering vector
        steering_vector = self.compute_steering_vector(preference_embedding)
        
        # Compute steering strength
        if params.adaptive:
            alpha = self.adaptive_steering_strength(
                preference_embedding, original_logits, params.alpha
            )
        else:
            alpha = torch.tensor(params.alpha, device=self.device)
        
        # Apply steering
        steered_logits = original_logits + alpha.unsqueeze(-1) * steering_vector
        
        # Compute metadata
        alpha_np = alpha.detach().cpu().numpy()
        steering_norm_np = torch.norm(steering_vector, dim=-1).detach().cpu().numpy()
        logits_change_np = torch.norm(steered_logits - original_logits, dim=-1).detach().cpu().numpy()
        max_steering_token_np = torch.argmax(torch.abs(steering_vector), dim=-1).detach().cpu().numpy()
        
        metadata = {
            'steering_strength': alpha_np if alpha_np.ndim > 0 else alpha_np.item(),
            'steering_norm': steering_norm_np if steering_norm_np.ndim > 0 else steering_norm_np.item(),
            'logits_change': logits_change_np if logits_change_np.ndim > 0 else logits_change_np.item(),
            'max_steering_token': max_steering_token_np if max_steering_token_np.ndim > 0 else max_steering_token_np.item()
        }
        
        return steered_logits, metadata
    
    def progressive_steering(self, 
                           original_logits: torch.Tensor,
                           preference_embedding: torch.Tensor,
                           generation_step: int,
                           params: SteeringParams = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply progressive steering that decays over generation steps"""
        if params is None:
            params = SteeringParams()
        
        # Compute decay factor
        decay = max(params.min_alpha / params.alpha, 
                   np.exp(-params.beta * generation_step))
        
        # Create modified params with decayed alpha
        decayed_params = SteeringParams(
            alpha=params.alpha * decay,
            beta=params.beta,
            min_alpha=params.min_alpha,
            adaptive=params.adaptive,
            token_level=params.token_level
        )
        
        steered_logits, metadata = self.steer_logits(
            original_logits, preference_embedding, decayed_params
        )
        
        metadata['decay_factor'] = float(decay)
        metadata['generation_step'] = int(generation_step)
        
        return steered_logits, metadata
    
    def token_level_steering(self, 
                           original_logits: torch.Tensor,
                           preference_embedding: torch.Tensor,
                           input_tokens: torch.Tensor,
                           params: SteeringParams = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply token-level steering based on input context"""
        if params is None:
            params = SteeringParams()
        
        # Simple token-level implementation
        # In practice, this could be more sophisticated
        steered_logits, metadata = self.steer_logits(
            original_logits, preference_embedding, params
        )
        
        metadata['token_level'] = True
        metadata['input_length'] = input_tokens.shape[-1] if input_tokens is not None else 0
        
        return steered_logits, metadata
    
    def evaluate_steering_effect(self, 
                               original_logits: torch.Tensor,
                               steered_logits: torch.Tensor,
                               top_k: int = 10) -> Dict[str, Any]:
        """Evaluate the effect of steering on token probabilities"""
        # Convert logits to probabilities
        orig_probs = F.softmax(original_logits, dim=-1)
        steered_probs = F.softmax(steered_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(steered_probs.log(), orig_probs, reduction='batchmean')
        
        # Get top-k token changes
        prob_diff = steered_probs - orig_probs
        top_k_changes = torch.topk(torch.abs(prob_diff), top_k, dim=-1)
        
        # Compute entropy changes
        orig_entropy = -torch.sum(orig_probs * torch.log(orig_probs + 1e-10), dim=-1)
        steered_entropy = -torch.sum(steered_probs * torch.log(steered_probs + 1e-10), dim=-1)
        entropy_change = steered_entropy - orig_entropy
        
        max_prob_change_tensor = torch.max(torch.abs(prob_diff), dim=-1)[0].detach().cpu().numpy()
        entropy_change_np = entropy_change.detach().cpu().numpy()
        
        return {
            'kl_divergence': kl_div.item(),
            'top_k_token_changes': top_k_changes.indices.detach().cpu().numpy(),
            'top_k_prob_changes': top_k_changes.values.detach().cpu().numpy(),
            'entropy_change': entropy_change_np if entropy_change_np.ndim > 0 else entropy_change_np.item(),
            'max_prob_change': max_prob_change_tensor if max_prob_change_tensor.ndim > 0 else max_prob_change_tensor.item()
        }
    
    def calibrate_steering_strength(self, 
                                  preference_embeddings: List[torch.Tensor],
                                  target_kl_divergence: float = 0.1,
                                  num_iterations: int = 10) -> float:
        """Calibrate steering strength to achieve target KL divergence"""
        # Simple binary search for optimal alpha
        alpha_low, alpha_high = 0.001, 1.0
        best_alpha = 0.1
        
        for _ in range(num_iterations):
            alpha_mid = (alpha_low + alpha_high) / 2
            
            # Test on sample data
            total_kl = 0
            num_samples = min(len(preference_embeddings), 10)
            
            for i in range(num_samples):
                # Generate random logits for testing
                test_logits = torch.randn(1, self.vocab_size, device=self.device)
                
                steered_logits, _ = self.steer_logits(
                    test_logits, 
                    preference_embeddings[i].unsqueeze(0),
                    SteeringParams(alpha=alpha_mid, adaptive=False)
                )
                
                eval_results = self.evaluate_steering_effect(test_logits, steered_logits)
                total_kl += eval_results['kl_divergence']
            
            avg_kl = total_kl / num_samples
            
            if avg_kl < target_kl_divergence:
                alpha_low = alpha_mid
                best_alpha = alpha_mid
            else:
                alpha_high = alpha_mid
        
        logging.info(f"Calibrated steering strength: {best_alpha:.4f}")
        return best_alpha
    
    def save_model(self, path: str) -> None:
        """Save model state"""
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'preference_dim': self.preference_dim,
            'hidden_dim': self.hidden_dim
        }, path)
        logging.info(f"Saved LogitsSteerer to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> 'LogitsSteerer':
        """Load model from saved state"""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            preference_dim=checkpoint['preference_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            device=device
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Loaded LogitsSteerer from {path}")
        return model 