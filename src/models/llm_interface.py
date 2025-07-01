"""
Abstract interface for Language Models in the personalized evaluation system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LLMOutput:
    """Structured output from LLM calls"""
    text: str
    logprobs: Optional[List[float]] = None
    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def token_count(self) -> int:
        """Get token count"""
        return len(self.tokens) if self.tokens else len(self.text.split())


@dataclass
class GenerationParams:
    """Parameters for text generation"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    logit_bias: Optional[Dict[str, float]] = None


class LLMInterface(ABC):
    """Abstract interface for Language Models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        # Only store generation-related parameters, not client parameters
        self.generation_kwargs = {k: v for k, v in kwargs.items() 
                                if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']}
    
    @abstractmethod
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> LLMOutput:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                       params: Optional[GenerationParams] = None) -> LLMOutput:
        """Generate text from a conversation"""
        pass
    
    @abstractmethod
    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Get log probabilities for a completion given a prompt"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        pass
    
    def batch_generate(self, prompts: List[str], 
                      params: Optional[GenerationParams] = None) -> List[LLMOutput]:
        """Generate text for multiple prompts (default implementation)"""
        return [self.generate(prompt, params) for prompt in prompts]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def prepare_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Prepare messages for chat completion"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages 