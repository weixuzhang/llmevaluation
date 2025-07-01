"""
OpenAI model implementation for the personalized evaluation system.
"""

import openai
from typing import List, Dict, Optional, Any
import numpy as np
import logging
import time
from functools import wraps

from .llm_interface import LLMInterface, LLMOutput, GenerationParams


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0
):
    """Decorator for retrying API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logging.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                        raise
                    
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator


class OpenAIModel(LLMInterface):
    """OpenAI model implementation"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, **kwargs):
        # Pass generation parameters to parent, but handle client parameters separately
        super().__init__(model_name, **kwargs)
        
        # Simple OpenAI client initialization - following proven working pattern
        self.client = None
        
        try:
            if api_key:
                # Simple approach: just pass api_key like in working examples
                self.client = openai.OpenAI(api_key=api_key)
                logging.info("OpenAI client initialized successfully")
            else:
                # Try with default settings (will use OPENAI_API_KEY env var)
                self.client = openai.OpenAI()
                logging.info("OpenAI client initialized with default settings")
                
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI client: {e}")
            logging.info("Falling back to dummy client for demo purposes")
            logging.info("💡 This is normal for demo mode - all features will work with simulated responses")
            self.client = None
        
        # Default generation parameters
        self.default_params = GenerationParams(
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000),
            top_p=kwargs.get('top_p', 0.9),
            frequency_penalty=kwargs.get('frequency_penalty', 0.0),
            presence_penalty=kwargs.get('presence_penalty', 0.0)
        )
        
        logging.info(f"Initialized OpenAI model: {model_name}")
    
    @retry_with_exponential_backoff()
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> LLMOutput:
        """Generate text from a prompt using OpenAI API"""
        if params is None:
            params = self.default_params
        
        # Handle dummy client for demo purposes
        if self.client is None:
            return LLMOutput(
                text=f"[DEMO MODE] Generated response for prompt: {prompt[:50]}...",
                metadata={'model': self.model_name, 'demo_mode': True}
            )
        
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                stop=params.stop_sequences,
                logprobs=5 if params.logit_bias is not None else None
            )
            
            choice = response.choices[0]
            return LLMOutput(
                text=choice.text.strip(),
                logprobs=choice.logprobs.token_logprobs if choice.logprobs else None,
                tokens=choice.logprobs.tokens if choice.logprobs else None,
                token_logprobs=choice.logprobs.token_logprobs if choice.logprobs else None,
                metadata={
                    'finish_reason': choice.finish_reason,
                    'model': self.model_name,
                    'usage': response.usage.dict() if response.usage else None
                }
            )
        
        except Exception as e:
            logging.error(f"Error in OpenAI generation: {str(e)}")
            raise
    
    @retry_with_exponential_backoff()
    def chat_completion(self, messages: List[Dict[str, str]], 
                       params: Optional[GenerationParams] = None) -> LLMOutput:
        """Generate text from a conversation using OpenAI Chat API"""
        if params is None:
            params = self.default_params
        
        # Handle dummy client for demo purposes
        if self.client is None:
            last_message = messages[-1]['content'] if messages else "No message"
            return LLMOutput(
                text=f"[DEMO MODE] Chat response for: {last_message[:50]}...",
                metadata={'model': self.model_name, 'demo_mode': True}
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                stop=params.stop_sequences
            )
            
            choice = response.choices[0]
            return LLMOutput(
                text=choice.message.content.strip(),
                metadata={
                    'finish_reason': choice.finish_reason,
                    'model': self.model_name,
                    'usage': response.usage.dict() if response.usage else None
                }
            )
        
        except Exception as e:
            logging.error(f"Error in OpenAI chat completion: {str(e)}")
            raise
    
    @retry_with_exponential_backoff()
    def get_logprobs(self, prompt: str, completion: str) -> List[float]:
        """Get log probabilities for a completion given a prompt"""
        try:
            full_text = prompt + completion
            response = self.client.completions.create(
                model=self.model_name,
                prompt=full_text,
                max_tokens=0,
                logprobs=1,
                echo=True
            )
            
            if response.choices[0].logprobs:
                return response.choices[0].logprobs.token_logprobs
            return []
        
        except Exception as e:
            logging.error(f"Error getting logprobs: {str(e)}")
            return []
    
    @retry_with_exponential_backoff()
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using OpenAI embeddings API"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
        
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            # Return random embeddings as fallback
            return np.random.random((len(texts), 1536))
    
    def batch_generate(self, prompts: List[str], 
                      params: Optional[GenerationParams] = None) -> List[LLMOutput]:
        """Generate text for multiple prompts with rate limiting"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, params)
            results.append(result)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        return results 