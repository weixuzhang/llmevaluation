# LLM Personalization with User Embeddings
# Main package initialization

__version__ = "0.1.0"
__author__ = "LLM Personalization Research"
__description__ = "Personalized text generation using user embeddings and logit adjustment"

from . import data
from . import models
from . import evaluation
from . import utils

__all__ = ['data', 'models', 'evaluation', 'utils']
