"""
Preference embedding module for learning user preferences from edit history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging


@dataclass
class EditPair:
    """Represents an original-edited text pair with metadata"""
    original: str
    edited: str
    user_id: Optional[str] = None
    task_type: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def compute_edit_distance(self) -> int:
        """Compute Levenshtein distance between original and edited text"""
        import editdistance
        return editdistance.eval(self.original, self.edited)


class PreferenceEmbedder(nn.Module):
    """
    Neural network for learning user preference embeddings from edit pairs.
    Uses contrastive learning to create embeddings that capture editing patterns.
    """
    
    def __init__(self, 
                 encoder_model: str = "sentence-transformers/all-mpnet-base-v2",
                 embedding_dim: int = 768,
                 preference_dim: int = 256,
                 temperature: float = 0.1,
                 device: str = "cpu"):
        super().__init__()
        
        self.device = device
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.preference_dim = preference_dim
        
        # Load pre-trained sentence transformer
        self.encoder = SentenceTransformer(encoder_model, device=device)
        self.encoder.eval()  # Freeze encoder initially
        
        # Preference projection layers
        self.preference_projector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # Concat original + edited
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, preference_dim),
            nn.LayerNorm(preference_dim)
        )
        
        # User-specific preference layers (optional)
        self.user_embeddings = nn.ParameterDict()
        
        self.to(device)
        logging.info(f"Initialized PreferenceEmbedder with {encoder_model}")
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using sentence transformer"""
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        return embeddings
    
    def compute_preference_embedding(self, original: str, edited: str) -> torch.Tensor:
        """Compute preference embedding from original-edited pair"""
        # Get embeddings for both texts
        orig_emb = self.encode_texts([original])
        edit_emb = self.encode_texts([edited])
        
        # Concatenate embeddings
        combined = torch.cat([orig_emb, edit_emb], dim=1)
        
        # Project to preference space
        preference_emb = self.preference_projector(combined)
        return preference_emb
    
    def forward(self, edit_pairs: List[EditPair]) -> Dict[str, torch.Tensor]:
        """Forward pass for a batch of edit pairs"""
        batch_size = len(edit_pairs)
        
        # Prepare texts for encoding
        originals = [pair.original for pair in edit_pairs]
        editeds = [pair.edited for pair in edit_pairs]
        
        # Get embeddings
        orig_embeddings = self.encode_texts(originals)
        edit_embeddings = self.encode_texts(editeds)
        
        # Combine and project
        combined = torch.cat([orig_embeddings, edit_embeddings], dim=1)
        preference_embeddings = self.preference_projector(combined)
        
        return {
            'preference_embeddings': preference_embeddings,
            'original_embeddings': orig_embeddings,
            'edited_embeddings': edit_embeddings
        }
    
    def contrastive_loss(self, edit_pairs: List[EditPair], 
                        positive_pairs: List[Tuple[int, int]],
                        negative_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """Compute contrastive loss for preference learning"""
        outputs = self.forward(edit_pairs)
        preference_embs = outputs['preference_embeddings']
        
        # Normalize embeddings
        preference_embs = F.normalize(preference_embs, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(preference_embs, preference_embs.T) / self.temperature
        
        # Positive pairs loss
        pos_loss = 0.0
        if positive_pairs:
            pos_indices = torch.tensor(positive_pairs, device=self.device)
            pos_sims = similarities[pos_indices[:, 0], pos_indices[:, 1]]
            pos_loss = -torch.log(torch.sigmoid(pos_sims)).mean()
        
        # Negative pairs loss
        neg_loss = 0.0
        if negative_pairs:
            neg_indices = torch.tensor(negative_pairs, device=self.device)
            neg_sims = similarities[neg_indices[:, 0], neg_indices[:, 1]]
            neg_loss = -torch.log(torch.sigmoid(-neg_sims)).mean()
        
        return pos_loss + neg_loss
    
    def add_user_embedding(self, user_id: str, embedding_dim: Optional[int] = None) -> None:
        """Add user-specific embedding parameter"""
        if embedding_dim is None:
            embedding_dim = self.preference_dim
        
        self.user_embeddings[user_id] = nn.Parameter(
            torch.randn(embedding_dim, device=self.device) * 0.1
        )
    
    def get_user_preference(self, user_id: str) -> Optional[torch.Tensor]:
        """Get user-specific preference embedding"""
        return self.user_embeddings.get(user_id)
    
    def infer_preference_from_edits(self, edit_pairs: List[EditPair], 
                                   user_id: Optional[str] = None) -> torch.Tensor:
        """Infer overall preference from multiple edit pairs"""
        if not edit_pairs:
            return torch.zeros(self.preference_dim, device=self.device)
        
        # Get embeddings for all pairs
        outputs = self.forward(edit_pairs)
        preference_embs = outputs['preference_embeddings']
        
        # Aggregate preferences (mean for now, could be learned)
        aggregated_pref = preference_embs.mean(dim=0)
        
        # Add user-specific component if available
        if user_id and user_id in self.user_embeddings:
            user_pref = self.user_embeddings[user_id]
            aggregated_pref = aggregated_pref + user_pref
        
        return aggregated_pref
    
    def similarity_retrieval(self, query_pair: EditPair, 
                           candidate_pairs: List[EditPair],
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve most similar edit pairs based on preference embeddings"""
        # Get query embedding
        query_emb = self.compute_preference_embedding(query_pair.original, query_pair.edited)
        
        # Get candidate embeddings
        candidate_embs = []
        for pair in candidate_pairs:
            emb = self.compute_preference_embedding(pair.original, pair.edited)
            candidate_embs.append(emb)
        
        if not candidate_embs:
            return []
        
        candidate_embs = torch.stack(candidate_embs)
        
        # Compute similarities
        query_emb = F.normalize(query_emb, dim=1)
        candidate_embs = F.normalize(candidate_embs, dim=1)
        similarities = torch.matmul(query_emb, candidate_embs.T).squeeze()
        
        # Get top-k
        top_k = min(top_k, len(candidate_pairs))
        top_indices = torch.topk(similarities, top_k)[1]
        
        results = []
        for idx in top_indices:
            results.append((idx.item(), similarities[idx].item()))
        
        return results
    
    def save_model(self, path: str) -> None:
        """Save model state"""
        torch.save({
            'state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'preference_dim': self.preference_dim,
            'temperature': self.temperature
        }, path)
        logging.info(f"Saved PreferenceEmbedder to {path}")
    
    @classmethod
    def load_model(cls, path: str, encoder_model: str = None, device: str = "cpu") -> 'PreferenceEmbedder':
        """Load model from saved state"""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            encoder_model=encoder_model or "sentence-transformers/all-mpnet-base-v2",
            embedding_dim=checkpoint['embedding_dim'],
            preference_dim=checkpoint['preference_dim'],
            temperature=checkpoint['temperature'],
            device=device
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Loaded PreferenceEmbedder from {path}")
        return model 