"""
Preference inference module for extracting user preferences from edit history.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from collections import Counter

from models.preference_embedder import PreferenceEmbedder, EditPair


@dataclass
class InferredPreference:
    """Represents an inferred user preference"""
    category: str  # e.g., "style", "content", "format"
    description: str  # Human-readable description
    confidence: float  # 0-1 confidence score
    supporting_edits: List[int]  # Indices of supporting edit pairs
    embedding: Optional[np.ndarray] = None  # Optional embedding representation


class PreferenceInferenceModule:
    """
    Module for inferring user preferences from edit history.
    
    Uses preference embeddings to identify patterns in user edits
    and extract structured preference information.
    """
    
    def __init__(self, preference_embedder: PreferenceEmbedder):
        self.preference_embedder = preference_embedder
        self.preference_categories = {
            'style': ['tone', 'formality', 'voice', 'perspective'],
            'content': ['topic_focus', 'detail_level', 'examples', 'structure'],
            'format': ['length', 'organization', 'bullet_points', 'paragraphs'],
            'language': ['complexity', 'vocabulary', 'technical_terms', 'clarity']
        }
        logging.info("Initialized PreferenceInferenceModule")
    
    def infer_preferences(self, 
                         edit_history: List[EditPair],
                         user_id: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Infer user preferences from edit history.
        
        Args:
            edit_history: List of original-edited text pairs
            user_id: Optional user identifier
            context: Additional context information
        
        Returns:
            Dictionary containing inferred preferences
        """
        if not edit_history:
            return self._create_empty_preferences()
        
        logging.info(f"Inferring preferences from {len(edit_history)} edit pairs")
        
        # Get preference embeddings for all edit pairs
        preference_embeddings = []
        for pair in edit_history:
            embedding = self.preference_embedder.compute_preference_embedding(
                pair.original, pair.edited
            )
            preference_embeddings.append(embedding)
        
        # Cluster similar edits to identify preference patterns
        preference_clusters = self._cluster_preferences(preference_embeddings, edit_history)
        
        # Extract structured preferences from clusters
        inferred_preferences = self._extract_preferences_from_clusters(
            preference_clusters, edit_history
        )
        
        # Add user-specific preferences if available
        if user_id:
            user_preferences = self._get_user_specific_preferences(user_id)
            inferred_preferences.update(user_preferences)
        
        # Add contextual preferences
        if context:
            contextual_preferences = self._infer_contextual_preferences(context, inferred_preferences)
            inferred_preferences.update(contextual_preferences)
        
        # Compute overall preference embedding
        overall_embedding = self.preference_embedder.infer_preference_from_edits(
            edit_history, user_id
        )
        
        result = {
            'structured_preferences': inferred_preferences,
            'preference_embedding': overall_embedding.detach().cpu().numpy(),
            'confidence': self._compute_overall_confidence(inferred_preferences),
            'num_supporting_edits': len(edit_history),
            'preference_summary': self._create_preference_summary(inferred_preferences)
        }
        
        logging.info(f"Inferred {len(inferred_preferences)} preference categories")
        return result
    
    def _cluster_preferences(self, 
                           embeddings: List[np.ndarray],
                           edit_pairs: List[EditPair],
                           similarity_threshold: float = 0.7) -> List[List[int]]:
        """Cluster similar preference embeddings"""
        if len(embeddings) <= 1:
            return [[0]] if embeddings else []
        
        # Convert to numpy array for clustering
        embeddings_array = np.vstack([emb.detach().cpu().numpy() for emb in embeddings])
        
        # Simple clustering based on cosine similarity
        clusters = []
        assigned = set()
        
        for i, emb_i in enumerate(embeddings_array):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            for j, emb_j in enumerate(embeddings_array[i+1:], i+1):
                if j in assigned:
                    continue
                
                # Compute cosine similarity
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                
                if similarity >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _extract_preferences_from_clusters(self, 
                                         clusters: List[List[int]],
                                         edit_pairs: List[EditPair]) -> Dict[str, InferredPreference]:
        """Extract structured preferences from edit clusters"""
        preferences = {}
        
        for cluster_id, cluster_indices in enumerate(clusters):
            if len(cluster_indices) < 2:  # Skip singleton clusters
                continue
            
            # Analyze edits in this cluster
            cluster_pairs = [edit_pairs[i] for i in cluster_indices]
            
            # Extract common patterns
            preference = self._analyze_edit_cluster(cluster_pairs, cluster_indices)
            
            # Only include high-confidence preferences
            if preference.confidence >= 0.5:
                pref_key = f"{preference.category}_{cluster_id}"
                preferences[pref_key] = preference
        
        return preferences
    
    def _analyze_edit_cluster(self, 
                            edit_pairs: List[EditPair],
                            indices: List[int]) -> InferredPreference:
        """Analyze a cluster of similar edits to extract preference"""
        
        # Analyze text characteristics
        length_changes = []
        formality_changes = []
        complexity_changes = []
        
        for pair in edit_pairs:
            # Length analysis
            orig_len = len(pair.original.split())
            edit_len = len(pair.edited.split())
            length_changes.append(edit_len - orig_len)
            
            # Simple formality heuristics
            orig_formal = self._estimate_formality(pair.original)
            edit_formal = self._estimate_formality(pair.edited)
            formality_changes.append(edit_formal - orig_formal)
            
            # Complexity heuristics  
            orig_complex = self._estimate_complexity(pair.original)
            edit_complex = self._estimate_complexity(pair.edited)
            complexity_changes.append(edit_complex - orig_complex)
        
        # Determine dominant preference category and description
        category, description, confidence = self._categorize_preference_pattern(
            length_changes, formality_changes, complexity_changes
        )
        
        return InferredPreference(
            category=category,
            description=description,
            confidence=confidence,
            supporting_edits=indices
        )
    
    def _estimate_formality(self, text: str) -> float:
        """Estimate text formality (simple heuristic)"""
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'moreover']
        informal_indicators = ["don't", "won't", "can't", "it's", "you're"]
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.5
        
        formal_score = formal_count / total_words
        informal_score = informal_count / total_words
        
        return formal_score - informal_score + 0.5  # Normalize to 0-1
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity (simple heuristic)"""
        words = text.split()
        if not words:
            return 0.5
        
        # Average word length
        avg_word_len = sum(len(word) for word in words) / len(words)
        
        # Sentence length (approximate)
        sentences = text.split('.')
        avg_sentence_len = len(words) / len(sentences) if sentences else len(words)
        
        # Normalize to 0-1 scale
        complexity = (avg_word_len / 10 + avg_sentence_len / 20) / 2
        return min(1.0, complexity)
    
    def _categorize_preference_pattern(self, 
                                     length_changes: List[int],
                                     formality_changes: List[float],
                                     complexity_changes: List[float]) -> Tuple[str, str, float]:
        """Categorize preference pattern based on changes"""
        
        # Compute average changes
        avg_length = np.mean(length_changes) if length_changes else 0
        avg_formality = np.mean(formality_changes) if formality_changes else 0
        avg_complexity = np.mean(complexity_changes) if complexity_changes else 0
        
        # Determine category and confidence
        patterns = []
        
        # Length preferences
        if abs(avg_length) > 5:  # Significant length change
            if avg_length > 0:
                patterns.append(("format", "prefers longer, more detailed responses", 0.8))
            else:
                patterns.append(("format", "prefers shorter, more concise responses", 0.8))
        
        # Formality preferences
        if abs(avg_formality) > 0.1:
            if avg_formality > 0:
                patterns.append(("style", "prefers more formal language", 0.7))
            else:
                patterns.append(("style", "prefers more casual language", 0.7))
        
        # Complexity preferences
        if abs(avg_complexity) > 0.1:
            if avg_complexity > 0:
                patterns.append(("language", "prefers more complex vocabulary", 0.6))
            else:
                patterns.append(("language", "prefers simpler language", 0.6))
        
        # Return highest confidence pattern or default
        if patterns:
            return max(patterns, key=lambda x: x[2])
        else:
            return ("content", "general editing preference", 0.4)
    
    def _get_user_specific_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific preferences if available"""
        user_pref = self.preference_embedder.get_user_preference(user_id)
        if user_pref is not None:
            return {
                'user_specific': {
                    'user_id': user_id,
                    'has_personalized_model': True,
                    'embedding_norm': float(np.linalg.norm(user_pref.detach().cpu().numpy()))
                }
            }
        return {}
    
    def _infer_contextual_preferences(self, 
                                    context: Dict[str, Any],
                                    existing_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Infer additional preferences from context"""
        contextual = {}
        
        if 'task_type' in context:
            task_type = context['task_type']
            if task_type == 'email':
                contextual['task_specific'] = {
                    'type': 'email_writing',
                    'likely_preferences': ['professional_tone', 'clear_structure']
                }
            elif task_type == 'summarization':
                contextual['task_specific'] = {
                    'type': 'summarization',
                    'likely_preferences': ['brevity', 'key_points']
                }
        
        if 'domain' in context:
            contextual['domain_specific'] = {
                'domain': context['domain'],
                'specialized_vocabulary': context['domain'] in ['technical', 'academic', 'medical']
            }
        
        return contextual
    
    def _compute_overall_confidence(self, preferences: Dict[str, Any]) -> float:
        """Compute overall confidence in preference inference"""
        if not preferences:
            return 0.0
        
        confidences = []
        for pref in preferences.values():
            if isinstance(pref, InferredPreference):
                confidences.append(pref.confidence)
            elif isinstance(pref, dict) and 'confidence' in pref:
                confidences.append(pref['confidence'])
        
        return np.mean(confidences) if confidences else 0.5
    
    def _create_preference_summary(self, preferences: Dict[str, Any]) -> str:
        """Create human-readable summary of preferences"""
        if not preferences:
            return "No clear preferences identified from edit history."
        
        summaries = []
        for key, pref in preferences.items():
            if isinstance(pref, InferredPreference):
                summaries.append(f"- {pref.category.title()}: {pref.description}")
            elif isinstance(pref, dict) and 'description' in pref:
                summaries.append(f"- {key}: {pref['description']}")
        
        if summaries:
            return "Identified preferences:\n" + "\n".join(summaries)
        else:
            return "Preferences inferred but require further analysis."
    
    def _create_empty_preferences(self) -> Dict[str, Any]:
        """Create empty preferences structure when no edit history available"""
        return {
            'structured_preferences': {},
            'preference_embedding': np.zeros(self.preference_embedder.preference_dim),
            'confidence': 0.0,
            'num_supporting_edits': 0,
            'preference_summary': "No edit history available for preference inference."
        } 