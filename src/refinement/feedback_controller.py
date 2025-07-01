"""
Feedback controller for managing iterative prompt refinement.
"""

from typing import Dict, List, Optional, Any
import logging
import json

from ..config import RefinementConfig


class FeedbackController:
    """
    Controller for updating prompts based on judge and meta-judge feedback.
    
    Manages the prompt refinement strategy and incorporates feedback
    to improve subsequent generations.
    """
    
    def __init__(self, config: RefinementConfig):
        self.config = config
        self.refinement_strategies = {
            'preference_injection': self._inject_preference_guidance,
            'feedback_incorporation': self._incorporate_judge_feedback,
            'meta_correction': self._apply_meta_judge_corrections,
            'iterative_improvement': self._apply_iterative_improvements
        }
        logging.info("Initialized FeedbackController")
    
    def update_prompt(self, 
                     current_prompt: str,
                     judge_feedback: Dict[str, Any],
                     meta_judge_feedback: Optional[Dict[str, Any]],
                     preferences: Dict[str, Any],
                     iteration: int) -> str:
        """
        Update prompt based on feedback for next iteration.
        
        Args:
            current_prompt: Current prompt text
            judge_feedback: Feedback from judge evaluation  
            meta_judge_feedback: Optional meta-judge feedback
            preferences: Inferred user preferences
            iteration: Current iteration number
        
        Returns:
            Updated prompt for next iteration
        """
        logging.info(f"Updating prompt for iteration {iteration}")
        
        # Start with current prompt
        updated_prompt = current_prompt
        
        # Apply different refinement strategies based on feedback
        
        # 1. Inject preference guidance if preferences are clear
        if preferences.get('confidence', 0) > 0.5:
            updated_prompt = self._inject_preference_guidance(
                updated_prompt, preferences, judge_feedback
            )
        
        # 2. Incorporate judge feedback suggestions
        if judge_feedback.get('suggestions'):
            updated_prompt = self._incorporate_judge_feedback(
                updated_prompt, judge_feedback, iteration
            )
        
        # 3. Apply meta-judge corrections if available
        if meta_judge_feedback and meta_judge_feedback.get('revised_suggestions'):
            updated_prompt = self._apply_meta_judge_corrections(
                updated_prompt, meta_judge_feedback, judge_feedback
            )
        
        # 4. Apply iteration-specific improvements
        updated_prompt = self._apply_iterative_improvements(
            updated_prompt, iteration, judge_feedback
        )
        
        # 5. Ensure prompt quality and length
        updated_prompt = self._optimize_prompt_structure(updated_prompt)
        
        logging.info(f"Prompt updated: {len(current_prompt)} -> {len(updated_prompt)} chars")
        return updated_prompt
    
    def _inject_preference_guidance(self, 
                                  prompt: str,
                                  preferences: Dict[str, Any],
                                  judge_feedback: Dict[str, Any]) -> str:
        """Inject preference guidance into prompt"""
        
        # Extract preference summary
        pref_summary = preferences.get('preference_summary', '')
        structured_prefs = preferences.get('structured_preferences', {})
        
        if not pref_summary and not structured_prefs:
            return prompt
        
        # Create preference guidance section
        guidance_lines = []
        
        if structured_prefs:
            for pref_key, pref_info in structured_prefs.items():
                if hasattr(pref_info, 'description'):
                    guidance_lines.append(f"- {pref_info.description}")
                elif isinstance(pref_info, dict) and 'description' in pref_info:
                    guidance_lines.append(f"- {pref_info['description']}")
        
        if guidance_lines:
            preference_section = "\n".join([
                "\nBased on user preferences, please ensure your response:",
                *guidance_lines,
                ""
            ])
            
            # Insert preference guidance before the main request
            if "Please" in prompt:
                insertion_point = prompt.find("Please")
                updated_prompt = (prompt[:insertion_point] + 
                                preference_section + 
                                prompt[insertion_point:])
            else:
                updated_prompt = prompt + preference_section
                
            return updated_prompt
        
        return prompt
    
    def _incorporate_judge_feedback(self, 
                                  prompt: str,
                                  judge_feedback: Dict[str, Any],
                                  iteration: int) -> str:
        """Incorporate judge feedback suggestions into prompt"""
        
        suggestions = judge_feedback.get('suggestions', [])
        alignment_score = judge_feedback.get('alignment_score', 0.5)
        
        if not suggestions or alignment_score > 0.8:
            return prompt
        
        # Filter and prioritize suggestions
        actionable_suggestions = self._filter_actionable_suggestions(suggestions)
        
        if not actionable_suggestions:
            return prompt
        
        # Create feedback section
        feedback_section = "\n".join([
            "\nTo improve alignment with user preferences:",
            *[f"- {suggestion}" for suggestion in actionable_suggestions[:3]],
            ""
        ])
        
        # Add feedback section to prompt
        return prompt + feedback_section
    
    def _apply_meta_judge_corrections(self, 
                                    prompt: str,
                                    meta_judge_feedback: Dict[str, Any],
                                    judge_feedback: Dict[str, Any]) -> str:
        """Apply meta-judge corrections to prompt"""
        
        revised_suggestions = meta_judge_feedback.get('revised_suggestions', [])
        judge_quality = meta_judge_feedback.get('judge_quality_score', 0.5)
        
        # Only apply if meta-judge identified significant issues with judge feedback
        if judge_quality > 0.7 or not revised_suggestions:
            return prompt
        
        # Replace or supplement judge suggestions with meta-judge revisions
        correction_section = "\n".join([
            "\nRefined guidance based on deeper analysis:",
            *[f"- {suggestion}" for suggestion in revised_suggestions[:3]],
            ""
        ])
        
        return prompt + correction_section
    
    def _apply_iterative_improvements(self, 
                                    prompt: str,
                                    iteration: int,
                                    judge_feedback: Dict[str, Any]) -> str:
        """Apply iteration-specific improvements"""
        
        alignment_score = judge_feedback.get('alignment_score', 0.5)
        
        # Add iteration-specific instructions
        if iteration == 1:
            # First refinement - focus on major alignment issues
            if alignment_score < 0.4:
                iteration_guidance = "\nFocus on addressing the major preference misalignments identified above.\n"
            else:
                iteration_guidance = "\nRefine the response to better match the user's preferences.\n"
        
        elif iteration == 2:
            # Second refinement - fine-tuning
            if alignment_score < 0.6:
                iteration_guidance = "\nMake targeted improvements to better align with user preferences.\n"
            else:
                iteration_guidance = "\nMake final refinements to optimize user preference alignment.\n"
        
        else:
            # Later iterations - careful adjustments
            iteration_guidance = "\nMake careful, targeted adjustments to optimize alignment.\n"
        
        return prompt + iteration_guidance
    
    def _filter_actionable_suggestions(self, suggestions: List[str]) -> List[str]:
        """Filter suggestions to keep only actionable ones"""
        
        actionable_keywords = [
            'make', 'add', 'remove', 'change', 'use', 'include', 'focus',
            'emphasize', 'reduce', 'increase', 'simplify', 'expand'
        ]
        
        actionable = []
        for suggestion in suggestions:
            if any(keyword in suggestion.lower() for keyword in actionable_keywords):
                # Clean up the suggestion
                cleaned = suggestion.strip()
                if cleaned and not cleaned.startswith('Could not'):
                    actionable.append(cleaned)
        
        return actionable[:5]  # Limit to top 5 suggestions
    
    def _optimize_prompt_structure(self, prompt: str) -> str:
        """Optimize prompt structure and length"""
        
        # Remove excessive whitespace
        lines = [line.strip() for line in prompt.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Rejoin with proper spacing
        optimized = '\n'.join(lines)
        
        # Ensure prompt doesn't get too long
        max_length = 2000  # Reasonable prompt length limit
        if len(optimized) > max_length:
            # Truncate while preserving structure
            sentences = optimized.split('. ')
            truncated_sentences = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) + 2 <= max_length:
                    truncated_sentences.append(sentence)
                    current_length += len(sentence) + 2
                else:
                    break
            
            optimized = '. '.join(truncated_sentences)
            if not optimized.endswith('.'):
                optimized += '.'
        
        return optimized
    
    def get_refinement_history(self, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of refinement history"""
        
        if not iterations:
            return {}
        
        alignment_progression = []
        suggestion_counts = []
        
        for iteration in iterations:
            judge_feedback = iteration.get('judge_feedback', {})
            alignment_progression.append(judge_feedback.get('alignment_score', 0.0))
            suggestions = judge_feedback.get('suggestions', [])
            suggestion_counts.append(len(suggestions))
        
        return {
            'total_iterations': len(iterations),
            'alignment_progression': alignment_progression,
            'initial_alignment': alignment_progression[0] if alignment_progression else 0.0,
            'final_alignment': alignment_progression[-1] if alignment_progression else 0.0,
            'improvement': (alignment_progression[-1] - alignment_progression[0]) if len(alignment_progression) >= 2 else 0.0,
            'avg_suggestions_per_iteration': sum(suggestion_counts) / len(suggestion_counts) if suggestion_counts else 0,
            'converged': alignment_progression[-1] > 0.8 if alignment_progression else False
        }
    
    def analyze_feedback_patterns(self, feedback_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in feedback to improve future refinements"""
        
        if not feedback_history:
            return {}
        
        # Count common suggestion themes
        suggestion_themes = {}
        alignment_trends = []
        
        for feedback in feedback_history:
            alignment_trends.append(feedback.get('alignment_score', 0.5))
            
            suggestions = feedback.get('suggestions', [])
            for suggestion in suggestions:
                # Simple keyword extraction for themes
                suggestion_lower = suggestion.lower()
                if 'formal' in suggestion_lower:
                    suggestion_themes['formality'] = suggestion_themes.get('formality', 0) + 1
                elif 'length' in suggestion_lower or 'concise' in suggestion_lower:
                    suggestion_themes['length'] = suggestion_themes.get('length', 0) + 1
                elif 'detail' in suggestion_lower:
                    suggestion_themes['detail_level'] = suggestion_themes.get('detail_level', 0) + 1
                elif 'structure' in suggestion_lower or 'organize' in suggestion_lower:
                    suggestion_themes['structure'] = suggestion_themes.get('structure', 0) + 1
        
        return {
            'common_themes': dict(sorted(suggestion_themes.items(), key=lambda x: x[1], reverse=True)),
            'alignment_trend': 'improving' if len(alignment_trends) > 1 and alignment_trends[-1] > alignment_trends[0] else 'stable',
            'avg_alignment': sum(alignment_trends) / len(alignment_trends),
            'feedback_consistency': self._compute_feedback_consistency(feedback_history)
        }
    
    def _compute_feedback_consistency(self, feedback_history: List[Dict[str, Any]]) -> float:
        """Compute consistency of feedback across iterations"""
        if len(feedback_history) < 2:
            return 1.0
        
        # Simple consistency metric based on alignment score variance
        scores = [fb.get('alignment_score', 0.5) for fb in feedback_history]
        variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)
        
        # Convert variance to consistency (lower variance = higher consistency)
        consistency = max(0.0, 1.0 - variance * 4)  # Scale factor to normalize
        return consistency 