"""
Iterative refinement engine for personalized LLM evaluation.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time

from models.llm_interface import LLMInterface, LLMOutput
from models.preference_embedder import PreferenceEmbedder, EditPair
from refinement.judge import Judge, MetaJudge
from refinement.preference_inference import PreferenceInferenceModule
from refinement.feedback_controller import FeedbackController
from config import RefinementConfig


@dataclass
class RefinementIteration:
    """Single iteration in the refinement process"""
    iteration: int
    prompt: str
    generation: LLMOutput
    inferred_preferences: Dict[str, Any]
    judge_feedback: Dict[str, Any]
    meta_judge_feedback: Optional[Dict[str, Any]] = None
    convergence_metrics: Optional[Dict[str, Any]] = None
    should_continue: bool = True


@dataclass 
class RefinementResult:
    """Complete refinement process result"""
    initial_prompt: str
    iterations: List[RefinementIteration]
    final_generation: LLMOutput
    total_iterations: int
    converged: bool
    convergence_reason: str
    total_time: float
    metrics: Dict[str, Any]


class RefinementEngine:
    """
    Main engine for iterative prompt-based refinement.
    
    Implements the loop:
    1. Generate initial response
    2. Infer user preferences from edit history  
    3. Judge evaluation of generation vs preferences
    4. Meta-judge evaluation of judge quality
    5. Update prompt based on feedback
    6. Repeat until convergence
    """
    
    def __init__(self, 
                 llm: LLMInterface,
                 preference_embedder: PreferenceEmbedder,
                 config: RefinementConfig):
        self.llm = llm
        self.preference_embedder = preference_embedder
        self.config = config
        
        # Initialize components
        self.judge = Judge(llm, config.judge_model)
        self.meta_judge = MetaJudge(llm, config.meta_judge_model) if config.use_meta_judge else None
        self.preference_inference = PreferenceInferenceModule(preference_embedder)
        self.feedback_controller = FeedbackController(config)
        
        logging.info("Initialized RefinementEngine")
    
    def refine(self, 
               initial_prompt: str,
               user_edit_history: List[EditPair],
               user_id: Optional[str] = None,
               context: Optional[Dict[str, Any]] = None) -> RefinementResult:
        """
        Execute complete refinement process
        
        Args:
            initial_prompt: Starting prompt for generation
            user_edit_history: Historical edits for preference inference
            user_id: Optional user identifier
            context: Additional context information
        
        Returns:
            Complete refinement result
        """
        start_time = time.time()
        iterations = []
        current_prompt = initial_prompt
        
        logging.info(f"Starting refinement process for prompt: {initial_prompt[:100]}...")
        
        for iteration in range(self.config.max_iterations):
            logging.info(f"Refinement iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Step 1: Generate response
            generation = self.llm.generate(current_prompt)
            
            # Step 2: Infer preferences from edit history
            inferred_preferences = self.preference_inference.infer_preferences(
                user_edit_history, user_id, context
            )
            
            # Step 3: Judge evaluation
            # Convert preferences to JSON-safe format
            safe_preferences = self._make_preferences_json_safe(inferred_preferences)
            judge_feedback = self.judge.evaluate_alignment(
                prompt=current_prompt,
                generation=generation.text,
                preferences=safe_preferences,
                context=context
            )
            
            # Step 4: Meta-judge evaluation (if enabled)
            meta_judge_feedback = None
            if self.meta_judge:
                meta_judge_feedback = self.meta_judge.evaluate_judge(
                    original_prompt=current_prompt,
                    generation=generation.text,
                    judge_feedback=judge_feedback,
                    preferences=safe_preferences
                )
            
            # Step 5: Compute convergence metrics
            convergence_metrics = self._compute_convergence_metrics(
                generation, judge_feedback, iterations
            )
            
            # Step 6: Check convergence
            should_continue = self._should_continue_refinement(
                convergence_metrics, iteration
            )
            
            # Create iteration record
            # Store the safe preferences to avoid serialization issues later
            iteration_record = RefinementIteration(
                iteration=iteration + 1,
                prompt=current_prompt,
                generation=generation,
                inferred_preferences=safe_preferences,
                judge_feedback=judge_feedback,
                meta_judge_feedback=meta_judge_feedback,
                convergence_metrics=convergence_metrics,
                should_continue=should_continue
            )
            iterations.append(iteration_record)
            
            # Step 7: Check if we should stop
            if not should_continue:
                convergence_reason = self._get_convergence_reason(convergence_metrics)
                logging.info(f"Convergence achieved: {convergence_reason}")
                break
            
            # Step 8: Update prompt for next iteration
            current_prompt = self.feedback_controller.update_prompt(
                current_prompt=current_prompt,
                judge_feedback=judge_feedback,
                meta_judge_feedback=meta_judge_feedback,
                preferences=safe_preferences,
                iteration=iteration + 1
            )
        
        # Create final result
        total_time = time.time() - start_time
        final_iteration = iterations[-1] if iterations else None
        
        result = RefinementResult(
            initial_prompt=initial_prompt,
            iterations=iterations,
            final_generation=final_iteration.generation if final_iteration else None,
            total_iterations=len(iterations),
            converged=not final_iteration.should_continue if final_iteration else False,
            convergence_reason=self._get_convergence_reason(
                final_iteration.convergence_metrics if final_iteration else {}
            ),
            total_time=total_time,
            metrics=self._compute_final_metrics(iterations)
        )
        
        logging.info(f"Refinement completed in {result.total_iterations} iterations ({total_time:.2f}s)")
        return result
    
    def _compute_convergence_metrics(self, 
                                   generation: LLMOutput,
                                   judge_feedback: Dict[str, Any],
                                   previous_iterations: List[RefinementIteration]) -> Dict[str, Any]:
        """Compute metrics to determine convergence"""
        metrics = {}
        
        # Judge-based metrics
        alignment_score = judge_feedback.get('alignment_score', 0.0)
        confidence_score = judge_feedback.get('confidence', 0.0)
        
        metrics['alignment_score'] = alignment_score
        metrics['confidence_score'] = confidence_score
        
        # Text similarity metrics (if we have previous iterations)
        if previous_iterations:
            prev_generation = previous_iterations[-1].generation.text
            current_generation = generation.text
            
            # Compute edit distance
            import editdistance
            edit_distance = editdistance.eval(prev_generation, current_generation)
            metrics['edit_distance'] = edit_distance
            
            # Compute normalized edit distance
            max_len = max(len(prev_generation), len(current_generation))
            normalized_edit_distance = edit_distance / max_len if max_len > 0 else 0
            metrics['normalized_edit_distance'] = normalized_edit_distance
            
            # Compute BERTScore if available
            try:
                from bert_score import score
                _, _, bertscore = score([current_generation], [prev_generation], lang="en")
                metrics['bertscore'] = bertscore.item()
            except ImportError:
                logging.warning("BERTScore not available, skipping")
                metrics['bertscore'] = 0.0
        
        return metrics
    
    def _should_continue_refinement(self, 
                                  convergence_metrics: Dict[str, Any],
                                  iteration: int) -> bool:
        """Determine if refinement should continue"""
        # Check maximum iterations
        if iteration >= self.config.max_iterations - 1:
            return False
        
        # Check alignment score threshold
        alignment_score = convergence_metrics.get('alignment_score', 0.0)
        if alignment_score >= self.config.convergence_threshold:
            return False
        
        # Check edit distance threshold
        edit_distance = convergence_metrics.get('edit_distance', float('inf'))
        if edit_distance <= self.config.min_edit_distance:
            return False
        
        # Check BERTScore threshold
        bertscore = convergence_metrics.get('bertscore', 0.0)
        if bertscore >= self.config.convergence_threshold:
            return False
        
        return True
    
    def _get_convergence_reason(self, convergence_metrics: Dict[str, Any]) -> str:
        """Get human-readable convergence reason"""
        alignment_score = convergence_metrics.get('alignment_score', 0.0)
        edit_distance = convergence_metrics.get('edit_distance', float('inf'))
        bertscore = convergence_metrics.get('bertscore', 0.0)
        
        if alignment_score >= self.config.convergence_threshold:
            return f"High alignment score: {alignment_score:.3f}"
        elif edit_distance <= self.config.min_edit_distance:
            return f"Low edit distance: {edit_distance}"
        elif bertscore >= self.config.convergence_threshold:
            return f"High BERTScore: {bertscore:.3f}"
        else:
            return "Maximum iterations reached"
    
    def _compute_final_metrics(self, iterations: List[RefinementIteration]) -> Dict[str, Any]:
        """Compute final summary metrics"""
        if not iterations:
            return {}
        
        metrics = {}
        
        # Alignment progression
        alignment_scores = [iter.judge_feedback.get('alignment_score', 0.0) 
                           for iter in iterations]
        metrics['initial_alignment'] = alignment_scores[0] if alignment_scores else 0.0
        metrics['final_alignment'] = alignment_scores[-1] if alignment_scores else 0.0
        metrics['alignment_improvement'] = metrics['final_alignment'] - metrics['initial_alignment']
        
        # Confidence progression
        confidence_scores = [iter.judge_feedback.get('confidence', 0.0) 
                            for iter in iterations]
        metrics['final_confidence'] = confidence_scores[-1] if confidence_scores else 0.0
        
        # Edit distance progression
        edit_distances = [iter.convergence_metrics.get('edit_distance', 0) 
                         for iter in iterations[1:]]  # Skip first iteration
        if edit_distances:
            metrics['avg_edit_distance'] = sum(edit_distances) / len(edit_distances)
            metrics['final_edit_distance'] = edit_distances[-1]
        
        return metrics
    
    def _make_preferences_json_safe(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Convert preferences containing InferredPreference objects to JSON-safe format"""
        safe_prefs = {}
        
        for key, value in preferences.items():
            if key == 'structured_preferences' and isinstance(value, dict):
                # Convert InferredPreference objects to dictionaries
                safe_structured = {}
                for pref_key, pref_obj in value.items():
                    if hasattr(pref_obj, 'category') and hasattr(pref_obj, 'description'):
                        # This is an InferredPreference object
                        safe_structured[pref_key] = {
                            'category': pref_obj.category,
                            'description': pref_obj.description,
                            'confidence': getattr(pref_obj, 'confidence', 0.0),
                            'supporting_edits': getattr(pref_obj, 'supporting_edits', [])
                        }
                    else:
                        safe_structured[pref_key] = value
                safe_prefs[key] = safe_structured
            elif key == 'preference_embedding' and hasattr(value, 'tolist'):
                # Convert numpy arrays to lists
                safe_prefs[key] = value.tolist()
            else:
                # Keep other values as-is
                safe_prefs[key] = value
        
        return safe_prefs
    
    def batch_refine(self, 
                    prompts: List[str],
                    user_edit_histories: List[List[EditPair]],
                    user_ids: Optional[List[str]] = None,
                    contexts: Optional[List[Dict[str, Any]]] = None) -> List[RefinementResult]:
        """Run refinement on multiple prompts"""
        results = []
        
        for i, (prompt, edit_history) in enumerate(zip(prompts, user_edit_histories)):
            user_id = user_ids[i] if user_ids else None
            context = contexts[i] if contexts else None
            
            result = self.refine(prompt, edit_history, user_id, context)
            results.append(result)
        
        return results 