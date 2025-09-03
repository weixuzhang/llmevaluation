import evaluate
import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    rouge1: float
    rougeL: float
    meteor: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rouge-1': self.rouge1,
            'rouge-L': self.rougeL, 
            'meteor': self.meteor
        }
        
    def __str__(self) -> str:
        return f"ROUGE-1: {self.rouge1:.4f}, ROUGE-L: {self.rougeL:.4f}, METEOR: {self.meteor:.4f}"

class PersonalizationEvaluator:
    """Evaluator for personalization experiments using ROUGE and METEOR metrics."""
    
    def __init__(self):
        """Initialize the evaluator with ROUGE and METEOR metrics."""
        try:
            self.rouge_metric = evaluate.load('rouge')
            self.meteor_metric = evaluate.load('meteor')
            logger.info("Loaded ROUGE and METEOR metrics")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            raise
            
    def evaluate_generation(self, predictions: List[str], 
                          references: List[str]) -> EvaluationResult:
        """
        Evaluate generated texts against references.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            EvaluationResult containing ROUGE and METEOR scores
        """
        if len(predictions) != len(references):
            raise ValueError(f"Predictions and references must have same length: "
                           f"{len(predictions)} vs {len(references)}")
                           
        # Prepare data for evaluation
        stripped_predictions = [pred.strip() for pred in predictions]
        stripped_references = [ref.strip() for ref in references]
        reference_lists = [[ref] for ref in stripped_references]
        
        # Compute ROUGE scores
        rouge_results = self.rouge_metric.compute(
            predictions=stripped_predictions,
            references=reference_lists,
            rouge_types=['rouge1', 'rougeL']
        )
        
        # Compute METEOR score
        meteor_results = self.meteor_metric.compute(
            predictions=stripped_predictions,
            references=reference_lists
        )
        
        return EvaluationResult(
            rouge1=rouge_results['rouge1'],
            rougeL=rouge_results['rougeL'],
            meteor=meteor_results['meteor']
        )
        
    def compare_methods(self, baseline_predictions: List[str],
                       personalized_predictions: List[str],
                       references: List[str]) -> Dict[str, EvaluationResult]:
        """
        Compare baseline and personalized generation methods.
        
        Args:
            baseline_predictions: Predictions from baseline model
            personalized_predictions: Predictions from personalized model
            references: Reference texts
            
        Returns:
            Dictionary with evaluation results for both methods
        """
        logger.info("Evaluating baseline predictions...")
        baseline_results = self.evaluate_generation(baseline_predictions, references)
        
        logger.info("Evaluating personalized predictions...")
        personalized_results = self.evaluate_generation(personalized_predictions, references)
        
        # Calculate improvements
        improvement = EvaluationResult(
            rouge1=personalized_results.rouge1 - baseline_results.rouge1,
            rougeL=personalized_results.rougeL - baseline_results.rougeL,
            meteor=personalized_results.meteor - baseline_results.meteor
        )
        
        return {
            'baseline': baseline_results,
            'personalized': personalized_results,
            'improvement': improvement
        }
        
    def print_comparison_results(self, results: Dict[str, EvaluationResult]) -> None:
        """Print formatted comparison results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nBaseline:      {results['baseline']}")
        print(f"Personalized:  {results['personalized']}")
        print(f"Improvement:   {results['improvement']}")
        
        # Calculate relative improvements
        baseline = results['baseline']
        improvement = results['improvement']
        
        rouge1_rel = (improvement.rouge1 / baseline.rouge1 * 100) if baseline.rouge1 > 0 else 0
        rougeL_rel = (improvement.rougeL / baseline.rougeL * 100) if baseline.rougeL > 0 else 0
        meteor_rel = (improvement.meteor / baseline.meteor * 100) if baseline.meteor > 0 else 0
        
        print(f"\nRelative Improvements:")
        print(f"ROUGE-1: {rouge1_rel:+.2f}%")
        print(f"ROUGE-L: {rougeL_rel:+.2f}%")
        print(f"METEOR:  {meteor_rel:+.2f}%")
        print("="*60)


def compute_rouge_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Convenience function to compute ROUGE and METEOR scores.
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        
    Returns:
        Dictionary with metric scores
    """
    evaluator = PersonalizationEvaluator()
    results = evaluator.evaluate_generation(predictions, references)
    return results.to_dict()
