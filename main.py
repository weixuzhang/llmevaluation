#!/usr/bin/env python3
"""
Main experiment script for LLM personalization with user embeddings.

This script implements the full pipeline:
1. Load LongLaMP dataset
2. Extract user embeddings from profiles
3. Generate texts with and without personalization
4. Evaluate using ROUGE and METEOR metrics
5. Report results and save outputs
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.data import LongLaMPDataset, load_longlamp_data
from src.data.data_types import ExperimentConfig
from src.models import PersonalizedGenerator
from src.evaluation import PersonalizationEvaluator
from src.utils import setup_logging, save_results, set_seed, create_experiment_config, print_config, save_generated_samples

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM personalization experiments with user embeddings"
    )
    
    # Required arguments
    parser.add_argument("--task", type=str, required=True,
                       choices=['LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4'],
                       help="LongLaMP task to run")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name (e.g., 'microsoft/DialoGPT-small')")
    
    # Optional arguments
    parser.add_argument("--num_users", type=int, default=10,
                       help="Number of users to use (-1 for all users)")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Personalization strength parameter")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Directory to save results")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level")
    parser.add_argument("--save_samples", action="store_true",
                       help="Save example generated samples")
    
    return parser.parse_args()

def run_experiment(config: ExperimentConfig, output_dir: str, 
                  experiment_name: str, save_samples: bool = False) -> Dict[str, Any]:
    """
    Run the full personalization experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        experiment_name: Name of the experiment
        save_samples: Whether to save example samples
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("Starting personalization experiment")
    
    # 1. Load dataset
    logger.info(f"Loading {config.task} dataset...")
    dataset = load_longlamp_data(config, split='test')
    logger.info(f"Loaded {len(dataset)} users")
    
    # 2. Initialize personalized generator
    logger.info("Initializing personalized generator...")
    generator = PersonalizedGenerator(config)
    
    # 3. Generate texts with and without personalization
    logger.info("Generating baseline texts (no personalization)...")
    baseline_outputs = generator.generate_batch(dataset.data, use_personalization=False)
    
    logger.info("Generating personalized texts...")
    personalized_outputs = generator.generate_batch(dataset.data, use_personalization=True)
    
    # 4. Extract inputs and targets
    inputs = [user_data.input_text for user_data in dataset.data]
    targets = [user_data.target_output for user_data in dataset.data]
    
    # 5. Evaluate results
    logger.info("Evaluating results...")
    evaluator = PersonalizationEvaluator()
    evaluation_results = evaluator.compare_methods(
        baseline_predictions=baseline_outputs,
        personalized_predictions=personalized_outputs,
        references=targets
    )
    
    # 6. Print results
    evaluator.print_comparison_results(evaluation_results)
    
    # 7. Prepare results dictionary
    results = {
        'config': config.__dict__,
        'evaluation': {
            'baseline': evaluation_results['baseline'].to_dict(),
            'personalized': evaluation_results['personalized'].to_dict(),
            'improvement': evaluation_results['improvement'].to_dict()
        },
        'samples': {
            'inputs': inputs,
            'targets': targets,
            'baseline_outputs': baseline_outputs,
            'personalized_outputs': personalized_outputs
        },
        'num_users': len(dataset.data),
        'num_tokens_generated': {
            'baseline': sum(len(output.split()) for output in baseline_outputs),
            'personalized': sum(len(output.split()) for output in personalized_outputs)
        }
    }
    
    # 8. Save results
    logger.info("Saving results...")
    save_results(results, output_dir, experiment_name)
    
    # 9. Save sample outputs if requested
    if save_samples:
        save_generated_samples(results['samples'], output_dir, experiment_name)
    
    logger.info("Experiment completed successfully!")
    return results

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_file = f"{args.output_dir}/logs/{args.experiment_name or 'experiment'}.log"
    setup_logging(args.log_level, log_file)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment configuration
    config = ExperimentConfig(
        task=args.task,
        model_name=args.model_name,
        num_users=args.num_users,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        embedding_model=args.embedding_model,
        seed=args.seed
    )
    
    # Print configuration
    print_config(config.__dict__)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.task}_{args.model_name.replace('/', '_')}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    try:
        # Run experiment
        results = run_experiment(
            config=config,
            output_dir=args.output_dir,
            experiment_name=experiment_name,
            save_samples=args.save_samples
        )
        
        # Print final summary
        improvement = results['evaluation']['improvement']
        print(f"\nExperiment completed!")
        print(f"ROUGE-1 improvement: {improvement['rouge-1']:+.4f}")
        print(f"ROUGE-L improvement: {improvement['rouge-L']:+.4f}")
        print(f"METEOR improvement: {improvement['meteor']:+.4f}")
        print(f"Results saved to: {args.output_dir}/{experiment_name}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
