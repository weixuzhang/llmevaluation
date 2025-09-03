#!/usr/bin/env python3
"""
Demo script showing how to use the personalization system.
This script demonstrates the key components without running a full experiment.
"""

import torch
from src.data import load_longlamp_data
from src.data.data_types import ExperimentConfig
from src.models import UserEmbedding, PersonalizedGenerator
from src.evaluation import PersonalizationEvaluator
from src.utils import setup_logging, set_seed

def demo_user_embeddings():
    """Demonstrate user embedding functionality."""
    print("=== User Embedding Demo ===")
    
    # Create a sample user with profiles
    from src.data.data_types import UserData
    
    user_data = UserData(
        user_id="demo_user",
        input_text="Write an abstract for a paper about machine learning.",
        profiles=[
            {"title": "Deep Learning for NLP", "abstract": "This paper explores deep learning methods for natural language processing..."},
            {"title": "Transformer Models", "abstract": "We present a comprehensive study of transformer architectures..."},
        ],
        target_output="This paper presents a novel approach to machine learning...",
        task="LongLaMP-2"
    )
    
    # Initialize user embedding model
    user_embedding = UserEmbedding(model_name='all-MiniLM-L6-v2')
    
    # Compute user embedding
    embedding = user_embedding.compute_user_embedding(user_data)
    print(f"User embedding shape: {embedding.shape}")
    print(f"User embedding norm: {torch.norm(embedding):.4f}")
    
    # Show profile texts
    profile_texts = user_data.get_profile_texts()
    print(f"Number of profile texts: {len(profile_texts)}")
    for i, text in enumerate(profile_texts):
        print(f"Profile {i+1}: {text[:100]}...")
    
    print()

def demo_generation():
    """Demonstrate personalized generation (simplified)."""
    print("=== Generation Demo ===")
    
    # Note: This is a simplified demo that doesn't actually run generation
    # due to computational requirements
    
    config = ExperimentConfig(
        task="LongLaMP-2",
        model_name="microsoft/DialoGPT-small",
        num_users=1,
        max_new_tokens=50,
        beta=0.1
    )
    
    print(f"Configuration:")
    print(f"- Task: {config.task}")
    print(f"- Model: {config.model_name}")
    print(f"- Personalization strength (beta): {config.beta}")
    print(f"- Max new tokens: {config.max_new_tokens}")
    
    print("Note: Actual generation requires GPU/significant compute time.")
    print("Use main.py for full experiments.")
    print()

def demo_evaluation():
    """Demonstrate evaluation functionality."""
    print("=== Evaluation Demo ===")
    
    # Sample predictions and references
    baseline_predictions = [
        "This paper presents a method for machine learning.",
        "We propose a new approach to data analysis."
    ]
    
    personalized_predictions = [
        "This paper presents a novel deep learning method for natural language processing.",
        "We propose a transformer-based approach to personalized data analysis."
    ]
    
    references = [
        "This paper presents a novel approach to machine learning using deep neural networks.",
        "We propose a new transformer-based approach to personalized data analysis."
    ]
    
    # Evaluate
    evaluator = PersonalizationEvaluator()
    results = evaluator.compare_methods(
        baseline_predictions=baseline_predictions,
        personalized_predictions=personalized_predictions,
        references=references
    )
    
    evaluator.print_comparison_results(results)

def demo_data_loading():
    """Demonstrate data loading functionality."""
    print("=== Data Loading Demo ===")
    
    try:
        config = ExperimentConfig(
            task="LongLaMP-2",
            model_name="dummy",
            num_users=2  # Just load 2 users for demo
        )
        
        print("Loading LongLaMP dataset (this may take a moment)...")
        dataset = load_longlamp_data(config, split='test')
        
        print(f"Loaded {len(dataset)} users")
        
        # Show first user
        if len(dataset) > 0:
            user_data = dataset[0]
            print(f"\nSample user data:")
            print(f"- User ID: {user_data.user_id}")
            print(f"- Input: {user_data.input_text[:100]}...")
            print(f"- Target: {user_data.target_output[:100]}...")
            print(f"- Number of profiles: {len(user_data.profiles)}")
            
            if user_data.profiles:
                profile_sample = user_data.profiles[0]
                print(f"- Sample profile keys: {list(profile_sample.keys())}")
        
    except Exception as e:
        print(f"Note: Could not load dataset - {e}")
        print("This is normal if you haven't downloaded the dataset yet.")
        print("Run: python scripts/download_data.py")
    
    print()

def main():
    """Run all demos."""
    setup_logging("INFO")
    set_seed(42)
    
    print("LLM Personalization System Demo")
    print("================================")
    print()
    
    # Run demos
    demo_data_loading()
    demo_user_embeddings()
    demo_generation()
    demo_evaluation()
    
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Download data: python scripts/download_data.py")
    print("2. Quick test: python scripts/run_quick_experiment.py")
    print("3. Full experiment: python main.py --task LongLaMP-2 --model_name microsoft/DialoGPT-small")

if __name__ == "__main__":
    main()
