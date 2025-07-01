"""
Demo script showing the complete personalized LLM evaluation system.

This demonstrates both Stage 1 (Iterative Refinement) and Stage 2 (Decoding-time Steering).
"""

import os
import logging
from typing import List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .config import ExperimentConfig, LLMConfig, RefinementConfig, PreferenceConfig
from .models.openai_model import OpenAIModel
from .models.preference_embedder import PreferenceEmbedder, EditPair
from .models.logits_steerer import LogitsSteerer, SteeringParams
from .refinement.refinement_engine import RefinementEngine


def create_sample_edit_history() -> List[EditPair]:
    """Create sample edit history for demonstration"""
    return [
        EditPair(
            original="The weather is nice today.",
            edited="Today's weather is absolutely beautiful and perfect for outdoor activities.",
            user_id="demo_user",
            task_type="general"
        ),
        EditPair(
            original="I like coding.",
            edited="I'm passionate about programming and software development.",
            user_id="demo_user", 
            task_type="general"
        ),
        EditPair(
            original="The book was good.",
            edited="The book was exceptionally well-written with compelling characters and an engaging plot.",
            user_id="demo_user",
            task_type="general"
        ),
        EditPair(
            original="Thanks for your help.",
            edited="Thank you so much for your invaluable assistance and support.",
            user_id="demo_user",
            task_type="general"
        )
    ]


def demo_stage1_iterative_refinement():
    """Demonstrate Stage 1: Iterative Refinement"""
    print("\n" + "="*60)
    print("🔄 STAGE 1: ITERATIVE REFINEMENT DEMO")
    print("="*60)
    
    # Create configuration
    config = ExperimentConfig(
        experiment_name="demo_stage1",
        llm_config=LLMConfig(
            model_name="gpt-4o-mini",
            temperature=0.7
        ),
        refinement_config=RefinementConfig(
            max_iterations=3,
            convergence_threshold=0.85,
            use_meta_judge=True
        ),
        preference_config=PreferenceConfig(
            embedding_dim=768,
            preference_dim=256
        )
    )
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        llm = OpenAIModel(
            model_name=config.llm_config.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=config.llm_config.temperature
        )
        
        preference_embedder = PreferenceEmbedder(
            embedding_dim=config.preference_config.embedding_dim,
            preference_dim=config.preference_config.preference_dim
        )
        
        refinement_engine = RefinementEngine(
            llm=llm,
            preference_embedder=preference_embedder,
            config=config.refinement_config
        )
        
        # Create sample data
        print("📝 Creating sample edit history...")
        edit_history = create_sample_edit_history()
        
        initial_prompt = """
        Write a professional email to a client about a project delay.
        The email should be apologetic but maintain confidence in the team's ability to deliver.
        """
        
        print(f"🎯 Initial prompt: {initial_prompt.strip()}")
        print(f"📊 Edit history: {len(edit_history)} examples")
        
        # Run refinement
        print("\n🚀 Starting iterative refinement...")
        result = refinement_engine.refine(
            initial_prompt=initial_prompt,
            user_edit_history=edit_history,
            user_id="demo_user",
            context={"task_type": "email", "domain": "business"}
        )
        
        # Display results
        print(f"\n📈 REFINEMENT RESULTS:")
        print(f"Total iterations: {result.total_iterations}")
        print(f"Converged: {result.converged}")
        print(f"Convergence reason: {result.convergence_reason}")
        print(f"Total time: {result.total_time:.2f}s")
        
        print(f"\n📊 METRICS:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\n💬 FINAL GENERATION:")
        if result.final_generation:
            print(f"  {result.final_generation.text}")
        
        # Show iteration details
        print(f"\n🔍 ITERATION DETAILS:")
        for i, iteration in enumerate(result.iterations):
            print(f"\nIteration {iteration.iteration}:")
            print(f"  Alignment score: {iteration.judge_feedback.get('alignment_score', 'N/A')}")
            print(f"  Confidence: {iteration.judge_feedback.get('confidence', 'N/A')}")
            print(f"  Should continue: {iteration.should_continue}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Stage 1 demo: {str(e)}")
        print("💡 Make sure you have set OPENAI_API_KEY environment variable")
        return None


def demo_stage2_logits_steering():
    """Demonstrate Stage 2: Decoding-time Preference Steering"""
    print("\n" + "="*60)
    print("🎛️ STAGE 2: DECODING-TIME STEERING DEMO")
    print("="*60)
    
    try:
        # Initialize components
        print("🔧 Initializing steering components...")
        
        # Note: This is a simplified demo - in practice you'd integrate with actual model logits
        vocab_size = 50257  # GPT-2 vocab size for demo
        preference_embedder = PreferenceEmbedder(preference_dim=256)
        logits_steerer = LogitsSteerer(
            vocab_size=vocab_size,
            preference_dim=256,
            hidden_dim=512
        )
        
        # Create sample preference embedding
        print("🎨 Creating preference embedding from edit history...")
        edit_history = create_sample_edit_history()
        
        # Infer preference embedding
        preference_embedding = preference_embedder.infer_preference_from_edits(
            edit_history, user_id="demo_user"
        )
        
        print(f"✨ Preference embedding shape: {preference_embedding.shape}")
        print(f"📏 Embedding norm: {preference_embedding.norm().item():.4f}")
        
        # Simulate logits steering
        print("\n🎯 Simulating logits steering...")
        import torch
        
        # Create fake original logits
        original_logits = torch.randn(1, vocab_size)
        
        # Different steering configurations
        steering_configs = [
            SteeringParams(alpha=0.05, adaptive=False),
            SteeringParams(alpha=0.1, adaptive=False),
            SteeringParams(alpha=0.1, adaptive=True),
        ]
        
        for i, params in enumerate(steering_configs):
            print(f"\n🔧 Configuration {i+1}: alpha={params.alpha}, adaptive={params.adaptive}")
            
            steered_logits, metadata = logits_steerer.steer_logits(
                original_logits=original_logits,
                preference_embedding=preference_embedding.unsqueeze(0),
                params=params
            )
            
            # Evaluate steering effect
            eval_results = logits_steerer.evaluate_steering_effect(
                original_logits, steered_logits, top_k=5
            )
            
            print(f"  Steering strength: {metadata['steering_strength'][0]:.4f}")
            print(f"  KL divergence: {eval_results['kl_divergence']:.4f}")
            print(f"  Max prob change: {eval_results['max_prob_change'][0]:.4f}")
            print(f"  Entropy change: {eval_results['entropy_change'][0]:.4f}")
        
        # Progressive steering demo
        print(f"\n⏳ Progressive steering over generation steps:")
        base_params = SteeringParams(alpha=0.2, beta=0.1, adaptive=True)
        
        for step in range(5):
            steered_logits, metadata = logits_steerer.progressive_steering(
                original_logits=original_logits,
                preference_embedding=preference_embedding.unsqueeze(0),
                generation_step=step,
                params=base_params
            )
            
            print(f"  Step {step}: decay={metadata['decay_factor']:.3f}, "
                  f"strength={metadata['steering_strength'][0]:.4f}")
        
        # Calibration demo
        print(f"\n⚖️ Calibrating steering strength...")
        preference_embeddings = [preference_embedding for _ in range(3)]
        optimal_alpha = logits_steerer.calibrate_steering_strength(
            preference_embeddings, target_kl_divergence=0.1
        )
        print(f"  Optimal alpha: {optimal_alpha:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Stage 2 demo: {str(e)}")
        return False


def demo_combined_system():
    """Demonstrate combined Stage 1 + Stage 2 system"""
    print("\n" + "="*60)
    print("🔗 COMBINED SYSTEM DEMO")
    print("="*60)
    
    print("This would demonstrate the full pipeline:")
    print("1. 🔄 Run iterative refinement to optimize prompt")
    print("2. 🎯 Extract final preference embedding") 
    print("3. 🎛️ Use embedding for decoding-time steering")
    print("4. 📊 Compare results across approaches")
    
    # For now, just show the concept
    refinement_result = demo_stage1_iterative_refinement()
    steering_success = demo_stage2_logits_steering()
    
    if refinement_result and steering_success:
        print("\n✅ Combined system demonstration completed successfully!")
        print("\n🔄 Stage 1 provided iterative refinement with judge feedback")
        print("🎛️ Stage 2 demonstrated preference-based logits steering")
        print("🔗 In practice, these would work together for optimal personalization")
    
    return refinement_result and steering_success


def main():
    """Main demo function"""
    print("🚀 PERSONALIZED LLM EVALUATION SYSTEM DEMO")
    print("=" * 80)
    
    print("\nThis demo showcases an advanced system for:")
    print("• 🔄 Iterative prompt refinement with judge/meta-judge feedback")
    print("• 🎯 User preference inference from edit history")
    print("• 🎛️ Decoding-time preference steering")
    print("• 📊 Comprehensive evaluation metrics")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Some features will use dummy/simulated responses")
    
    try:
        # Run individual demos
        print("\n" + "🎬 Starting demonstrations..." + "\n")
        
        # Stage 1: Iterative Refinement
        refinement_result = demo_stage1_iterative_refinement()
        
        # Stage 2: Logits Steering  
        steering_success = demo_stage2_logits_steering()
        
        # Combined System
        combined_success = demo_combined_system()
        
        # Summary
        print("\n" + "="*60)
        print("📋 DEMO SUMMARY")
        print("="*60)
        print(f"✅ Stage 1 (Iterative Refinement): {'Success' if refinement_result else 'Failed'}")
        print(f"✅ Stage 2 (Logits Steering): {'Success' if steering_success else 'Failed'}")
        print(f"✅ Combined System: {'Success' if combined_success else 'Failed'}")
        
        if refinement_result:
            print(f"\n🎯 Key achievements:")
            print(f"  • Processed {len(create_sample_edit_history())} edit pairs")
            print(f"  • Ran {refinement_result.total_iterations} refinement iterations")
            print(f"  • Achieved convergence: {refinement_result.converged}")
            print(f"  • Final alignment: {refinement_result.metrics.get('final_alignment', 'N/A')}")
        
        print(f"\n💡 Next steps:")
        print(f"  • Integrate with your own LLM endpoints")
        print(f"  • Train on your specific edit datasets")
        print(f"  • Customize preference categories and metrics")
        print(f"  • Scale to production workloads")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 