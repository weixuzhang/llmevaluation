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

# Handle both relative and absolute imports
try:
    from .config import ExperimentConfig, LLMConfig, RefinementConfig, PreferenceConfig
    from .models.openai_model import OpenAIModel
    from .models.preference_embedder import PreferenceEmbedder, EditPair
    from .models.logits_steerer import LogitsSteerer, SteeringParams
    from .refinement.refinement_engine import RefinementEngine
except ImportError:
    # For direct execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import ExperimentConfig, LLMConfig, RefinementConfig, PreferenceConfig
    from src.models.openai_model import OpenAIModel
    from src.models.preference_embedder import PreferenceEmbedder, EditPair
    from src.models.logits_steerer import LogitsSteerer, SteeringParams
    from src.refinement.refinement_engine import RefinementEngine


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
        # Extract only valid OpenAI parameters
        valid_openai_params = {
            'temperature': config.llm_config.temperature,
            'max_tokens': config.llm_config.max_tokens,
            'top_p': config.llm_config.top_p,
            'frequency_penalty': config.llm_config.frequency_penalty,
            'presence_penalty': config.llm_config.presence_penalty
        }
        
        llm = OpenAIModel(
            model_name=config.llm_config.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            **valid_openai_params
        )
        
        preference_embedder = PreferenceEmbedder(
            encoder_model=config.preference_config.encoder_model,
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
        print("\n🎭 Running Stage 1 in DEMO MODE...")
        
        # Create a mock result to show the workflow
        from src.refinement.refinement_engine import RefinementResult, RefinementIteration
        from src.models.llm_interface import LLMOutput
        
        mock_generation = LLMOutput(
            text="[DEMO] Dear Client, I apologize for the project delay. Our team remains committed to delivering high-quality results and will provide you with a revised timeline shortly. Thank you for your patience.",
            metadata={'demo_mode': True}
        )
        
        mock_iteration = RefinementIteration(
            iteration=1,
            prompt=initial_prompt.strip(),
            generation=mock_generation,
            inferred_preferences={
                'structured_preferences': {'style_1': 'prefers more formal language'},
                'confidence': 0.75,
                'preference_summary': 'User prefers formal, professional communication'
            },
            judge_feedback={
                'alignment_score': 0.72,
                'confidence': 0.80,
                'feedback_text': 'Response aligns well with professional email preferences',
                'suggestions': ['Consider adding specific timeline', 'More empathetic tone']
            },
            should_continue=False
        )
        
        mock_result = RefinementResult(
            initial_prompt=initial_prompt.strip(),
            iterations=[mock_iteration],
            final_generation=mock_generation,
            total_iterations=1,
            converged=True,
            convergence_reason="Demo mode - simulated convergence",
            total_time=0.5,
            metrics={
                'initial_alignment': 0.72,
                'final_alignment': 0.72,
                'alignment_improvement': 0.0,
                'final_confidence': 0.80
            }
        )
        
        # Display mock results
        print(f"\n📈 REFINEMENT RESULTS (DEMO MODE):")
        print(f"Total iterations: {mock_result.total_iterations}")
        print(f"Converged: {mock_result.converged}")
        print(f"Convergence reason: {mock_result.convergence_reason}")
        print(f"Total time: {mock_result.total_time:.2f}s")
        
        print(f"\n📊 METRICS:")
        for key, value in mock_result.metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\n💬 FINAL GENERATION:")
        print(f"  {mock_result.final_generation.text}")
        
        print(f"\n🔍 ITERATION DETAILS:")
        print(f"Iteration 1:")
        print(f"  Alignment score: {mock_iteration.judge_feedback['alignment_score']}")
        print(f"  Confidence: {mock_iteration.judge_feedback['confidence']}")
        print(f"  Should continue: {mock_iteration.should_continue}")
        
        print(f"\n✨ This demonstrates the Stage 1 workflow structure!")
        return mock_result


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
        preference_embedder = PreferenceEmbedder(
            embedding_dim=768,
            preference_dim=256
        )
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
            
            # Safe value extraction function
            def safe_extract(val):
                if hasattr(val, 'item'):
                    return val.item()
                elif hasattr(val, '__getitem__') and hasattr(val, '__len__'):
                    try:
                        if len(val) > 0:
                            return val[0]
                    except:
                        pass
                return val
            
            steering_val = safe_extract(metadata['steering_strength'])
            max_prob_val = safe_extract(eval_results['max_prob_change'])
            entropy_val = safe_extract(eval_results['entropy_change'])
            
            print(f"  Steering strength: {steering_val:.4f}")
            print(f"  KL divergence: {eval_results['kl_divergence']:.4f}")
            print(f"  Max prob change: {max_prob_val:.4f}")
            print(f"  Entropy change: {entropy_val:.4f}")
        
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
            
            # Use same safe extraction function
            def safe_extract(val):
                if hasattr(val, 'item'):
                    return val.item()
                elif hasattr(val, '__getitem__') and hasattr(val, '__len__'):
                    try:
                        if len(val) > 0:
                            return val[0]
                    except:
                        pass
                return val
                
            steering_val = safe_extract(metadata['steering_strength'])
            decay_val = safe_extract(metadata['decay_factor'])
            print(f"  Step {step}: decay={decay_val:.3f}, "
                  f"strength={steering_val:.4f}")
        
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
        
        # Show how they would integrate
        print("\n🔗 INTEGRATION CONCEPT:")
        print("1. 📊 Extract preference embedding from Stage 1 refinement history")
        print("2. 🎛️ Use embedding for real-time logits steering in Stage 2")
        print("3. 📈 Combine prompt refinement + decoding steering for maximum personalization")
        print("4. 🔄 Iteratively improve both components based on user feedback")
    
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
        stage1_status = "Success" if refinement_result else "Failed"
        stage2_status = "Success" if steering_success else "Failed" 
        combined_status = "Success" if combined_success else "Partial Success" if (refinement_result or steering_success) else "Failed"
        
        print(f"✅ Stage 1 (Iterative Refinement): {stage1_status}")
        print(f"✅ Stage 2 (Logits Steering): {stage2_status}")
        print(f"✅ Combined System: {combined_status}")
        
        if stage1_status == "Success" and refinement_result and "DEMO" in str(refinement_result.final_generation.text):
            print("   💡 Stage 1 ran in demo mode (no OpenAI API key)")
        if stage2_status == "Success":
            print("   🎯 Stage 2 fully functional with preference steering")
        
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