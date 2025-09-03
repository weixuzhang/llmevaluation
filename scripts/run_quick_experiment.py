#!/usr/bin/env python3
"""
Quick experiment script for testing the personalization system.
Uses a small model and few users for fast iteration.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main
import argparse

def run_quick_experiment():
    """Run a quick experiment with small model and few users."""
    
    # Override sys.argv for quick testing
    test_args = [
        "run_quick_experiment.py",
        "--task", "LongLaMP-2",
        "--model_name", "microsoft/DialoGPT-small",
        "--num_users", "5",
        "--max_new_tokens", "50",
        "--beta", "0.1",
        "--experiment_name", "quick_test",
        "--save_samples",
        "--log_level", "INFO"
    ]
    
    print("Running quick experiment with the following settings:")
    print("- Task: LongLaMP-2 (Abstract Generation)")
    print("- Model: microsoft/DialoGPT-small")
    print("- Users: 5")
    print("- Max tokens: 50")
    print("- Beta: 0.1")
    print()
    
    # Save original argv and replace
    original_argv = sys.argv
    sys.argv = test_args
    
    try:
        main()
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    run_quick_experiment()
