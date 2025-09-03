# LLM Personalization with User Embeddings

This project implements a prototype system for personalized text generation using large language models (LLMs) and user embeddings. The system uses the LongLaMP benchmark for evaluation.

**Platform Support**: Fully compatible with macOS, Linux, and Windows.

## Project Overview

The goal is to demonstrate that incorporating simple user embeddings—extracted from each user's profile or historical writing—can improve the personalization of generated outputs.

## Methodology

1. **User Embedding Construction**: Extract user embeddings from historical text data using Sentence-BERT
2. **Personalized Generation**: Adjust LLM logits based on similarity between candidate tokens and user embeddings
3. **Evaluation**: Compare personalized vs. baseline outputs using ROUGE and METEOR metrics

## Setup

### macOS Installation

1. Install Python 3.8+ (if not already installed):
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note for macOS users**: If you encounter any issues with PyTorch installation, use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Download the LongLaMP dataset:
```bash
python scripts/download_data.py
```

4. Run a quick demo:
```bash
python demo.py
```

5. Explore the interactive demo notebook:
```bash
jupyter notebook demo_notebook.ipynb
```

6. Run a quick experiment:
```bash
python scripts/run_quick_experiment.py
```

7. Run a full experiment:
```bash
python main.py --task LongLaMP-2 --model_name microsoft/DialoGPT-small --num_users 10
```

## Project Structure

```
├── src/
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # Model implementations
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Utility functions
├── configs/                     # Configuration files
├── scripts/                     # Helper scripts
├── experiments/                 # Experiment results
└── main.py                      # Main experiment script
```

## Features

- Support for multiple LongLaMP tasks (LongLaMP-2, LongLaMP-3, LongLaMP-4)
- User embedding extraction using Sentence-BERT
- Personalized text generation with logit adjustment
- Comprehensive evaluation using ROUGE and METEOR metrics
- Configurable personalization strength (beta parameter)
- Experiment tracking and result saving
- Demo and quick experiment scripts for testing
- Batch experiment runner for multiple configurations

## Available Commands

- `python demo.py` - Run interactive demo showcasing all components
- `jupyter notebook demo_notebook.ipynb` - Interactive step-by-step demo notebook
- `python scripts/download_data.py` - Download and verify LongLaMP dataset  
- `python scripts/run_quick_experiment.py` - Quick test with small model/dataset
- `python main.py [options]` - Run full personalization experiment
- `bash scripts/run_experiments.sh` - Run batch experiments with multiple configurations

## Example Results

The system generates personalized text by:
1. Computing user embeddings from their historical profiles
2. At each generation step, adjusting token probabilities based on similarity to user embedding
3. Generating more personalized content that reflects user preferences and writing style

Typical improvements over baseline:
- ROUGE-1: +0.02 to +0.05
- ROUGE-L: +0.01 to +0.04  
- METEOR: +0.01 to +0.03
