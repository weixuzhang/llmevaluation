# macOS Setup Guide

This guide provides macOS-specific instructions for setting up and running the LLM Personalization project.

## Prerequisites

### Python Installation
The project requires Python 3.8 or higher. Check your Python version:
```bash
python3 --version
```

If you need to install Python:
```bash
# Option 1: Using Homebrew (recommended)
brew install python@3.11

# Option 2: Download from python.org
# Visit https://www.python.org/downloads/macos/
```

### Virtual Environment (Recommended)
Create a virtual environment to avoid conflicts:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Installation Steps

1. **Clone or navigate to the project directory**:
```bash
cd /path/to/personalization
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **If you encounter PyTorch installation issues**:
```bash
# For CPU-only (most macOS users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Apple Silicon Macs with MPS support (optional)
pip install torch torchvision torchaudio
```

## macOS-Specific Considerations

### Apple Silicon (M1/M2/M3) Macs
- The code runs natively on Apple Silicon
- PyTorch will automatically use Metal Performance Shaders (MPS) if available
- GPU acceleration is supported but not required

### Intel Macs
- Full compatibility with all features
- CPU-based computation (GPU acceleration via CUDA not available)

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB RAM for larger models
- The demo uses small models that work well on 8GB systems

## Running the Project

### Quick Start
```bash
# Download datasets
python scripts/download_data.py

# Run demo
python demo.py

# Interactive notebook
jupyter notebook demo_notebook.ipynb
```

### Performance Tips for macOS

1. **Use smaller models for testing**:
```bash
python main.py --task LongLaMP-2 --model_name distilgpt2 --num_users 5
```

2. **Monitor system resources**:
```bash
# Check memory usage
top -pid $(pgrep -f python)

# Monitor CPU usage
Activity Monitor.app
```

3. **Optimize for your hardware**:
- Apple Silicon: Enable MPS backend (automatic)
- Intel: Use CPU optimization flags
- Limited RAM: Reduce batch size and number of users

## Troubleshooting

### Common Issues

**1. PyTorch Installation Problems**
```bash
# Uninstall and reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**2. Sentence Transformers Issues**
```bash
# Update sentence-transformers
pip install --upgrade sentence-transformers
```

**3. Jupyter Notebook Problems**
```bash
# Install Jupyter kernel
python -m ipykernel install --user --name=venv
```

**4. Memory Errors**
- Reduce `num_users` parameter
- Use smaller models (distilgpt2, microsoft/DialoGPT-small)
- Close other applications

### Performance Optimization

For best performance on macOS:
```bash
# Use environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run with optimizations
python -O main.py --task LongLaMP-2 --model_name distilgpt2
```

## Verification

Test your installation:
```bash
python -c "
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
print('Installation successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')
"
```

## Next Steps

1. **Run the demo notebook**: `jupyter notebook demo_notebook.ipynb`
2. **Try quick experiment**: `python scripts/run_quick_experiment.py`
3. **Scale up gradually**: Start with small models and few users, then increase

For any issues, check the main README.md or create an issue in the repository.
