#!/bin/bash

# ImageAgent Environment Setup Script
# Sets up conda environment with HiDream-I1, HiDream-E1, and Qwen-VL
# Uses environment.yml as the primary installation method

set -e

echo "=========================================="
echo "ImageAgent Environment Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml not found in current directory"
    echo "Please run this script from the ImageAgent root directory"
    exit 1
fi

# Environment name
ENV_NAME="img-agent"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    # Deactivate if currently active
    if [ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]; then
        conda deactivate
    fi
    conda env remove -n ${ENV_NAME} -y
fi

echo "Creating conda environment from environment.yml..."
echo "This will install most packages (PyTorch will be installed separately)..."
conda env create -f environment.yml

echo "Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Installing PyTorch from PyTorch index (CUDA 12.4)..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo "Verifying PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')" || {
    echo "❌ PyTorch installation failed!"
    exit 1
}

echo "Installing Flash Attention (requires torch to be installed first)..."
echo "Note: This may take several minutes as it compiles from source"
if pip install flash-attn==2.6.3 --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
    echo "✅ Flash Attention installed successfully"
else
    echo "⚠️  Flash Attention installation failed (this is optional)"
    echo "   You can install it later with: pip install flash-attn==2.6.3 --no-build-isolation"
    echo "   Or try a pre-built wheel if available"
fi

echo "Verifying key packages..."
python -c "
import sys
errors = []
try:
    import torch
    print(f'✅ torch: {torch.__version__}')
except Exception as e:
    errors.append(f'❌ torch: {e}')

try:
    import transformers
    print(f'✅ transformers: {transformers.__version__}')
except Exception as e:
    errors.append(f'❌ transformers: {e}')

try:
    import peft
    print(f'✅ peft: {peft.__version__}')
except Exception as e:
    errors.append(f'❌ peft: {e}')

try:
    import diffusers
    print(f'✅ diffusers: {diffusers.__version__}')
except Exception as e:
    errors.append(f'❌ diffusers: {e}')

try:
    from transformers import AutoModelForImageTextToText
    print('✅ AutoModelForImageTextToText: available')
except Exception as e:
    errors.append(f'❌ AutoModelForImageTextToText: {e}')

try:
    from peft import PeftMixedModel
    print('✅ PeftMixedModel: available')
except Exception as e:
    errors.append(f'❌ PeftMixedModel: {e}')

try:
    import omegaconf
    print(f'✅ omegaconf: {omegaconf.__version__}')
except Exception as e:
    errors.append(f'❌ omegaconf: {e}')

if errors:
    print('\n⚠️  Some packages failed to import:')
    for err in errors:
        print(f'   {err}')
    sys.exit(1)
"

echo ""
echo "Installing additional packages (if needed)..."

# Install HiDream-I1 requirements if directory exists
if [ -d "HiDream-I1" ] && [ -f "HiDream-I1/requirements.txt" ]; then
    echo "Installing HiDream-I1 requirements..."
    pip install -r HiDream-I1/requirements.txt
fi

# Install LaTeX compiler (Tectonic) if not already installed
if ! command -v tectonic &> /dev/null; then
    echo "Installing LaTeX compiler (Tectonic)..."
    conda install -c conda-forge tectonic -y || {
        echo "⚠️  Tectonic installation failed (optional for LaTeX compilation)"
    }
fi

echo ""
echo "Verifying and installing critical packages that may be missing..."

# Install/upgrade bitsandbytes (required for DPO training with int8 quantization)
echo "Installing/upgrading bitsandbytes (required for transformers 4.57.1)..."
pip install -U "bitsandbytes>=0.49.0" || {
    echo "⚠️  bitsandbytes installation failed (DPO training may not work)"
}

# Install qwen-vl-utils (required for ImageAnalyzerQwen3)
echo "Installing qwen-vl-utils..."
pip install qwen-vl-utils==0.0.14 || {
    echo "⚠️  qwen-vl-utils installation failed (image analysis may not work)"
}

# Install h5py (required for cached vision training)
echo "Installing h5py (required for cached vision embeddings)..."
pip install h5py || {
    echo "⚠️  h5py installation failed (cached vision training may not work)"
}

# Install wandb (required for training callbacks)
echo "Installing wandb (required for training logging)..."
pip install wandb==0.22.2 || {
    echo "⚠️  wandb installation failed (training logging may not work)"
}

# Install matplotlib (required for evaluation table generation)
echo "Installing matplotlib (required for evaluation table generation)..."
pip install matplotlib || {
    echo "⚠️  matplotlib installation failed (table generation may not work)"
}

# Verify critical packages are importable
echo ""
echo "Final verification of critical packages..."
python -c "
import sys
errors = []
try:
    import qwen_vl_utils
    print('✅ qwen-vl-utils: available')
except Exception as e:
    errors.append(f'❌ qwen-vl-utils: {e}')

try:
    import bitsandbytes
    print(f'✅ bitsandbytes: {bitsandbytes.__version__}')
except Exception as e:
    errors.append(f'❌ bitsandbytes: {e}')

try:
    import h5py
    print(f'✅ h5py: {h5py.__version__}')
except Exception as e:
    errors.append(f'❌ h5py: {e}')

try:
    import wandb
    print('✅ wandb: available')
except Exception as e:
    errors.append(f'❌ wandb: {e}')

try:
    import matplotlib.pyplot as plt
    print('✅ matplotlib: available')
except Exception as e:
    errors.append(f'❌ matplotlib: {e}')

if errors:
    print('\n⚠️  Some critical packages failed to import:')
    for err in errors:
        print(f'   {err}')
    print('\nYou may need to install them manually.')
" || echo "⚠️  Package verification had issues"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate img-agent"
echo ""
echo "Or if conda is not initialized in your shell:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate img-agent"
echo ""
echo "Next steps:"
echo "1. Run: huggingface-cli login"
echo "2. Accept Llama 3.1 license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "3. Test training script import:"
echo "   python -c 'from training.planner_training.train_planner_standard_trajectory_text import load_config; print(\"✅ OK\")'"
echo "4. Test image generation: ./scripts/generate_five_images.sh"
echo "5. Compile LaTeX paper: cd latex && tectonic main.tex"
echo ""
