#!/bin/bash
# HiDream-E1 LoRA Finetuning - Multi-GPU Training
# Usage: bash scripts/start_training.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

# Training Data Configuration
DEFAULT_DATA_DIR="$PROJECT_ROOT/imageagent_results_5000"  # Training data directory
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/hidream_e1_lora_theme_transform_5000"  # Output checkpoint directory

# GPU Configuration - Multi-GPU Support
DEFAULT_GPUS="0,1,2,3,4,5,6,7"  # Use ALL 8 GPUs for maximum speed
NUM_GPUS=8  # Number of GPUs

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEFAULT_GPUS
    echo "Setting CUDA_VISIBLE_DEVICES=$DEFAULT_GPUS"
fi

# Count number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Helper Functions
# ============================================================================

print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    print_color $PURPLE "============================================================================"
    print_color $PURPLE "$1"
    print_color $PURPLE "============================================================================"
    echo ""
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "🎨 HiDream-E1 LoRA Finetuning - Starting Training"

# Source credentials
print_color $BLUE "📝 Loading credentials..."
if [ -f "$PROJECT_ROOT/credentials.sh" ]; then
    source "$PROJECT_ROOT/credentials.sh"
    print_color $GREEN "✅ Credentials loaded"
    print_color $CYAN "   - HuggingFace: Token set"
    print_color $CYAN "   - WandB Project: $WANDB_PROJECT"
else
    print_color $YELLOW "⚠️  Warning: credentials.sh not found"
    print_color $YELLOW "   Using system defaults"
fi

echo ""

# Check GPU
print_color $CYAN "🖥️  GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
echo ""
print_color $YELLOW "📌 Using GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
print_color $CYAN "🔀 Training Mode: $([ $NUM_GPUS -gt 1 ] && echo "Multi-GPU (DDP)" || echo "Single GPU")"

echo ""

# Activate conda environment if needed
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    print_color $BLUE "🐍 Using conda environment: $CONDA_DEFAULT_ENV"
else
    print_color $BLUE "🐍 Using system Python"
fi

# Check Python
PYTHON_VERSION=$(python --version)
print_color $CYAN "📦 Python Version: $PYTHON_VERSION"

echo ""

# Fix flash-attn issue if present
print_color $BLUE "🔧 Checking for flash-attn compatibility..."
if python -c "import flash_attn" 2>/dev/null; then
    print_color $YELLOW "⚠️  flash-attn detected but may be incompatible"
    print_color $YELLOW "   Uninstalling to avoid conflicts..."
    pip uninstall -y flash-attn 2>/dev/null || true
    print_color $GREEN "✅ flash-attn removed"
else
    print_color $GREEN "✅ flash-attn not present (good)"
fi

# ============================================================================
# Environment Setup
# ============================================================================

print_header "🔧 Environment Setup"

# Set environment to disable flash-attn
export ATTN_BACKEND=xformers
export DIFFUSERS_ATTN_IMPLEMENTATION=eager
print_color $GREEN "✓ Disabled flash-attn (using xformers/eager)"

# Display training configuration
print_color $CYAN "📝 Training Configuration:"
print_color $CYAN "   Config: training/config/training_config.yaml"
print_color $CYAN "   Data Dir: $DEFAULT_DATA_DIR"
print_color $CYAN "   Output Dir: $DEFAULT_OUTPUT_DIR"
print_color $CYAN "   GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
print_color $CYAN "   Per-GPU Batch: 1"
print_color $CYAN "   Gradient Accum: 4"
print_color $CYAN "   Effective Batch: $(($NUM_GPUS * 1 * 4)) ($(($NUM_GPUS)) × 1 × 4)"

# ============================================================================
# Start Training
# ============================================================================

print_header "🚀 Starting HiDream-E1 Training"

# Change to project root
cd "$PROJECT_ROOT"

# Run training with torchrun for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    print_color $BLUE "🏃 Executing training with torchrun (multi-GPU)..."
    echo ""
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 \
        training/train.py \
        --config training/config/training_config.yaml \
        --output_dir "$DEFAULT_OUTPUT_DIR" \
        --data_dirs "$DEFAULT_DATA_DIR"
else
    print_color $BLUE "🏃 Executing training (single GPU)..."
    echo ""
    python training/train.py \
        --config training/config/training_config.yaml \
        --output_dir "$DEFAULT_OUTPUT_DIR" \
        --data_dirs "$DEFAULT_DATA_DIR"
fi

EXIT_CODE=$?

# ============================================================================
# Post-training
# ============================================================================

echo ""

if [ $EXIT_CODE -eq 0 ]; then
    print_header "🎉 Training Complete!"
    
    print_color $GREEN "✓ HiDream-E1 training finished successfully"
    print_color $YELLOW "📁 Checkpoints saved in: ./checkpoints/hidream_e1_lora_theme_transform/"
    
    # List checkpoints
    if [ -d "$PROJECT_ROOT/checkpoints/hidream_e1_lora_theme_transform" ]; then
        echo ""
        print_color $CYAN "📊 Available checkpoints:"
        ls -lh "$PROJECT_ROOT/checkpoints/hidream_e1_lora_theme_transform/" | grep "^d" | awk '{print "   " $9}'
    fi
    
    echo ""
    print_color $PURPLE "Next steps:"
    print_color $PURPLE "  1. Evaluate HiDream-E1 model"
    print_color $PURPLE "  2. Test on new image editing tasks"
    print_color $PURPLE "  3. Generate more training data if needed"
    
else
    print_header "❌ Training Failed"
    print_color $RED "Training exited with code: $EXIT_CODE"
    print_color $YELLOW "Check logs above for errors"
    exit $EXIT_CODE
fi
