#!/bin/bash

# Start Action Planner Training - Text-Only Mode (All Variants)
# Trains 4 models sequentially: Standard, RL, RW, DPO
# Fine-tune Qwen3-VL using text prompts only (no images, ultra fast)

# Note: We DON'T use 'set -e' here because we want to continue training
# other models even if one fails, and we want to run checkpoint recovery

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

# Standard Training Configuration
STANDARD_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/standard/planner_training_data.json"
STANDARD_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_text_only"

# RL Training Configuration
RL_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/rl/planner_training_data.json"
RL_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_text_only_rl"

# RW (Reward-Weighted) Training Configuration
RW_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/rw_text/planner_training_data.json"
RW_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_text_only_rw"

# DPO (Direct Preference Optimization) Training Configuration
DPO_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/dpo_text/planner_training_data_dpo.json"
DPO_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_text_only_dpo"

# GPU Configuration
DEFAULT_GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=8

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEFAULT_GPUS
    echo "Setting CUDA_VISIBLE_DEVICES=$DEFAULT_GPUS"
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Training Scripts
STANDARD_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_text_only.py"
RL_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_rl_text.py"
RW_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_rw_text.py"
DPO_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_dpo_text.py"

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

check_file() {
    local file=$1
    local description=$2
    
    if [ ! -f "$file" ]; then
        print_color $RED "❌ $description not found: $file"
        exit 1
    fi
    print_color $GREEN "✓ $description found"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "🧠 Sequential Text-Only Training - Pre-flight Checks"

print_color $YELLOW "⚠️  Text-Only Mode: Images will be ignored (10× faster)"
print_color $CYAN "   Use for: Fast iterations, testing, low-resource training"
echo ""

print_color $CYAN "📋 Training Plan (4 models):"
print_color $CYAN "   1️⃣  Standard Text-Only: $STANDARD_DATA_PATH"
print_color $CYAN "       → $STANDARD_OUTPUT_DIR"
print_color $CYAN "   2️⃣  RL Text-Only: $RL_DATA_PATH"
print_color $CYAN "       → $RL_OUTPUT_DIR"
print_color $CYAN "   3️⃣  RW Text-Only: $RW_DATA_PATH"
print_color $CYAN "       → $RW_OUTPUT_DIR"
print_color $CYAN "   4️⃣  DPO Text-Only: $DPO_DATA_PATH"
print_color $CYAN "       → $DPO_OUTPUT_DIR"
echo ""

# Check files
check_file "$STANDARD_SCRIPT" "Standard training script"
check_file "$RW_SCRIPT" "RW training script"
check_file "$DPO_SCRIPT" "DPO training script"
check_file "$STANDARD_DATA_PATH" "Standard training data"
check_file "$RL_DATA_PATH" "RL training data"
check_file "$RW_DATA_PATH" "RW training data"
check_file "$DPO_DATA_PATH" "DPO training data"

# Check GPU
print_color $CYAN "🎮 GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
echo ""
print_color $YELLOW "📌 Using GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
print_color $CYAN "🔀 Training Mode: $([ $NUM_GPUS -gt 1 ] && echo "Multi-GPU (DDP)" || echo "Single GPU")"
print_color $CYAN "⚡ Speedup: ~10× faster than vision-language training"

# Load credentials
if [ -f "$PROJECT_ROOT/credentials.sh" ]; then
    print_color $BLUE "🔑 Loading credentials..."
    source "$PROJECT_ROOT/credentials.sh"
    print_color $GREEN "✓ Credentials loaded"
else
    print_color $YELLOW "⚠️  credentials.sh not found - wandb logging may not work"
fi

# Disable flash-attn
export ATTN_BACKEND=xformers
export DIFFUSERS_ATTN_IMPLEMENTATION=eager

# ============================================================================
# Environment Setup
# ============================================================================

print_header "🔧 Environment Setup"

# Check for conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    print_color $YELLOW "⚠️  Conda not found, trying to continue..."
fi

# Activate environment
if command -v conda >/dev/null 2>&1; then
    print_color $BLUE "Activating img-agent environment..."
    conda activate img-agent || {
        print_color $RED "❌ Failed to activate img-agent environment"
        exit 1
    }
    print_color $GREEN "✓ Environment activated"
else
    print_color $YELLOW "⚠️  Conda not available, using system Python"
fi

# Display Python info
PYTHON_PATH=$(which python)
PYTHON_VERSION=$(python --version)
print_color $CYAN "🐍 Python: $PYTHON_PATH"
print_color $CYAN "📦 Version: $PYTHON_VERSION"

# Exit codes for each model set default to 0
# If a model fails, the exit code will be non-zero, otherwise it will be 0
STANDARD_EXIT_CODE=0
RL_EXIT_CODE=0
RW_EXIT_CODE=0
DPO_EXIT_CODE=0

# ============================================================================
# Training Function
# ============================================================================

# Function to check and fix missing final checkpoint
check_and_fix_final_checkpoint() {
    local checkpoint_dir=$1
    local model_name=$2
    
    print_color $CYAN "🔍 Checking final checkpoint for $model_name..."
    
    # Check if final checkpoint exists and is valid
    if [ ! -f "${checkpoint_dir}/final/adapter_model.safetensors" ]; then
        print_color $YELLOW "⚠️  Final checkpoint missing for ${model_name}"
        
        # Find last checkpoint (highest number)
        local last_ckpt=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null | sort -V | tail -1)
        
        if [ -n "$last_ckpt" ]; then
            print_color $CYAN "📋 Copying $(basename $last_ckpt) to final/..."
            rm -rf "${checkpoint_dir}/final"
            cp -r "$last_ckpt" "${checkpoint_dir}/final"
            print_color $GREEN "✅ Final checkpoint created from $(basename $last_ckpt)"
        else
            print_color $RED "❌ No checkpoints found in ${checkpoint_dir}!"
            return 1
        fi
    else
        print_color $GREEN "✅ Final checkpoint exists for ${model_name}"
    fi
    return 0
}

train_model() {
    local mode=$1
    local data_path=$2
    local output_dir=$3
    local training_script=$4
    local port=$5
    local step=$6
    
    print_header "🚀 [$step/4] Starting $mode Text-Only Training"
    
    print_color $CYAN "📝 Configuration:"
    print_color $CYAN "   Mode: $mode (text-only)"
    print_color $CYAN "   Script: $(basename $training_script)"
    print_color $CYAN "   Data: $data_path"
    print_color $CYAN "   Output: $output_dir"
    print_color $CYAN "   GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
    
    # Display correct batch sizes per model
    case "$mode" in
        "Standard"|"RL"|"RW")
            print_color $CYAN "   Per-GPU Batch: 2"
            print_color $CYAN "   Gradient Accum: 4"
            print_color $CYAN "   Effective Batch: $(($NUM_GPUS * 2 * 4))"
            ;;
        "DPO")
            print_color $CYAN "   Per-GPU Batch: 1"
            print_color $CYAN "   Gradient Accum: 8"
            print_color $CYAN "   Effective Batch: $(($NUM_GPUS * 1 * 8))"
            ;;
    esac
    echo ""
    
    case "$mode" in
        "Standard")
            print_color $GREEN "⚡ Baseline: Uniform random sampling"
            ;;
        "RL")
            print_color $GREEN "🏆 Reward-filtered: threshold >= 4.0"
            ;;
        "RW")
            print_color $GREEN "⚖️  Reward-weighted: Emphasizes high-quality samples"
            ;;
        "DPO")
            print_color $GREEN "🎯 Preference learning: Chosen vs rejected pairs"
            ;;
    esac
    print_color $YELLOW "⚠️  Text-only: No visual context, 10× faster"
    echo ""
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run training
    print_color $BLUE "🏃 Executing training script with torchrun..."
    echo ""
    
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$port "$training_script" \
        --data-path "$data_path" \
        --output-dir "$output_dir"
    
    local torchrun_exit=$?
    
    # Always check and fix final checkpoint after training
    check_and_fix_final_checkpoint "$output_dir" "$mode"
    local fix_exit=$?
    
    # Return success if checkpoint was fixed or already exists
    if [ $fix_exit -eq 0 ]; then
        print_color $GREEN "✓ Training complete: $output_dir/final/"
        return 0
    else
        print_color $RED "✗ Training failed - no valid checkpoints"
        return 1
    fi
}

# ============================================================================
# Start Sequential Training
# ============================================================================

print_header "🚀 Starting Sequential Text-Only Training (4 Models)"

# Train Standard Model
train_model "Standard" "$STANDARD_DATA_PATH" "$STANDARD_OUTPUT_DIR" "$STANDARD_SCRIPT" 29502 1
STANDARD_EXIT_CODE=$?

if [ $STANDARD_EXIT_CODE -ne 0 ]; then
    print_color $RED "❌ Standard training failed (exit code: $STANDARD_EXIT_CODE)"
    print_color $YELLOW "⚠️  Continuing with remaining models..."
fi

print_color $GREEN "✅ Standard text-only model complete!"
echo ""
print_color $CYAN "⏳ Waiting 10 seconds before next training..."
sleep 10

# Train RL Model
train_model "RL" "$RL_DATA_PATH" "$RL_OUTPUT_DIR" "$RL_SCRIPT" 29503 2
RL_EXIT_CODE=$?

if [ $RL_EXIT_CODE -ne 0 ]; then
    print_color $RED "❌ RL training failed (exit code: $RL_EXIT_CODE)"
    print_color $YELLOW "⚠️  Continuing with remaining models..."
fi

print_color $GREEN "✅ RL text-only model complete!"
echo ""
print_color $CYAN "⏳ Waiting 10 seconds before next training..."
sleep 10

# Train RW Model
train_model "RW" "$RW_DATA_PATH" "$RW_OUTPUT_DIR" "$RW_SCRIPT" 29504 3
RW_EXIT_CODE=$?

if [ $RW_EXIT_CODE -ne 0 ]; then
    print_color $RED "❌ RW training failed (exit code: $RW_EXIT_CODE)"
    print_color $YELLOW "⚠️  Continuing with remaining models..."
fi

print_color $GREEN "✅ RW text-only model complete!"
echo ""
print_color $CYAN "⏳ Waiting 30 seconds before DPO training (GPU memory cleanup)..."
sleep 30

# Train DPO Model
train_model "DPO" "$DPO_DATA_PATH" "$DPO_OUTPUT_DIR" "$DPO_SCRIPT" 29505 4
DPO_EXIT_CODE=$?

# ============================================================================
# Final Summary
# ============================================================================

print_header "📊 Text-Only Training Summary (4 Models)"

# Count successes
SUCCESS_COUNT=0

if [ $STANDARD_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [1/4] Standard: SUCCESS → $STANDARD_OUTPUT_DIR"
    ((SUCCESS_COUNT++))
else
    print_color $RED "❌ [1/4] Standard: FAILED (exit code: $STANDARD_EXIT_CODE)"
fi

if [ $RL_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [2/4] RL: SUCCESS → $RL_OUTPUT_DIR"
    ((SUCCESS_COUNT++))
else
    print_color $RED "❌ [2/4] RL: FAILED (exit code: $RL_EXIT_CODE)"
fi

if [ $RW_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [3/4] RW: SUCCESS → $RW_OUTPUT_DIR"
    ((SUCCESS_COUNT++))
else
    print_color $RED "❌ [3/4] RW: FAILED (exit code: $RW_EXIT_CODE)"
fi

if [ $DPO_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [4/4] DPO: SUCCESS → $DPO_OUTPUT_DIR"
    ((SUCCESS_COUNT++))
else
    print_color $RED "❌ [4/4] DPO: FAILED (exit code: $DPO_EXIT_CODE)"
fi

echo ""

if [ $SUCCESS_COUNT -eq 4 ]; then
    print_header "🎉 All 4 Text-Only Models Trained Successfully!"
    print_color $PURPLE "Next: bash scripts/evaluation/start_eval_all_planner_text.sh"
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    print_header "⚠️  Partial Success ($SUCCESS_COUNT/4 models)"
    exit 1
else
    print_header "❌ All Training Failed"
    exit 1
fi

