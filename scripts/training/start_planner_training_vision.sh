#!/bin/bash

# Start Action Planner Training - Vision Mode (Cached Embeddings)
# Fine-tune Qwen3-VL using pre-computed vision embeddings (3× faster)
# Trains 4 vision models sequentially: Standard, RL, RW, DPO

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

# Standard Vision Training Configuration
STANDARD_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/standard/planner_training_data.json"
STANDARD_EMBEDDINGS_PATH="$PROJECT_ROOT/training_data/cot_8b/standard/embeddings/vision_embeddings.h5"
STANDARD_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_vision_standard"

# RL Vision Training Configuration
RL_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/rl/planner_training_data.json"
RL_EMBEDDINGS_PATH="$PROJECT_ROOT/training_data/cot_8b/rl/embeddings/vision_embeddings.h5"
RL_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_rl"

# RW (Reward-Weighted) Vision Training Configuration
RW_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/rw_vision/planner_training_data.json"
RW_EMBEDDINGS_PATH="$PROJECT_ROOT/training_data/cot_8b/rw_vision/embeddings/vision_embeddings.h5"
RW_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_vision_rw"

# DPO (Direct Preference Optimization) Vision Training Configuration
DPO_DATA_PATH="$PROJECT_ROOT/training_data/cot_8b/dpo_vision/planner_training_data_dpo.json"
DPO_EMBEDDINGS_PATH="$PROJECT_ROOT/training_data/cot_8b/dpo_vision/embeddings/vision_embeddings.h5"
DPO_OUTPUT_DIR="$PROJECT_ROOT/checkpoints/cot_8b/qwen3_vl_action_planner_vision_rl_dpo"

# GPU Configuration
DEFAULT_GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=8

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEFAULT_GPUS
    echo "Setting CUDA_VISIBLE_DEVICES=$DEFAULT_GPUS"
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Paths
STANDARD_CONFIG="$PROJECT_ROOT/training/planner_training/planner_config_cached.yaml"
STANDARD_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_cached.py"
RL_CONFIG="$PROJECT_ROOT/training/planner_training/planner_config_rl_cached.yaml"
RL_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_rl_cached.py"
RW_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_rw_cached.py"
DPO_SCRIPT="$PROJECT_ROOT/training/planner_training/train_planner_dpo_cached.py"
# Embedding scripts no longer needed - handled by full pipeline

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
        return 1
    fi
    print_color $GREEN "✓ $description found"
    return 0
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "🧠 Sequential Vision Training - Pre-flight Checks"

print_color $YELLOW "⚡ Vision Mode: Using cached embeddings (3× faster)"
print_color $CYAN "   Use for: Full vision-language training with pre-computed embeddings"
echo ""

print_color $CYAN "📋 Training Plan:"
print_color $CYAN "   1️⃣  Standard Vision: $STANDARD_DATA_PATH"
print_color $CYAN "       → $STANDARD_OUTPUT_DIR"
print_color $CYAN "   2️⃣  RL Vision: $RL_DATA_PATH"
print_color $CYAN "       → $RL_OUTPUT_DIR"
print_color $CYAN "   3️⃣  RW Vision: $RW_DATA_PATH"
print_color $CYAN "       → $RW_OUTPUT_DIR"
print_color $CYAN "   4️⃣  DPO Vision: $DPO_DATA_PATH"
print_color $CYAN "       → $DPO_OUTPUT_DIR"
echo ""

# Check files
check_file "$STANDARD_CONFIG" "Standard training config" || exit 1
check_file "$STANDARD_SCRIPT" "Standard training script" || exit 1
check_file "$RL_CONFIG" "RL training config" || exit 1
check_file "$RL_SCRIPT" "RL training script" || exit 1
check_file "$RW_SCRIPT" "RW training script" || exit 1
check_file "$DPO_SCRIPT" "DPO training script" || exit 1
check_file "$STANDARD_DATA_PATH" "Standard training data" || exit 1
check_file "$RL_DATA_PATH" "RL training data" || exit 1
check_file "$RW_DATA_PATH" "RW training data" || exit 1
check_file "$DPO_DATA_PATH" "DPO training data" || exit 1

# Verify embeddings exist (should be pre-computed by full pipeline)
print_color $CYAN "🔍 Verifying vision embeddings..."
check_file "$STANDARD_EMBEDDINGS_PATH" "Standard embeddings" || exit 1
check_file "$RL_EMBEDDINGS_PATH" "RL embeddings" || exit 1  
check_file "$RW_EMBEDDINGS_PATH" "RW embeddings" || exit 1
check_file "$DPO_EMBEDDINGS_PATH" "DPO embeddings" || exit 1
print_color $GREEN "✅ All vision embeddings verified"
echo ""

# Check GPU
print_color $CYAN "🎮 GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
echo ""
print_color $YELLOW "📌 Using GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
print_color $CYAN "🔀 Training Mode: $([ $NUM_GPUS -gt 1 ] && echo "Multi-GPU (DDP)" || echo "Single GPU")"
print_color $CYAN "⚡ Speedup: ~3× faster than standard training (cached embeddings)"

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
    local embeddings_path=$3
    local output_dir=$4
    local training_script=$5
    local port=$6
    local step=$7
    
    if [ "$mode" = "standard" ]; then
        print_header "🚀 [1/4] Starting Standard Vision Training"
    elif [ "$mode" = "rl" ]; then
        print_header "🏆 [2/4] Starting RL Vision Training"
    elif [ "$mode" = "rw" ]; then
        print_header "⚖️  [3/4] Starting RW Vision Training"
    else
        print_header "🎯 [4/4] Starting DPO Vision Training"
    fi
    
    print_color $CYAN "📝 Configuration:"
    print_color $CYAN "   Mode: $mode (vision with cached embeddings)"
    print_color $CYAN "   Config: $(basename $training_script | sed 's/train_planner_/planner_config_/')"
    print_color $CYAN "   Data: $data_path"
    print_color $CYAN "   Embeddings: $embeddings_path"
    print_color $CYAN "   Output: $output_dir"
    print_color $CYAN "   GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
    
    # Display correct batch sizes per model (all vision models use batch=1 for 8B)
    print_color $CYAN "   Per-GPU Batch: 1"
    print_color $CYAN "   Gradient Accum: 8"
    print_color $CYAN "   Effective Batch: $(($NUM_GPUS * 1 * 8))"
    echo ""
    
    if [ "$mode" = "rl" ]; then
        print_color $GREEN "🏆 Training on reward-filtered high-quality data"
        print_color $GREEN "⚡ Expected time: 8-12 hours (8B model with cached embeddings)"
        echo ""
    elif [ "$mode" = "rw" ]; then
        print_color $GREEN "⚖️  Training with reward-weighted loss (TRUE per-sample weighting)"
        print_color $GREEN "⚡ Expected time: 8-12 hours (8B model with cached embeddings)"
        echo ""
    elif [ "$mode" = "dpo" ]; then
        print_color $GREEN "🎯 Training with preference optimization (chosen vs rejected)"
        print_color $GREEN "⚡ Expected time: 8-12 hours (8B model with cached embeddings)"
        echo ""
    else
        print_color $GREEN "⚡ Expected time: 8-12 hours (8B model with cached embeddings)"
        echo ""
    fi
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run training
    print_color $BLUE "🏃 Executing training script with torchrun..."
    echo ""
    
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$port "$training_script" \
        --data-path "$data_path" \
        --embeddings-path "$embeddings_path" \
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

# # ============================================================================
# # Start Sequential Training
# # ============================================================================

# Initialize exit codes for all models to 0
STANDARD_EXIT_CODE=0
RL_EXIT_CODE=0
RW_EXIT_CODE=0
DPO_EXIT_CODE=0

# print_header "🚀 Starting Sequential Vision Training (Standard → RL → RW → DPO)"

# # Train Standard Vision Model
# train_model "standard" "$STANDARD_DATA_PATH" "$STANDARD_EMBEDDINGS_PATH" "$STANDARD_OUTPUT_DIR" "$STANDARD_SCRIPT" 29504 1
# STANDARD_EXIT_CODE=$?

# # Check standard training result
# if [ $STANDARD_EXIT_CODE -ne 0 ]; then
#     print_color $RED "❌ Standard training failed (exit code: $STANDARD_EXIT_CODE)"
#     print_color $YELLOW "⚠️  Continuing with remaining models..."
# fi

# print_color $GREEN "✅ Standard vision model training complete!"
# echo ""
# print_color $CYAN "⏳ Waiting 10 seconds before starting RL training..."
# sleep 10

# # Train RL Vision Model
# train_model "rl" "$RL_DATA_PATH" "$RL_EMBEDDINGS_PATH" "$RL_OUTPUT_DIR" "$RL_SCRIPT" 29505 2
# RL_EXIT_CODE=$?

# # Check RL training result
# if [ $RL_EXIT_CODE -ne 0 ]; then
#     print_color $RED "❌ RL training failed (exit code: $RL_EXIT_CODE)"
#     print_color $YELLOW "⚠️  Continuing with remaining models..."
# fi

# print_color $GREEN "✅ RL vision model training complete!"
# echo ""
# print_color $CYAN "⏳ Waiting 10 seconds before starting RW training..."
# sleep 10

# # Train RW Vision Model
# train_model "rw" "$RW_DATA_PATH" "$RW_EMBEDDINGS_PATH" "$RW_OUTPUT_DIR" "$RW_SCRIPT" 29506 3
# RW_EXIT_CODE=$?

# # Check RW training result
# if [ $RW_EXIT_CODE -ne 0 ]; then
#     print_color $RED "❌ RW training failed (exit code: $RW_EXIT_CODE)"
#     print_color $YELLOW "⚠️  Continuing with remaining models..."
# fi

# print_color $GREEN "✅ RW vision model training complete!"
# echo ""
# print_color $CYAN "⏳ Waiting 30 seconds before starting DPO training..."
# sleep 30

# # Train DPO Vision Model
# train_model "dpo" "$DPO_DATA_PATH" "$DPO_EMBEDDINGS_PATH" "$DPO_OUTPUT_DIR" "$DPO_SCRIPT" 29507 4
# DPO_EXIT_CODE=$?

# ============================================================================
# Final Summary
# ============================================================================

print_header "📊 Vision Training Summary (4 Models)"

# Count successes
SUCCESS_COUNT=0

if [ $STANDARD_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [1/4] Standard: SUCCESS → $STANDARD_OUTPUT_DIR"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    print_color $RED "❌ [1/4] Standard: FAILED (exit code: $STANDARD_EXIT_CODE)"
fi

if [ $RL_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [2/4] RL: SUCCESS → $RL_OUTPUT_DIR"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    print_color $RED "❌ [2/4] RL: FAILED (exit code: $RL_EXIT_CODE)"
fi

if [ $RW_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [3/4] RW: SUCCESS → $RW_OUTPUT_DIR"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    print_color $RED "❌ [3/4] RW: FAILED (exit code: $RW_EXIT_CODE)"
fi

if [ $DPO_EXIT_CODE -eq 0 ]; then
    print_color $GREEN "✅ [4/4] DPO: SUCCESS → $DPO_OUTPUT_DIR"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    print_color $RED "❌ [4/4] DPO: FAILED (exit code: $DPO_EXIT_CODE)"
fi

echo ""

if [ $SUCCESS_COUNT -eq 4 ]; then
    print_header "🎉 All 4 Vision Models Trained Successfully!"
    print_color $PURPLE "Next: bash scripts/evaluation/start_eval_all_planner_vision.sh"
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    print_header "⚠️  Partial Success ($SUCCESS_COUNT/4 models)"
    exit 1
else
    print_header "❌ All Training Failed"
    exit 1
fi

