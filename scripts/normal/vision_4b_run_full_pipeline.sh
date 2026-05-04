#!/bin/bash

# Full Vision Training Pipeline (Trajectory-Based with Cached Embeddings) - 4B Model
# Orchestrates: Data Generation → Embedding Precomputation → Training (4 models)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

# CRITICAL: RESULTS_DIR must match the RESULTS_DIR in the evaluation wrapper script:
#   scripts/evaluation/normal/start_eval_all_planner_vision_trajectory_4b.sh
# If you change this path, you MUST also update the evaluation script's RESULTS_DIR
# to ensure full_dataset_for_eval.json is created from the correct source directory.
# Note: Vision evaluation scripts create full_dataset_for_eval.json via text scripts,
#       so this must match the text evaluation wrapper's RESULTS_DIR as well.
RESULTS_DIR="$PROJECT_ROOT/imageagent_results_normal_cot"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/normal/cot_4b_trajectory"
TRAJECTORY_BASE="$TRAJECTORY_DIR"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/normal/cot_4b_trajectory/vision"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Start timer
START_TIME=$(date +%s)

# ============================================================================
# Step 1: Verify Trajectory Split
# ============================================================================

print_header "Step 1/8: Verify Trajectory Split (4B Model)"

if [ ! -f "$TRAJECTORY_BASE/test_samples_cot_4b.txt" ]; then
    echo -e "${YELLOW}⚠️  Trajectory split not found. Creating split for 4B...${NC}"
    
    python "$PROJECT_ROOT/scripts/training/normal/split_trajectories.py" \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$TRAJECTORY_DIR" \
        --test-threshold 5.5 \
        --test-target-samples 200 \
        --reward-metric overall_quality \
        --prefix cot_4b
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create trajectory split${NC}"
        exit 1
    fi
fi

TRAIN_COUNT=$(wc -l < "$TRAJECTORY_BASE/train_samples_cot_4b.txt")
TEST_COUNT=$(wc -l < "$TRAJECTORY_BASE/test_samples_cot_4b.txt")

echo -e "${GREEN}✓ Trajectory split verified (4B)${NC}"
echo -e "  Train: $TRAIN_COUNT samples"
echo -e "  Test: $TEST_COUNT samples"

# ============================================================================
# Step 2: Generate Training Data (if not exists)
# ============================================================================

print_header "Step 2/8: Verify Training Data (4B)"

# Set trajectory prefix for 4B complex
export TRAJECTORY_PREFIX="cot_4b"

# Check if vision data already generated
if [ ! -f "$TRAJECTORY_DIR/standard_vision/planner_training_data.json" ]; then
    echo -e "${BLUE}Generating vision training data for 4B...${NC}"
   
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_standard_training_data_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_rl_training_data_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_rw_training_data_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_dpo_training_data_trajectory.sh"

    echo ""
    echo -e "${BLUE}Generating SW training data for 4B...${NC}"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_sw_training_data_trajectory.sh"
else
    echo -e "${GREEN}✓ Vision training data already exists for 4B${NC}"
fi

# ============================================================================
# Step 3: Precompute Vision Embeddings
# ============================================================================

print_header "Step 3/8: Precompute Vision Embeddings (if needed) - 4B"

# Launch parallel embedding computation for all 4 datasets (one per GPU)
declare -a PIDS
GPU_ID=0

for dataset_name in standard rl rw dpo sw; do
    DATA_DIR="$TRAJECTORY_DIR/${dataset_name}_vision"
    EMBEDDINGS_DIR="$DATA_DIR/embeddings"
   
    # Determine data file name (DPO has different name)
    if [ "$dataset_name" = "dpo" ]; then
        DATA_FILE="$DATA_DIR/planner_training_data_dpo.json"
    else
        DATA_FILE="$DATA_DIR/planner_training_data.json"
    fi
   
    # Check if embeddings already exist
    if [ ! -d "$EMBEDDINGS_DIR" ] || [ ! -f "$EMBEDDINGS_DIR/vision_embeddings.h5" ]; then
        echo -e "${BLUE}[GPU $GPU_ID] Precomputing embeddings for $dataset_name (4B)...${NC}"
       
        python "$PROJECT_ROOT/scripts/precompute_embeddings_parallel.py" \
            --data-path "$DATA_FILE" \
            --output-dir "$EMBEDDINGS_DIR" \
            --gpu-id $GPU_ID \
            --dataset-name "$dataset_name" &
       
        PIDS+=($!)
    else
        echo -e "${GREEN}✓ Embeddings for $dataset_name already exist (4B)${NC}"
    fi
   
    GPU_ID=$((GPU_ID + 1))
done

# Wait for all embedding processes
if [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⏳ Waiting for embedding precomputation to complete (4B)...${NC}"
    for pid in "${PIDS[@]}"; do
        wait $pid
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Embedding computation failed${NC}"
            exit 1
        fi
    done
fi

echo ""
echo -e "${GREEN}✓ All vision embeddings ready for 4B${NC}"

# ============================================================================
# Step 4-7: Train Models Sequentially (4B)
# ============================================================================

TRAIN_START=$(date +%s)

cd "$PROJECT_ROOT/training/planner_training"

# Standard Training
print_header "Step 4/8: Train Standard Vision Model - 4B (port 29514)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29514 \
    train_planner_standard_trajectory_cached.py \
    --config configs_4b/planner_config_standard_trajectory_cached.yaml

echo -e "${GREEN}✓ Standard vision 4B model trained${NC}"
sleep 10

# RL Training
print_header "Step 5/8: Train RL Vision Model - 4B (port 29515)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29515 \
    train_planner_rl_trajectory_cached.py \
    --config configs_4b/planner_config_rl_trajectory_cached.yaml

echo -e "${GREEN}✓ RL vision 4B model trained${NC}"
sleep 10

# RW Training
print_header "Step 6/8: Train RW Vision Model - 4B (port 29516)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29516 \
    train_planner_rw_trajectory_cached.py \
    --config configs_4b/planner_config_rw_trajectory_cached.yaml

echo -e "${GREEN}✓ RW vision 4B model trained${NC}"
sleep 10

# DPO Training
print_header "Step 7/8: Train DPO Vision Model - 4B (port 29517)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29517 \
    train_planner_dpo_trajectory_cached.py \
    --config configs_4b/planner_config_dpo_trajectory_cached.yaml

echo -e "${GREEN}✓ DPO vision 4B model trained${NC}"
sleep 10

# SW Training
print_header "Step 8/8: Train SW Model (port 29518)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29518 \
    train_planner_sw_trajectory_cached.py \
    --config configs_4b/planner_config_sw_trajectory_cached.yaml

echo -e "${GREEN}✓ SW model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/sw"

# Return to project root
cd "$PROJECT_ROOT"

# ============================================================================
# Stage 4: Evaluate All Models (on test samples)
# ============================================================================

print_header "📈 Evaluate Vision Models (Trajectory-Based) - 4B"

echo -e "${CYAN}Evaluating 8 models on test trajectory samples...${NC}"
echo -e "${CYAN}  - Baseline (Qwen3-VL-4B-Instruct)${NC}"
echo -e "${CYAN}  - Standard Vision 4B (trajectory-sampled training)${NC}"
echo -e "${CYAN}  - RL Vision 4B (trajectory-filtered training)${NC}"
echo -e "${CYAN}  - RW Vision 4B (trajectory-weighted training)${NC}"
echo -e "${CYAN}  - DPO Vision 4B (trajectory-preference training)${NC}"
echo -e "${CYAN}  - SW Vision (trajectory standardized-weighted training)${NC}"
echo -e "${YELLOW}⏱️  Expected time: ~30-60 minutes for test set${NC}"
echo -e "${PURPLE}🤖 GPT-4o Action Judge: ALWAYS ENABLED (hardcoded in evaluation script)${NC}"
echo -e "${CYAN}🎯 Goal: Test if trajectory-based training improves vision 4B performance${NC}"
echo ""

EVAL_START=$(date +%s)

# Change back to project root for evaluation
cd "$PROJECT_ROOT"

# Use unified config-based evaluation for normal vision 4B
bash scripts/evaluation/run_parallel_evaluation.sh \
    --config scripts/evaluation/configs/normal_vision_4b.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Evaluation failed${NC}"
    echo -e "${YELLOW}⚠️  Check logs for details${NC}"
    exit 1
fi

EVAL_TIME=$(($(date +%s) - EVAL_START))
EVAL_HOURS=$((EVAL_TIME / 3600))
EVAL_MINS=$(((EVAL_TIME % 3600) / 60))

echo -e "${GREEN}✅ Evaluation complete in ${EVAL_HOURS}h ${EVAL_MINS}m${NC}"
echo ""

# ============================================================================
# Final Summary
# ============================================================================

print_header "🎉 Trajectory-Based Vision 4B Pipeline Complete!"

echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
echo ""

# Check if training was done (TRAIN_START is set)
if [ -n "$TRAIN_START" ]; then
    TRAIN_TIME=$(($(date +%s) - TRAIN_START))
    TRAIN_HOURS=$((TRAIN_TIME / 3600))
    TRAIN_MINS=$(((TRAIN_TIME % 3600) / 60))
    
    echo -e "${CYAN}📊 Summary:${NC}"
    echo -e "${CYAN}   Training: ${TRAIN_HOURS}h ${TRAIN_MINS}m${NC}"
    echo -e "${CYAN}   Evaluation: ${EVAL_HOURS}h ${EVAL_MINS}m${NC}"
    echo ""
else
    # Evaluation only mode
    echo -e "${CYAN}📊 Summary:${NC}"
    echo -e "${CYAN}   Evaluation: ${EVAL_HOURS}h ${EVAL_MINS}m${NC}"
    echo ""
fi

echo -e "${CYAN}📂 Results:${NC}"
echo -e "${CYAN}   Checkpoints: $CHECKPOINT_BASE/${NC}"
echo -e "${CYAN}     Standard: $CHECKPOINT_BASE/standard/final${NC}"
echo -e "${CYAN}     RL:       $CHECKPOINT_BASE/rl/final${NC}"
echo -e "${CYAN}     RW:       $CHECKPOINT_BASE/rw/final${NC}"
echo -e "${CYAN}     DPO:      $CHECKPOINT_BASE/dpo/final${NC}"
echo -e "${CYAN}   Evaluation: evaluation_results/vision_parallel_cot_4b_trajectory/${NC}"
echo ""

echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "${YELLOW}   - Review evaluation results in evaluation_results/vision_parallel_cot_4b_trajectory/${NC}"
echo -e "${YELLOW}   - Compare with 8B baseline (evaluation_results/vision_parallel_cot_8b_trajectory/)${NC}"
echo ""

