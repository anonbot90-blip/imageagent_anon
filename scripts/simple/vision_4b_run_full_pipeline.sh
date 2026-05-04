#!/bin/bash

# Full Vision Training Pipeline (Trajectory-Based with Cached Embeddings) - SIMPLE Dataset 4B
# Dataset: imageagent_results_10000_cot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_10000_cot"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/simple/cot_4b_trajectory"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/simple/cot_4b_trajectory/vision"

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

if [ ! -f "$TRAJECTORY_DIR/test_samples_cot_4b.txt" ]; then
    echo -e "${RED}❌ Error: Trajectory split not found${NC}"
    echo "Please run text pipeline first or create split manually"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAJECTORY_DIR/train_samples_cot_4b.txt")
TEST_COUNT=$(wc -l < "$TRAJECTORY_DIR/test_samples_cot_4b.txt")

echo -e "${GREEN}✓ Trajectory split verified (4B)${NC}"
echo -e "  Train: $TRAIN_COUNT samples"
echo -e "  Test: $TEST_COUNT samples"

# ============================================================================
# Step 2: Generate Training Data (if not exists)
# ============================================================================

print_header "Step 2/8: Verify Training Data (4B)"

# Set trajectory prefix for 4B
export TRAJECTORY_PREFIX="cot_4b"

# Check if vision data already generated (check all 5 datasets)
MISSING_DATA=0
for dataset in standard rl rw dpo sw; do
    if [ "$dataset" = "dpo" ]; then
        DATA_FILE="$TRAJECTORY_DIR/${dataset}_vision/planner_training_data_dpo.json"
    else
        DATA_FILE="$TRAJECTORY_DIR/${dataset}_vision/planner_training_data.json"
    fi
    
    if [ ! -f "$DATA_FILE" ]; then
        MISSING_DATA=1
        break
    fi
done

if [ $MISSING_DATA -eq 1 ]; then
    echo -e "${BLUE}Generating vision training data for 4B...${NC}"
   
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/simple/generate_standard_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/simple/generate_rl_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/simple/generate_rw_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/simple/generate_dpo_trajectory.sh"
    RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/simple/generate_sw_trajectory.sh"
    
    echo -e "${GREEN}✓ All vision training data generated for 4B${NC}"
else
    echo -e "${GREEN}✓ All vision training data already exists for 4B (standard, rl, rw, dpo, sw)${NC}"
fi

# ============================================================================
# Step 3: Precompute Vision Embeddings
# ============================================================================

print_header "Step 3/8: Precompute Vision Embeddings (if needed) - 4B"

# NOTE: Embeddings are deterministic - if you need to force recomputation,
# delete the embeddings/ directories: rm -rf training_data/simple/cot_4b_trajectory/*/embeddings/

# Launch parallel embedding computation for all 5 datasets (one per GPU)
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
    --config configs_simple_4b/planner_config_standard_trajectory_cached.yaml

echo -e "${GREEN}✓ Standard vision 4B model trained${NC}"
sleep 10

# RL Training
print_header "Step 5/8: Train RL Vision Model - 4B (port 29515)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29515 \
    train_planner_rl_trajectory_cached.py \
    --config configs_simple_4b/planner_config_rl_trajectory_cached.yaml

echo -e "${GREEN}✓ RL vision 4B model trained${NC}"
sleep 10

# RW Training
print_header "Step 6/8: Train RW Vision Model - 4B (port 29516)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29516 \
    train_planner_rw_trajectory_cached.py \
    --config configs_simple_4b/planner_config_rw_trajectory_cached.yaml

echo -e "${GREEN}✓ RW vision 4B model trained${NC}"
sleep 10

# DPO Training
print_header "Step 7/8: Train DPO Vision Model - 4B (port 29517)"
# echo -e "${CYAN}  - SW Vision (trajectory-standardized weighted training)${NC}"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29517 \
    train_planner_dpo_trajectory_cached.py \
    --config configs_simple_4b/planner_config_dpo_trajectory_cached.yaml \

# SW Training
print_header "Step 8/8: Train SW Vision Model (port 29519)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29519 \
    train_planner_sw_trajectory_cached.py \
    --config configs_simple_4b/planner_config_sw_trajectory_cached.yaml

echo -e "${GREEN}✓ SW vision model trained${NC}"
sleep 10

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
echo -e "${CYAN}  - SW Vision (trajectory-standardized weighted training)${NC}"
echo -e "${YELLOW}⏱️  Expected time: ~30-60 minutes for test set${NC}"
echo -e "${PURPLE}🤖 GPT-4o Action Judge: ALWAYS ENABLED (hardcoded in evaluation script)${NC}"
echo -e "${CYAN}🎯 Goal: Test if trajectory-based training improves vision 4B performance${NC}"
echo ""

EVAL_START=$(date +%s)

# Use the unified evaluation system
bash "$PROJECT_ROOT/scripts/evaluation/run_parallel_evaluation.sh" \
    --config "$PROJECT_ROOT/scripts/evaluation/configs/simple_vision_4b.yaml"

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
echo -e "${CYAN}   Evaluation: evaluation_results/simple/vision_parallel_cot_4b_trajectory/${NC}"
echo ""

echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "${YELLOW}   - Review evaluation results in evaluation_results/simple/vision_parallel_cot_4b_trajectory/${NC}"
echo -e "${YELLOW}   - Compare with 8B baseline (evaluation_results/simple/vision_parallel_cot_8b_trajectory/)${NC}"
echo ""

