#!/bin/bash

# Full Text-Only Training Pipeline (Trajectory-Based) - SIMPLE Dataset
# Dataset: imageagent_results_10000_cot
# Orchestrates: Data Generation → Training (5 models) → Evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration - SIMPLE DATASET
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_10000_cot"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/simple/cot_8b_trajectory"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/simple/cot_8b_trajectory"

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

print_header "Step 1/7: Verify Trajectory Split (SIMPLE 10K)"

if [ ! -f "$TRAJECTORY_DIR/test_samples_cot_8b.txt" ]; then
    echo -e "${YELLOW}⚠️  Trajectory split not found. Creating split...${NC}"
    
    python "$PROJECT_ROOT/scripts/training/simple/split_trajectories.py" \
        --results-dir "$RESULTS_DIR" \
        --test-threshold 5.5 \
        --test-count 200 \
        --reward-metric overall_quality \
        --output-dir "$TRAJECTORY_DIR" \
        --prefix cot_8b
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create trajectory split${NC}"
        exit 1
    fi
fi

TRAIN_COUNT=$(wc -l < "$TRAJECTORY_DIR/train_samples_cot_8b.txt")
TEST_COUNT=$(wc -l < "$TRAJECTORY_DIR/test_samples_cot_8b.txt")

echo -e "${GREEN}✓ Trajectory split verified${NC}"
echo -e "  Train: $TRAIN_COUNT samples"
echo -e "  Test: $TEST_COUNT samples"

# ============================================================================
# Step 2: Generate Training Data
# ============================================================================

print_header "Step 2/7: Generate Training Data (5 datasets)"

echo -e "${BLUE}Generating Standard training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" TRAJECTORY_PREFIX="cot_8b" bash "$PROJECT_ROOT/scripts/training/simple/generate_standard_trajectory.sh"

echo ""
echo -e "${BLUE}Generating RL training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" TRAJECTORY_PREFIX="cot_8b" bash "$PROJECT_ROOT/scripts/training/simple/generate_rl_trajectory.sh"

echo ""
echo -e "${BLUE}Generating RW training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" TRAJECTORY_PREFIX="cot_8b" bash "$PROJECT_ROOT/scripts/training/simple/generate_rw_trajectory.sh"

echo ""
echo -e "${BLUE}Generating DPO training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" TRAJECTORY_PREFIX="cot_8b" bash "$PROJECT_ROOT/scripts/training/simple/generate_dpo_trajectory.sh"

echo ""
echo -e "${BLUE}Generating SW training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" TRAJECTORY_PREFIX="cot_8b" bash "$PROJECT_ROOT/scripts/training/simple/generate_sw_trajectory.sh"

echo ""
echo -e "${GREEN}✓ All training data generated${NC}"

# ============================================================================
# Step 3-7: Train Models Sequentially
# ============================================================================

cd "$PROJECT_ROOT/training/planner_training"

# Standard Training
print_header "Step 3/7: Train Standard Model (port 29500)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_planner_standard_trajectory_text.py \
    --config configs_simple_8b/planner_config_standard_trajectory_text.yaml

echo -e "${GREEN}✓ Standard model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/standard"
sleep 10

# RL Training  
print_header "Step 4/7: Train RL Model (port 29501)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    train_planner_rl_trajectory_text.py \
    --config configs_simple_8b/planner_config_rl_trajectory_text.yaml

echo -e "${GREEN}✓ RL model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/rl"
sleep 10

# RW Training
print_header "Step 5/7: Train RW Model (port 29502)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29502 \
    train_planner_rw_trajectory_text.py \
    --config configs_simple_8b/planner_config_rw_trajectory_text.yaml

echo -e "${GREEN}✓ RW model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/rw"
sleep 10

# DPO Training
print_header "Step 6/7: Train DPO Model (port 29503)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29503 \
    train_planner_dpo_trajectory_text.py \
    --config configs_simple_8b/planner_config_dpo_trajectory_text.yaml

echo -e "${GREEN}✓ DPO model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/dpo"
sleep 10

# SW Training
print_header "Step 7/7: Train SW Model (port 29508)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29508 \
    train_planner_sw_trajectory_text.py \
    --config configs_simple_8b/planner_config_sw_trajectory_text.yaml

echo -e "${GREEN}✓ SW model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/sw"

TRAIN_TIME=$(($(date +%s) - START_TIME))
TRAIN_HOURS=$((TRAIN_TIME / 3600))
TRAIN_MINS=$(((TRAIN_TIME % 3600) / 60))

# ============================================================================
# Stage 3: Evaluate All Models (on test samples)
# ============================================================================

print_header "📈 Evaluate Text-Only Models (SIMPLE 10K)"

echo -e "${CYAN}Evaluating 8 models on test trajectory samples...${NC}"
echo -e "${CYAN}  - Baseline (Qwen3-VL-8B-Instruct)${NC}"
echo -e "${CYAN}  - Standard Text-Only (trajectory-sampled training)${NC}"
echo -e "${CYAN}  - RL Text-Only (trajectory-filtered training)${NC}"
echo -e "${CYAN}  - RW Text-Only (trajectory-weighted training)${NC}"
echo -e "${CYAN}  - DPO Text-Only (trajectory-preference training)${NC}"
echo -e "${CYAN}  - SW Text
  - GPT-4o Planner (API-based)-Only
  - GPT-4o Planner (API-based) (trajectory-standardized weighted training)${NC}"
echo -e "${YELLOW}⏱️  Expected time: ~30-60 minutes for test set${NC}"
echo -e "${PURPLE}🤖 GPT-4o Action Judge: ALWAYS ENABLED (hardcoded in evaluation script)${NC}"
echo -e "${CYAN}🎯 Dataset: imageagent_results_10000_cot${NC}"
echo ""

EVAL_START=$(date +%s)

# Change back to project root for evaluation
cd "$PROJECT_ROOT"

# Use unified config-based evaluation for simple 8B (same as 4B)
bash scripts/evaluation/run_parallel_evaluation.sh \
    --config scripts/evaluation/configs/simple_text_8b.yaml

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

print_header "🎉 SIMPLE Dataset Training & Evaluation Complete!"

echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
echo ""

echo -e "${CYAN}📊 Summary:${NC}"
echo -e "${CYAN}   Training: ${TRAIN_HOURS}h ${TRAIN_MINS}m${NC}"
echo -e "${CYAN}   Evaluation: ${EVAL_HOURS}h ${EVAL_MINS}m${NC}"
echo ""

echo -e "${CYAN}📂 Results:${NC}"
echo -e "${CYAN}   Dataset: $RESULTS_DIR${NC}"
echo -e "${CYAN}   Checkpoints: $CHECKPOINT_BASE/${NC}"
echo -e "${CYAN}     Standard: $CHECKPOINT_BASE/standard/final${NC}"
echo -e "${CYAN}     RL:       $CHECKPOINT_BASE/rl/final${NC}"
echo -e "${CYAN}     RW:       $CHECKPOINT_BASE/rw/final${NC}"
echo -e "${CYAN}     DPO:      $CHECKPOINT_BASE/dpo/final${NC}"
echo -e "${CYAN}     SW:       $CHECKPOINT_BASE/sw/final${NC}"
echo -e "${CYAN}   Evaluation: evaluation_results/simple/text_parallel_cot_8b_trajectory/${NC}"
echo ""

echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "${YELLOW}   - Review evaluation results${NC}"
echo -e "${YELLOW}   - Compare with normal/complex variants${NC}"
echo ""

