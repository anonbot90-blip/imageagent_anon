#!/bin/bash

# Full Text-Only Training Pipeline (Trajectory-Based) - NORMAL Dataset
# Dataset: imageagent_results_normal_cot_test
# Orchestrates: Data Generation → Training (5 models) → Evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration - NORMAL DATASET
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_normal_cot"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/normal/cot_8b_trajectory"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/normal/cot_8b_trajectory"

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

print_header "Step 1/7: Verify Trajectory Split (NORMAL 10K)"

if [ ! -f "$TRAJECTORY_DIR/test_samples_cot_8b.txt" ]; then
    echo -e "${YELLOW}⚠️  Trajectory split not found. Creating split...${NC}"
    
    python "$PROJECT_ROOT/scripts/training/normal/split_trajectories.py" \
        --results-dir "$RESULTS_DIR" \
        --test-threshold 5.5 \
        --test-target-samples 4 \
        --reward-metric overall_quality \
        --output-base "$TRAJECTORY_DIR" \
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

print_header "Step 2/7: Generate Training Data (5 datasets for 8B)"

# Set trajectory prefix for 8B (CRITICAL: must be set before data generation)
export TRAJECTORY_PREFIX="cot_8b"

echo -e "${BLUE}Generating Standard training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_standard_trajectory.sh"

echo ""
echo -e "${BLUE}Generating RL training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_rl_trajectory.sh"

echo ""
echo -e "${BLUE}Generating RW training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_rw_trajectory.sh"

echo ""
echo -e "${BLUE}Generating DPO training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_dpo_trajectory.sh"

echo ""
echo -e "${BLUE}Generating SW training data...${NC}"
RESULTS_DIR="$RESULTS_DIR" bash "$PROJECT_ROOT/scripts/training/normal/generate_sw_trajectory.sh"

echo ""
echo -e "${GREEN}✓ All training data generated for 8B${NC}"

# ============================================================================
# Step 3-7: Train Models Sequentially (8B)
# ============================================================================

cd "$PROJECT_ROOT/training/planner_training"

# Standard Training
print_header "Step 3/7: Train Standard Model - 8B (port 29510)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29510 \
    train_planner_standard_trajectory_text.py \
    --config configs_8b/planner_config_standard_trajectory_text.yaml

echo -e "${GREEN}✓ Standard 8B model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/standard"
sleep 10

# RL Training  
print_header "Step 4/7: Train RL Model - 8B (port 29511)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29511 \
    train_planner_rl_trajectory_text.py \
    --config configs_8b/planner_config_rl_trajectory_text.yaml

echo -e "${GREEN}✓ RL 8B model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/rl"
sleep 10

# RW Training
print_header "Step 5/7: Train RW Model - 8B (port 29512)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29512 \
    train_planner_rw_trajectory_text.py \
    --config configs_8b/planner_config_rw_trajectory_text.yaml

echo -e "${GREEN}✓ RW 8B model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/rw"
sleep 10

# DPO Training
print_header "Step 6/7: Train DPO Model - 8B (port 29513)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29513 \
    train_planner_dpo_trajectory_text.py \
    --config configs_8b/planner_config_dpo_trajectory_text.yaml

echo -e "${GREEN}✓ DPO 8B model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/dpo"
sleep 10

# SW Training
print_header "Step 7/7: Train SW Model - 8B (port 29518)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29518 \
    train_planner_sw_trajectory_text.py \
    --config configs_8b/planner_config_sw_trajectory_text.yaml

echo -e "${GREEN}✓ SW 8B model trained${NC}"
echo -e "  Checkpoint: $CHECKPOINT_BASE/sw"
sleep 10

TRAIN_TIME=$(($(date +%s) - START_TIME))
TRAIN_HOURS=$((TRAIN_TIME / 3600))
TRAIN_MINS=$(((TRAIN_TIME % 3600) / 60))

# ============================================================================
# Stage 3: Evaluate All Models (on test samples)
# ============================================================================

print_header "📈 Evaluate Text-Only Models (NORMAL 10K - 8B)"

echo -e "${CYAN}Evaluating 8 models on test trajectory samples...${NC}"
echo -e "${CYAN}  - Baseline (Qwen3-VL-8B-Instruct)${NC}"
echo -e "${CYAN}  - Standard Text-Only 8B (trajectory-sampled training)${NC}"
echo -e "${CYAN}  - RL Text-Only 8B (trajectory-filtered training)${NC}"
echo -e "${CYAN}  - RW Text-Only 8B (trajectory-weighted training)${NC}"
echo -e "${CYAN}  - DPO Text-Only 8B (trajectory-preference training)${NC}"
echo -e "${CYAN}  - SW Text-Only 8B (trajectory-standardized weighted training)${NC}"
echo -e "${CYAN}  - GPT-4o Planner (API-based)${NC}"
echo -e "${YELLOW}⏱️  Expected time: ~30-60 minutes for test set${NC}"
echo -e "${PURPLE}🤖 GPT-4o Action Judge: ALWAYS ENABLED (hardcoded in evaluation script)${NC}"
echo -e "${CYAN}🎯 Dataset: imageagent_results_normal_cot_test${NC}"
echo ""

EVAL_START=$(date +%s)

# Change back to project root for evaluation
cd "$PROJECT_ROOT"

# Use unified config-based evaluation for normal 8B
bash scripts/evaluation/run_parallel_evaluation.sh \
    --config scripts/evaluation/configs/normal_text_8b.yaml

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

print_header "🎉 NORMAL Dataset Training & Evaluation Complete!"

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
echo -e "${CYAN}   Evaluation: evaluation_results/normal/text_parallel_cot_8b_trajectory/${NC}"
echo ""

echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "${YELLOW}   - Review evaluation results${NC}"
echo -e "${YELLOW}   - Compare with normal/complex variants${NC}"
echo ""

