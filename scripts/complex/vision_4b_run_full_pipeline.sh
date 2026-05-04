#!/bin/bash

# Full Vision Training Pipeline - Complex V2 Dataset - 4B Model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_complex_v2_10k_cot"
# TRAJECTORY_PREFIX for exports only
TRAJECTORY_PREFIX="cot_4b"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/complex/cot_4b_trajectory"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/complex/cot_4b_trajectory/vision"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
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

START_TIME=$(date +%s)

print_header "Vision Training Pipeline - Complex V2 4B"

if [ ! -f "$TRAJECTORY_DIR/test_samples_cot_4b.txt" ]; then
    echo -e "${RED}❌ Trajectory split not found. Run text pipeline first.${NC}"
    exit 1
fi

for dataset in standard rl rw dpo sw; do
    if [ ! -f "$TRAJECTORY_DIR/${dataset}_vision/planner_training_data.json" ] && \
       [ ! -f "$TRAJECTORY_DIR/${dataset}_vision/planner_training_data_dpo.json" ]; then
        echo -e "${RED}❌ Vision training data not found. Run text pipeline first.${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ Training data verified${NC}"

print_header "Precomputing Vision Embeddings (Parallel on 5 GPUs)"

PIDS=()
GPU_ID=0

for dataset in standard rl rw dpo sw; do
    DATA_DIR="$TRAJECTORY_DIR/${dataset}_vision"
    EMBEDDINGS_DIR="$DATA_DIR/embeddings"
    
    if [ "$dataset" = "dpo" ]; then
        DATA_FILE="$DATA_DIR/planner_training_data_dpo.json"
    else
        DATA_FILE="$DATA_DIR/planner_training_data.json"
    fi
    
    if [ ! -f "$EMBEDDINGS_DIR/vision_embeddings.h5" ]; then
        echo -e "${BLUE}[GPU $GPU_ID] Precomputing $dataset embeddings...${NC}"
        python "$PROJECT_ROOT/scripts/precompute_embeddings_parallel.py" \
            --data-path "$DATA_FILE" \
            --output-dir "$EMBEDDINGS_DIR" \
            --gpu-id $GPU_ID \
            --dataset-name "$dataset" &
        PIDS+=($!)
    else
        echo -e "${GREEN}✓ $dataset embeddings exist${NC}"
    fi
    
    GPU_ID=$((GPU_ID + 1))
done

if [ ${#PIDS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⏳ Waiting for embeddings...${NC}"
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
fi

echo -e "${GREEN}✓ All embeddings ready${NC}"

TRAIN_START=$(date +%s)

cd "$PROJECT_ROOT/training/planner_training"

print_header "Train Standard Vision Model - Complex V2 4B (port 29534)"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29534 \
    train_planner_standard_trajectory_cached.py \
    --config configs_complex_4b/planner_config_standard_trajectory_cached.yaml
echo -e "${GREEN}✓ Standard vision trained${NC}"
sleep 10

print_header "Train RL Vision Model - Complex V2 4B (port 29535)"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29535 \
    train_planner_rl_trajectory_cached.py \
    --config configs_complex_4b/planner_config_rl_trajectory_cached.yaml
echo -e "${GREEN}✓ RL vision trained${NC}"
sleep 10

print_header "Train RW Vision Model - Complex V2 4B (port 29536)"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29536 \
    train_planner_rw_trajectory_cached.py \
    --config configs_complex_4b/planner_config_rw_trajectory_cached.yaml
echo -e "${GREEN}✓ RW vision trained${NC}"
sleep 10

print_header "Train DPO Vision Model - Complex V2 4B (port 29537)"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29537 \
    train_planner_dpo_trajectory_cached.py \
    --config configs_complex_4b/planner_config_dpo_trajectory_cached.yaml
echo -e "${GREEN}✓ DPO vision trained${NC}"
sleep 10

print_header "Train SW Vision Model - Complex V2 4B (port 29539)"
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29539 \
    train_planner_sw_trajectory_cached.py \
    --config configs_complex_4b/planner_config_sw_trajectory_cached.yaml
echo -e "${GREEN}✓ SW vision trained${NC}"

cd "$PROJECT_ROOT"

# ============================================================================
# Evaluate All Models (on test samples)
# ============================================================================

print_header "📈 Evaluate Vision Models - Complex V2 4B"

echo -e "${BLUE}Evaluating 8 models on test samples...${NC}"
echo -e "  - Baseline (Qwen3-VL-4B-Instruct)"
echo -e "  - Edit-Only (Direct Editing)"
echo -e "  - Standard Vision"
echo -e "  - RL Vision"
echo -e "  - RW Vision"
echo -e "  - DPO Vision"
echo -e "  - SW Vision"
echo ""

EVAL_START=$(date +%s)

# Use the unified evaluation system
bash "$PROJECT_ROOT/scripts/evaluation/run_parallel_evaluation.sh" \
    --config "$PROJECT_ROOT/scripts/evaluation/configs/complex_vision_4b.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Evaluation failed${NC}"
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

print_header "🎉 Vision Pipeline Complete - Complex V2 4B"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
TRAIN_TIME=$((END_TIME - TRAIN_START))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
TRAIN_HOURS=$((TRAIN_TIME / 3600))
TRAIN_MINS=$(((TRAIN_TIME % 3600) / 60))

echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
echo ""

echo -e "${CYAN}📊 Summary:${NC}"
echo -e "${CYAN}   Training: ${TRAIN_HOURS}h ${TRAIN_MINS}m${NC}"
echo -e "${CYAN}   Evaluation: ${EVAL_HOURS}h ${EVAL_MINS}m${NC}"
echo -e "${CYAN}   Total: ${HOURS}h ${MINUTES}m${NC}"
echo ""

echo -e "${CYAN}📂 Results:${NC}"
echo -e "${CYAN}   Checkpoints: $CHECKPOINT_BASE/vision/${NC}"
echo -e "${CYAN}   Evaluation: evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/${NC}"
echo ""
