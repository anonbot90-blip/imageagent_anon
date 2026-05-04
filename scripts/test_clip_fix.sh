#!/bin/bash
# Test CLIP fix and new tables on 2 samples (both text and vision)
# Safe test that won't overwrite existing results

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    CLIP FIX & NEW TABLES TEST                              ║${NC}"
echo -e "${CYAN}║                    Testing on 2 samples (Text + Vision)                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
DATA_PATH="$PROJECT_ROOT/training_data/standard/planner_training_data.json"
TEST_SAMPLES_FILE="$PROJECT_ROOT/test_samples.txt"
NUM_SAMPLES=2

# Test output directories
TEXT_OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/test_clip_fix_text"
VISION_OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/test_clip_fix_vision"

# Checkpoints
BASELINE_CHECKPOINT="Qwen/Qwen3-VL-4B-Instruct"
STANDARD_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_text_only/final"
RL_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_text_only_rl/final"
RW_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_text_only_rw/final"
DPO_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_text_only_dpo/final"

STANDARD_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_vision_standard/final"
RL_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_rl/final"
RW_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_vision_rw/final"
DPO_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints_40000/qwen3_vl_action_planner_vision_rl_dpo/final"

HIDREAM_CHECKPOINT="$PROJECT_ROOT/training/hidream_training/checkpoints/final"
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"

# ============================================================================
# PART 1: TEXT-ONLY EVALUATION
# ============================================================================

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  PART 1/2: TEXT-ONLY PIPELINE (5 models × 2 samples)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Text output directories
TEXT_BASELINE_DIR="$TEXT_OUTPUT_BASE/baseline"
TEXT_STANDARD_DIR="$TEXT_OUTPUT_BASE/standard_text"
TEXT_RL_DIR="$TEXT_OUTPUT_BASE/rl_text"
TEXT_RW_DIR="$TEXT_OUTPUT_BASE/rw_text"
TEXT_DPO_DIR="$TEXT_OUTPUT_BASE/dpo_text"
TEXT_CONSOLIDATED_DIR="$TEXT_OUTPUT_BASE/consolidated_text"

# Create directories
mkdir -p "$TEXT_BASELINE_DIR" "$TEXT_STANDARD_DIR" "$TEXT_RL_DIR" "$TEXT_RW_DIR" "$TEXT_DPO_DIR" "$TEXT_CONSOLIDATED_DIR"

echo -e "${CYAN}[1/5] Evaluating Baseline (text)...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$TEXT_BASELINE_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[2/5] Evaluating Standard Text-Only...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$STANDARD_TEXT_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$TEXT_STANDARD_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[3/5] Evaluating RL Text-Only...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$RL_TEXT_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$TEXT_RL_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[4/5] Evaluating RW Text-Only...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$RW_TEXT_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$TEXT_RW_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[5/5] Evaluating DPO Text-Only...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$DPO_TEXT_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$TEXT_DPO_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo ""
echo -e "${CYAN}Consolidating text results...${NC}"
bash scripts/evaluation/consolidate_text_results.sh \
    --baseline-dir "$TEXT_BASELINE_DIR" \
    --standard-text-dir "$TEXT_STANDARD_DIR" \
    --rl-text-dir "$TEXT_RL_DIR" \
    --rw-text-dir "$TEXT_RW_DIR" \
    --dpo-text-dir "$TEXT_DPO_DIR" \
    --output-dir "$TEXT_CONSOLIDATED_DIR" \
    > /dev/null 2>&1

echo -e "${GREEN}✅ Text-only evaluation complete!${NC}"
echo ""

# ============================================================================
# PART 2: VISION EVALUATION
# ============================================================================

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  PART 2/2: VISION PIPELINE (5 models × 2 samples)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Vision output directories
VISION_BASELINE_DIR="$VISION_OUTPUT_BASE/baseline"
VISION_STANDARD_DIR="$VISION_OUTPUT_BASE/standard_vision"
VISION_RL_DIR="$VISION_OUTPUT_BASE/rl_vision"
VISION_RW_DIR="$VISION_OUTPUT_BASE/rw_vision"
VISION_DPO_DIR="$VISION_OUTPUT_BASE/dpo_vision"
VISION_CONSOLIDATED_DIR="$VISION_OUTPUT_BASE/consolidated_vision"

# Create directories
mkdir -p "$VISION_BASELINE_DIR" "$VISION_STANDARD_DIR" "$VISION_RL_DIR" "$VISION_RW_DIR" "$VISION_DPO_DIR" "$VISION_CONSOLIDATED_DIR"

echo -e "${CYAN}[1/5] Evaluating Baseline (vision)...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$VISION_BASELINE_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[2/5] Evaluating Standard Vision...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$STANDARD_VISION_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$VISION_STANDARD_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[3/5] Evaluating RL Vision...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$RL_VISION_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$VISION_RL_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[4/5] Evaluating RW Vision...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$RW_VISION_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$VISION_RW_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo -e "${CYAN}[5/5] Evaluating DPO Vision...${NC}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_planner.py \
    --checkpoint "$DPO_VISION_CHECKPOINT" \
    --data "$DATA_PATH" \
    --split val \
    --num_samples $NUM_SAMPLES \
    --output "$VISION_DPO_DIR" \
    --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
    --hidream-config "$HIDREAM_CONFIG" \
    --save-images \
    --save-predictions \
    > /dev/null 2>&1

echo ""
echo -e "${CYAN}Consolidating vision results...${NC}"
bash scripts/evaluation/consolidate_vision_results.sh \
    --baseline-dir "$VISION_BASELINE_DIR" \
    --standard-vision-dir "$VISION_STANDARD_DIR" \
    --rl-vision-dir "$VISION_RL_DIR" \
    --rw-vision-dir "$VISION_RW_DIR" \
    --dpo-vision-dir "$VISION_DPO_DIR" \
    --output-dir "$VISION_CONSOLIDATED_DIR" \
    > /dev/null 2>&1

echo -e "${GREEN}✅ Vision evaluation complete!${NC}"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  VERIFICATION RESULTS${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check CLIP scores
echo -e "${YELLOW}Checking CLIP Scores:${NC}"
echo ""

echo -e "${CYAN}Text-Only Baseline:${NC}"
python3 << EOF
import json
with open('$TEXT_BASELINE_DIR/evaluation_summary_val.json') as f:
    data = json.load(f)
    image_metrics = data.get('aggregated_image_metrics', {})
    if 'image_clip_score_mean' in image_metrics:
        print(f"  ✅ CLIP score found: {image_metrics['image_clip_score_mean']:.4f}")
    elif any('clip_error' in k for k in image_metrics.keys()):
        print(f"  ❌ CLIP error found")
    else:
        print(f"  ❌ CLIP score missing")
EOF

echo -e "${CYAN}Vision Baseline:${NC}"
python3 << EOF
import json
with open('$VISION_BASELINE_DIR/evaluation_summary_val.json') as f:
    data = json.load(f)
    image_metrics = data.get('aggregated_image_metrics', {})
    if 'image_clip_score_mean' in image_metrics:
        print(f"  ✅ CLIP score found: {image_metrics['image_clip_score_mean']:.4f}")
    elif any('clip_error' in k for k in image_metrics.keys()):
        print(f"  ❌ CLIP error found")
    else:
        print(f"  ❌ CLIP score missing")
EOF

echo ""
echo -e "${YELLOW}Checking Generated Tables:${NC}"
echo ""

# Check text tables
if [ -f "$TEXT_CONSOLIDATED_DIR/planner_metrics_table.png" ]; then
    echo -e "  ✅ Text planner metrics table: ${TEXT_CONSOLIDATED_DIR}/planner_metrics_table.png"
else
    echo -e "  ❌ Text planner metrics table missing"
fi

if [ -f "$TEXT_CONSOLIDATED_DIR/image_metrics_table.png" ]; then
    echo -e "  ✅ Text image metrics table: ${TEXT_CONSOLIDATED_DIR}/image_metrics_table.png"
else
    echo -e "  ❌ Text image metrics table missing"
fi

# Check vision tables
if [ -f "$VISION_CONSOLIDATED_DIR/planner_metrics_table.png" ]; then
    echo -e "  ✅ Vision planner metrics table: ${VISION_CONSOLIDATED_DIR}/planner_metrics_table.png"
else
    echo -e "  ❌ Vision planner metrics table missing"
fi

if [ -f "$VISION_CONSOLIDATED_DIR/image_metrics_table.png" ]; then
    echo -e "  ✅ Vision image metrics table: ${VISION_CONSOLIDATED_DIR}/image_metrics_table.png"
else
    echo -e "  ❌ Vision image metrics table missing"
fi

echo ""
echo -e "${YELLOW}Generated Sample Images:${NC}"
echo ""

# Count text samples
TEXT_SAMPLE_COUNT=$(find "$TEXT_BASELINE_DIR/samples" -name "comparison.png" 2>/dev/null | wc -l)
echo -e "  📸 Text-only samples: $TEXT_SAMPLE_COUNT images"

# Count vision samples
VISION_SAMPLE_COUNT=$(find "$VISION_BASELINE_DIR/samples" -name "comparison.png" 2>/dev/null | wc -l)
echo -e "  📸 Vision samples: $VISION_SAMPLE_COUNT images"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  TEST COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Results saved to:${NC}"
echo -e "  Text:   $TEXT_OUTPUT_BASE"
echo -e "  Vision: $VISION_OUTPUT_BASE"
echo ""

