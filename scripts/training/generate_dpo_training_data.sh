#!/bin/bash

# Generate DPO (Direct Preference Optimization) Training Data
# Creates (chosen, rejected) preference pairs for the same prompt/image
# Chosen: high-quality plans (reward >= 4.5, typically 5/5 scores)
# Rejected: mediocre plans (reward 2.5-3.5, typically 3/5 scores)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_16000_cot"
OUTPUT_DIR_TEXT="$PROJECT_ROOT/training_data/cot_8b/dpo_text"
OUTPUT_DIR_VISION="$PROJECT_ROOT/training_data/cot_8b/dpo_vision"
REWARD_METRIC="overall_quality"
CHOSEN_THRESHOLD=4.0   # High-quality plans (5/5 scores)
REJECTED_MIN=2.5       # Low end of rejected range (3/5 scores)
REJECTED_MAX=3.5       # High end of rejected range (3/5 scores)
EXCLUDE_FILE="$PROJECT_ROOT/test_samples_cot_8b.txt"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Main Script
# ============================================================================

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  DPO Preference Pairs Generation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration:"
echo "  Results directory: $RESULTS_DIR"
echo "  Chosen threshold: >= $CHOSEN_THRESHOLD"
echo "  Rejected range: $REJECTED_MIN - $REJECTED_MAX"
echo "  Reward metric: $REWARD_METRIC"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# ============================================================================
# Generate DPO Pairs
# ============================================================================

echo -e "${GREEN}Generating DPO preference pairs...${NC}"
if [ -f "$EXCLUDE_FILE" ]; then
    echo "  Excluding test samples from: $EXCLUDE_FILE"
fi
echo ""

CMD="python \"$PROJECT_ROOT/scripts/training/generate_dpo_pairs.py\" \
    \"$RESULTS_DIR\" \
    --output-dir-text \"$OUTPUT_DIR_TEXT\" \
    --output-dir-vision \"$OUTPUT_DIR_VISION\" \
    --chosen-threshold $CHOSEN_THRESHOLD \
    --rejected-min $REJECTED_MIN \
    --rejected-max $REJECTED_MAX \
    --reward-metric \"$REWARD_METRIC\""

if [ -f "$EXCLUDE_FILE" ]; then
    CMD="$CMD --exclude-file \"$EXCLUDE_FILE\""
fi

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ DPO preference pairs generated successfully${NC}"
else
    echo "❌ Failed to generate DPO preference pairs"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ DPO Data Generation Complete${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Generated datasets:"
echo "  1. Text-only: $OUTPUT_DIR_TEXT/planner_training_data_dpo.json"
echo "  2. Vision:    $OUTPUT_DIR_VISION/planner_training_data_dpo.json"
echo ""
echo "Next steps:"
echo "  1. Analyze pair quality:"
echo "     python scripts/training/analyze_dpo_pairs.py \\"
echo "       $OUTPUT_DIR_TEXT/planner_training_data_dpo.json"
echo ""
echo "  2. Precompute vision embeddings for BOTH chosen and rejected:"
echo "     bash scripts/precompute_image_embeddings.sh \\"
echo "       --data-path $OUTPUT_DIR_VISION/planner_training_data_dpo.json \\"
echo "       --output-path $OUTPUT_DIR_VISION/embeddings/vision_embeddings.h5"
echo ""
echo "  3. Train DPO models:"
echo "     bash scripts/training/start_planner_training_text_only_dpo.sh"
echo "     bash scripts/training/start_planner_training_vision_dpo.sh"
echo ""

