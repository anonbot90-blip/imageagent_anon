#!/bin/bash

# Generate Reward-Weighted Training Data
# Uses ALL samples with reward >= 3.0, assigns weights based on quality
# Higher quality samples get higher weights during training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_16000_cot"
OUTPUT_DIR_TEXT="$PROJECT_ROOT/training_data/cot_8b/rw_text"
OUTPUT_DIR_VISION="$PROJECT_ROOT/training_data/cot_8b/rw_vision"
REWARD_METRIC="overall_quality"
MIN_THRESHOLD=3.5 
NUM_SAMPLES=3600  # 5K samples for CoT 8B training
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
echo -e "${BLUE}  Reward-Weighted Training Data Generation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration:"
echo "  Results directory: $RESULTS_DIR"
echo "  Minimum threshold: $MIN_THRESHOLD"
echo "  Reward metric: $REWARD_METRIC"
echo "  Target samples: $NUM_SAMPLES"
echo "  Random seed: 42"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# ============================================================================
# Generate Text-Only RW Data
# ============================================================================

echo -e "${GREEN}[1/2] Generating Text-Only RW Data${NC}"
echo "  Output: $OUTPUT_DIR_TEXT"
if [ -f "$EXCLUDE_FILE" ]; then
    echo "  Excluding test samples from: $EXCLUDE_FILE"
fi
echo ""

CMD="python \"$PROJECT_ROOT/scripts/generate_planner_training_data.py\" \
    \"$RESULTS_DIR\" \
    --output-dir \"$OUTPUT_DIR_TEXT\" \
    --num-datapoints $NUM_SAMPLES \
    --rl-data \
    --threshold $MIN_THRESHOLD \
    --reward-metric \"$REWARD_METRIC\""

if [ -f "$EXCLUDE_FILE" ]; then
    CMD="$CMD --exclude-file \"$EXCLUDE_FILE\""
fi

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Text-only RW data generated successfully${NC}"
    
    # Show sample count
    if [ -f "$OUTPUT_DIR_TEXT/planner_training_data.json" ]; then
        SAMPLE_COUNT=$(jq '.total_samples' "$OUTPUT_DIR_TEXT/planner_training_data.json")
        echo "  Total samples: $SAMPLE_COUNT"
    fi
else
    echo "❌ Failed to generate text-only RW data"
    exit 1
fi

# ============================================================================
# Generate Vision RW Data (for cached training)
# ============================================================================

echo ""
echo -e "${GREEN}[2/2] Generating Vision RW Data${NC}"
echo "  Output: $OUTPUT_DIR_VISION"
if [ -f "$EXCLUDE_FILE" ]; then
    echo "  Excluding test samples from: $EXCLUDE_FILE"
fi
echo ""

CMD="python \"$PROJECT_ROOT/scripts/generate_planner_training_data.py\" \
    \"$RESULTS_DIR\" \
    --output-dir \"$OUTPUT_DIR_VISION\" \
    --num-datapoints $NUM_SAMPLES \
    --rl-data \
    --threshold $MIN_THRESHOLD \
    --reward-metric \"$REWARD_METRIC\""

if [ -f "$EXCLUDE_FILE" ]; then
    CMD="$CMD --exclude-file \"$EXCLUDE_FILE\""
fi

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Vision RW data generated successfully${NC}"
    
    # Show sample count
    if [ -f "$OUTPUT_DIR_VISION/planner_training_data.json" ]; then
        SAMPLE_COUNT=$(jq '.total_samples' "$OUTPUT_DIR_VISION/planner_training_data.json")
        echo "  Total samples: $SAMPLE_COUNT"
    fi
else
    echo "❌ Failed to generate vision RW data"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Reward-Weighted Data Generation Complete${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Generated datasets:"
echo "  1. Text-only: $OUTPUT_DIR_TEXT/planner_training_data.json"
echo "  2. Vision:    $OUTPUT_DIR_VISION/planner_training_data.json"
echo ""
echo "Next steps:"
echo "  1. Precompute vision embeddings:"
echo "     bash scripts/precompute_image_embeddings.sh \\"
echo "       --data-path $OUTPUT_DIR_VISION/planner_training_data.json \\"
echo "       --output-path $OUTPUT_DIR_VISION/embeddings/vision_embeddings.h5"
echo ""
echo "  2. Train RW models:"
echo "     bash scripts/training/start_planner_training_text_only_rw.sh"
echo "     bash scripts/training/start_planner_training_vision_rw.sh"
echo ""

