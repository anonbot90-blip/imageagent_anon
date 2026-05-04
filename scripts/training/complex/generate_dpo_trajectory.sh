#!/bin/bash

# Generate DPO Training Data for Complex V2 Dataset
# Creates chosen/rejected trajectory pairs
# Chosen: Trajectories with avg reward >= 4.0
# Rejected: Trajectories with avg reward 2.5-3.5
# Supports both 8B and 4B models via TRAJECTORY_PREFIX

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/imageagent_results_complex_v2_10k_cot_10k_cot}"
TRAJECTORY_PREFIX="${TRAJECTORY_PREFIX:-complex_v2_10k_cot_8b}"
# Detect size from TRAJECTORY_PREFIX (cot_4b or cot_8b)
if [[ "$TRAJECTORY_PREFIX" == *"4b"* ]]; then
    TRAJECTORY_SIZE="4b"
else
    TRAJECTORY_SIZE="8b"
fi
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/complex/cot_${TRAJECTORY_SIZE}_trajectory"
OUTPUT_DIR_TEXT="$TRAJECTORY_DIR/dpo_text"
OUTPUT_DIR_VISION="$TRAJECTORY_DIR/dpo_vision"
EXCLUDE_FILE="$TRAJECTORY_DIR/test_samples_cot_${TRAJECTORY_SIZE}.txt"

# DPO thresholds
CHOSEN_THRESHOLD=4.0
REJECTED_MIN=2.5
REJECTED_MAX=3.5
MIN_SCORE_DIFF=0.5

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# ============================================================================
# Main
# ============================================================================

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  Generate DPO Training Data - Complex V2${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Trajectory Prefix: $TRAJECTORY_PREFIX"
echo "Results Directory: $RESULTS_DIR"
echo "Chosen Threshold: >= $CHOSEN_THRESHOLD"
echo "Rejected Range: $REJECTED_MIN - $REJECTED_MAX"
echo ""

# Verify results directory
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${YELLOW}❌ Error: Results directory not found: $RESULTS_DIR${NC}"
    exit 1
fi

# Verify trajectory split exists
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo -e "${YELLOW}❌ Error: Trajectory split not found at: $EXCLUDE_FILE${NC}"
    echo "Please run split_trajectories.py first"
    exit 1
fi

TEST_COUNT=$(wc -l < "$EXCLUDE_FILE")
echo -e "${GREEN}✓ Using trajectory test exclusion file${NC}"
echo -e "  Excluding: $TEST_COUNT test samples"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_TEXT"
mkdir -p "$OUTPUT_DIR_VISION"

# ============================================================================
# Generate DPO Pairs
# ============================================================================

echo -e "${BLUE}Step 1/1: Generating DPO preference pairs...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/training/generate_dpo_pairs.py" \
    "$RESULTS_DIR" \
    --output-dir-text "$OUTPUT_DIR_TEXT" \
    --output-dir-vision "$OUTPUT_DIR_VISION" \
    --exclude-file "$EXCLUDE_FILE" \
    --chosen-threshold $CHOSEN_THRESHOLD \
    --rejected-min $REJECTED_MIN \
    --rejected-max $REJECTED_MAX

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to generate DPO pairs${NC}"
    exit 1
fi

echo -e "${GREEN}✓ DPO pairs generated for both text and vision${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ DPO Training Data Generated - Complex V2${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output directories:"
echo "  Text:   $OUTPUT_DIR_TEXT"
echo "  Vision: $OUTPUT_DIR_VISION"
echo ""
echo "Strategy: Chosen/Rejected trajectory pairs"
echo "  Chosen: Trajectories with avg reward >= $CHOSEN_THRESHOLD"
echo "  Rejected: Trajectories with avg reward $REJECTED_MIN - $REJECTED_MAX"
echo "  Min score difference: $MIN_SCORE_DIFF"
echo "Trajectory Prefix: $TRAJECTORY_PREFIX"
echo ""

