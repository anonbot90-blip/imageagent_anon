#!/bin/bash

# Generate DPO Training Data (Trajectory-Based)
# Creates (chosen, rejected) preference pairs at trajectory level
# Chosen: trajectories with avg >= 4.0
# Rejected: trajectories with avg 2.5-3.5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/imageagent_results_16000_cot}"
TRAJECTORY_PREFIX="${TRAJECTORY_PREFIX:-cot_8b}"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/${TRAJECTORY_PREFIX}_trajectory"
OUTPUT_DIR_TEXT="$TRAJECTORY_DIR/dpo_text"
OUTPUT_DIR_VISION="$TRAJECTORY_DIR/dpo_vision"
CHOSEN_THRESHOLD=4.0
REJECTED_MIN=2.5
REJECTED_MAX=3.5
REWARD_METRIC="overall_quality"
EXCLUDE_FILE="$TRAJECTORY_DIR/test_samples_${TRAJECTORY_PREFIX}.txt"

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
echo -e "${PURPLE}  Generate DPO Training Data (Trajectory-Based)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Verify trajectory split exists
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo -e "${YELLOW}❌ Error: Trajectory split not found at: $EXCLUDE_FILE${NC}"
    echo "Please run: python scripts/training/split_trajectories.py first"
    exit 1
fi

TEST_COUNT=$(wc -l < "$EXCLUDE_FILE")
echo -e "${GREEN}✓ Using trajectory test exclusion file${NC}"
echo -e "  Excluding: $TEST_COUNT test samples"
echo -e "  Chosen: $REWARD_METRIC >= $CHOSEN_THRESHOLD"
echo -e "  Rejected: $REJECTED_MIN <= $REWARD_METRIC <= $REJECTED_MAX"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_TEXT"
mkdir -p "$OUTPUT_DIR_VISION"

# ============================================================================
# Generate Text DPO Pairs
# ============================================================================

echo -e "${BLUE}Step 1/2: Generating DPO text preference pairs...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/training/generate_dpo_pairs.py" \
    "$RESULTS_DIR" \
    --output-dir-text "$OUTPUT_DIR_TEXT" \
    --reward-metric "$REWARD_METRIC" \
    --chosen-threshold $CHOSEN_THRESHOLD \
    --rejected-min $REJECTED_MIN \
    --rejected-max $REJECTED_MAX \
    --exclude-file "$EXCLUDE_FILE"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to generate text DPO pairs${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Text DPO pairs generated${NC}"
echo ""

# ============================================================================
# Generate Vision DPO Pairs
# ============================================================================

echo -e "${BLUE}Step 2/2: Generating DPO vision preference pairs...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/training/generate_dpo_pairs.py" \
    "$RESULTS_DIR" \
    --output-dir-vision "$OUTPUT_DIR_VISION" \
    --reward-metric "$REWARD_METRIC" \
    --chosen-threshold $CHOSEN_THRESHOLD \
    --rejected-min $REJECTED_MIN \
    --rejected-max $REJECTED_MAX \
    --exclude-file "$EXCLUDE_FILE"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to generate vision DPO pairs${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Vision DPO pairs generated${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ DPO Training Data Generated (Trajectory-Based)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output directories:"
echo "  Text:   $OUTPUT_DIR_TEXT"
echo "  Vision: $OUTPUT_DIR_VISION"
echo ""
echo "Strategy: Trajectory-level preference pairs"
echo "  Chosen trajectories:  $REWARD_METRIC >= $CHOSEN_THRESHOLD"
echo "  Rejected trajectories: $REJECTED_MIN <= $REWARD_METRIC <= $REJECTED_MAX"
echo ""

