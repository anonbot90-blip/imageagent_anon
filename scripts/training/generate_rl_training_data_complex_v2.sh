#!/bin/bash

# Generate RL Training Data for Complex V2 Dataset
# Selects samples from trajectories with avg reward >= 3.0
# Supports both 8B and 4B models via TRAJECTORY_PREFIX

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/imageagent_results_complex_v2_10k_cot}"
TRAJECTORY_PREFIX="${TRAJECTORY_PREFIX:-complex_v2_cot_8b}"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/${TRAJECTORY_PREFIX}_trajectory"
OUTPUT_DIR_TEXT="$TRAJECTORY_DIR/rl_text"
OUTPUT_DIR_VISION="$TRAJECTORY_DIR/rl_vision"
REWARD_THRESHOLD=3.0
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
echo -e "${PURPLE}  Generate RL Training Data - Complex V2${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Trajectory Prefix: $TRAJECTORY_PREFIX"
echo "Results Directory: $RESULTS_DIR"
echo "Reward Threshold: $REWARD_THRESHOLD"
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
echo -e "  RL Filter: $REWARD_METRIC >= $REWARD_THRESHOLD"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_TEXT"
mkdir -p "$OUTPUT_DIR_VISION"

# ============================================================================
# Generate Text Training Data
# ============================================================================

echo -e "${BLUE}Step 1/2: Generating RL text training data...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/generate_planner_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR_TEXT" \
    --exclude-file "$EXCLUDE_FILE" \
    --rl-data \
    --threshold $REWARD_THRESHOLD \
    --reward-metric "$REWARD_METRIC"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to generate text training data${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Text training data generated${NC}"
echo ""

# ============================================================================
# Generate Vision Training Data
# ============================================================================

echo -e "${BLUE}Step 2/2: Generating vision training data...${NC}"
echo ""

mkdir -p "$OUTPUT_DIR_VISION"
cp "$OUTPUT_DIR_TEXT/planner_training_data.json" "$OUTPUT_DIR_VISION/"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to copy vision training data${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Vision training data generated (copied from text)${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ RL Training Data Generated - Complex V2${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output directories:"
echo "  Text:   $OUTPUT_DIR_TEXT"
echo "  Vision: $OUTPUT_DIR_VISION"
echo ""
echo "Strategy: Trajectories with avg reward >= $REWARD_THRESHOLD"
echo "Trajectory Prefix: $TRAJECTORY_PREFIX"
echo ""

