#!/bin/bash

# Generate Standardized Weighted Training Data (Trajectory-Based)
# Includes ALL samples (threshold=0.0) with trajectory-level standardized weighting
# Applies z-score normalized weights based on trajectory average reward

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/imageagent_results_10000_cot}"
TRAJECTORY_PREFIX="${TRAJECTORY_PREFIX:-simple_10k_cot}"
# Detect size from TRAJECTORY_PREFIX (cot_4b or cot_8b)
if [[ "$TRAJECTORY_PREFIX" == *"4b"* ]]; then
    TRAJECTORY_SIZE="4b"
else
    TRAJECTORY_SIZE="8b"
fi
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/simple/cot_${TRAJECTORY_SIZE}_trajectory"
OUTPUT_DIR_TEXT="$TRAJECTORY_DIR/sw_text"
OUTPUT_DIR_VISION="$TRAJECTORY_DIR/sw_vision"
REWARD_THRESHOLD=0.0  # Include ALL samples
REWARD_METRIC="overall_quality"
EXCLUDE_FILE="$TRAJECTORY_DIR/test_samples_cot_${TRAJECTORY_SIZE}.txt"
TRAJECTORY_REWARDS_FILE="$TRAJECTORY_DIR/trajectory_rewards_cot_${TRAJECTORY_SIZE}.json"

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
echo -e "${PURPLE}  Generate SW Training Data (Trajectory-Based Standardized)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Verify trajectory split exists
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo -e "${YELLOW}❌ Error: Trajectory split not found at: $EXCLUDE_FILE${NC}"
    echo "Please run: python scripts/training/split_trajectories.py first"
    exit 1
fi

if [ ! -f "$TRAJECTORY_REWARDS_FILE" ]; then
    echo -e "${YELLOW}❌ Error: Trajectory rewards not found at: $TRAJECTORY_REWARDS_FILE${NC}"
    echo "Please run: python scripts/training/split_trajectories.py first"
    exit 1
fi

TEST_COUNT=$(wc -l < "$EXCLUDE_FILE")
echo -e "${GREEN}✓ Using trajectory test exclusion file${NC}"
echo -e "  Excluding: $TEST_COUNT test samples"
echo -e "  SW Filter: $REWARD_METRIC >= $REWARD_THRESHOLD (includes ALL samples)"
echo -e "  Applies standardized trajectory-based weighting (z-scores)"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_TEXT"
mkdir -p "$OUTPUT_DIR_VISION"

# ============================================================================
# Generate SW Training Data with Standardized Weights
# ============================================================================

echo -e "${BLUE}Generating SW training data with standardized weights...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/training/generate_sw_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR_TEXT" \
    --exclude-file "$EXCLUDE_FILE" \
    --trajectory-rewards-file "$TRAJECTORY_REWARDS_FILE" \
    --threshold $REWARD_THRESHOLD \
    --reward-metric "$REWARD_METRIC"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to generate SW training data${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SW text training data generated${NC}"
echo ""

# ============================================================================
# Generate Vision Training Data
# ============================================================================

echo -e "${BLUE}Generating SW vision training data...${NC}"
echo ""

# Simply copy text data to vision directory (vision embeddings computed separately)
mkdir -p "$OUTPUT_DIR_VISION"
cp "$OUTPUT_DIR_TEXT/planner_training_data.json" "$OUTPUT_DIR_VISION/"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}❌ Failed to copy vision training data${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SW vision training data generated (copied from text)${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ SW Training Data Generated (Trajectory-Based Standardized)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output directories:"
echo "  Text:   $OUTPUT_DIR_TEXT"
echo "  Vision: $OUTPUT_DIR_VISION"
echo ""
echo "Strategy: ALL samples from ALL trajectories (threshold = 0.0)"
echo "Weighting: Standardized (z-score) based on trajectory average rewards"
echo "  - Above-average trajectories get positive weights"
echo "  - Below-average trajectories get negative weights"
echo "  - Weights follow standard normal distribution ~N(0,1)"
echo ""


