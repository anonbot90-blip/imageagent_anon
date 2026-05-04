#!/bin/bash

# Generate Standardized-Weighted (SW) Training Data (Trajectory-Based) - Complex Theme
# Uses z-score normalization for continuous weighting (including negative weights)
# Applies weights based on trajectory average reward with z-score normalization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/imageagent_results_10000_cot_test}"
TRAJECTORY_PREFIX="${TRAJECTORY_PREFIX:-normal_complex_test_8b}"
# Detect size from TRAJECTORY_PREFIX (cot_4b or cot_8b)
if [[ "$TRAJECTORY_PREFIX" == *"4b"* ]]; then
    TRAJECTORY_SIZE="4b"
else
    TRAJECTORY_SIZE="8b"
fi
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/normal/cot_${TRAJECTORY_SIZE}_trajectory"
OUTPUT_DIR_TEXT="$TRAJECTORY_DIR/sw_text"
OUTPUT_DIR_VISION="$TRAJECTORY_DIR/sw_vision"
REWARD_THRESHOLD=0.0
REWARD_METRIC="overall_quality"
EXCLUDE_FILE="$TRAJECTORY_DIR/test_samples_cot_${TRAJECTORY_SIZE}.txt"
TRAJECTORY_REWARDS_FILE="$TRAJECTORY_DIR/trajectory_rewards_${TRAJECTORY_PREFIX}.json"

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
echo -e "${PURPLE}  Generate SW Training Data (Trajectory-Based)${NC}"
echo -e "${PURPLE}  Complex Theme Dataset${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Verify trajectory split exists
if [ ! -f "$EXCLUDE_FILE" ]; then
    echo -e "${YELLOW}❌ Error: Trajectory split not found at: $EXCLUDE_FILE${NC}"
    echo "Please run: python scripts/training/complex_theme/split_trajectories.py first"
    exit 1
fi

if [ ! -f "$TRAJECTORY_REWARDS_FILE" ]; then
    echo -e "${YELLOW}❌ Error: Trajectory rewards not found at: $TRAJECTORY_REWARDS_FILE${NC}"
    echo "Please run: python scripts/training/complex_theme/split_trajectories.py first"
    exit 1
fi

TEST_COUNT=$(wc -l < "$EXCLUDE_FILE")
echo -e "${GREEN}✓ Using trajectory test exclusion file${NC}"
echo -e "  Excluding: $TEST_COUNT test samples"
echo -e "  SW: Z-score normalized weights (mean=0, std=1)"
echo -e "  Applies trajectory-based standardized weighting to all samples"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_TEXT"
mkdir -p "$OUTPUT_DIR_VISION"

# ============================================================================
# Generate Text Training Data
# ============================================================================

echo -e "${BLUE}Step 1/2: Generating SW text training data...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/training/generate_sw_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR_TEXT" \
    --exclude-file "$EXCLUDE_FILE" \
    --trajectory-rewards-file "$TRAJECTORY_REWARDS_FILE" \
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

echo -e "${BLUE}Step 2/2: Generating SW vision training data...${NC}"
echo ""

# Simply copy text data to vision directory (vision embeddings computed separately)
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
echo -e "${GREEN}✓ SW Training Data Generated (Trajectory-Based)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output directories:"
echo "  Text:   $OUTPUT_DIR_TEXT"
echo "  Vision: $OUTPUT_DIR_VISION"
echo ""
echo "Strategy: Z-score normalized weights (mean=0, std=1)"
echo "Note: Applies trajectory-based standardized weighting (can be negative)"
echo ""

