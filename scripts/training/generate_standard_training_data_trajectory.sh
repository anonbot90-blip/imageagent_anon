#!/bin/bash

# Generate Standard Training Data (Trajectory-Based)
# Selects 1 random sample per trajectory from train set
# Equal distribution across quality levels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/imageagent_results_16000_cot}"
TRAJECTORY_PREFIX="${TRAJECTORY_PREFIX:-cot_8b}"
TRAJECTORY_DIR="$PROJECT_ROOT/training_data/${TRAJECTORY_PREFIX}_trajectory"
OUTPUT_DIR_TEXT="$TRAJECTORY_DIR/standard_text"
OUTPUT_DIR_VISION="$TRAJECTORY_DIR/standard_vision"
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
echo -e "${PURPLE}  Generate Standard Training Data (Trajectory-Based)${NC}"
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
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_TEXT"
mkdir -p "$OUTPUT_DIR_VISION"

# ============================================================================
# Generate Text Training Data
# ============================================================================

echo -e "${BLUE}Step 1/2: Generating text-only training data...${NC}"
echo ""

python "$PROJECT_ROOT/scripts/generate_planner_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR_TEXT" \
    --exclude-file "$EXCLUDE_FILE"

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
echo -e "${GREEN}✓ Standard Training Data Generated (Trajectory-Based)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output directories:"
echo "  Text:   $OUTPUT_DIR_TEXT"
echo "  Vision: $OUTPUT_DIR_VISION"
echo ""
echo "Strategy: All samples from train trajectories"
echo ""

