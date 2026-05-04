#!/bin/bash

# Trajectory-Based Training Pipeline (Simplified Approach)
# Uses existing training infrastructure with trajectory-based train/test split

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_16000_cot"
TRAJECTORY_DATA_DIR="$PROJECT_ROOT/training_data/cot_8b_trajectory"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/cot_8b_trajectory"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  Trajectory-Based Training Pipeline${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# Step 1: Verify trajectory split exists
# ============================================================================

echo -e "${BLUE}Step 1/5: Verifying trajectory split...${NC}"
echo ""

if [ ! -f "$TRAJECTORY_DATA_DIR/test_samples_cot_8b.txt" ]; then
    echo -e "${YELLOW}⚠️  Trajectory split not found. Creating split...${NC}"
    
    mkdir -p "$TRAJECTORY_DATA_DIR"
    
    python "$PROJECT_ROOT/scripts/training/split_trajectories.py" \
        --results-dir "$RESULTS_DIR" \
        --test-threshold 2.5 \
        --test-count 200 \
        --reward-metric overall_quality \
        --output-dir "$TRAJECTORY_DATA_DIR" \
        --prefix cot_8b
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create trajectory split${NC}"
        exit 1
    fi
fi

TRAIN_COUNT=$(wc -l < "$TRAJECTORY_DATA_DIR/train_samples_cot_8b.txt")
TEST_COUNT=$(wc -l < "$TRAJECTORY_DATA_DIR/test_samples_cot_8b.txt")

echo -e "${GREEN}✓ Trajectory split exists${NC}"
echo -e "  Train samples: $TRAIN_COUNT"
echo -e "  Test samples: $TEST_COUNT"
echo ""

# ============================================================================
# Step 2: Generate Standard Training Data
# ============================================================================

echo -e "${BLUE}Step 2/5: Generating Standard training data...${NC}"
echo ""

STANDARD_DIR="$TRAJECTORY_DATA_DIR/standard_text"
mkdir -p "$STANDARD_DIR"

python "$PROJECT_ROOT/scripts/generate_planner_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$STANDARD_DIR" \
    --exclude-file "$TRAJECTORY_DATA_DIR/test_samples_cot_8b.txt" \
    --num-datapoints 3600 \
    --equal-dist

echo -e "${GREEN}✓ Standard training data generated${NC}"
echo ""

# ============================================================================
# Step 3: Generate RL Training Data
# ============================================================================

echo -e "${BLUE}Step 3/5: Generating RL training data...${NC}"
echo ""

RL_DIR="$TRAJECTORY_DATA_DIR/rl_text"
mkdir -p "$RL_DIR"

python "$PROJECT_ROOT/scripts/generate_planner_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$RL_DIR" \
    --exclude-file "$TRAJECTORY_DATA_DIR/test_samples_cot_8b.txt" \
    --rl-data \
    --reward-threshold 3.0 \
    --max-samples 3600

echo -e "${GREEN}✓ RL training data generated${NC}"
echo ""

# ============================================================================
# Step 4: Generate RW Training Data
# ============================================================================

echo -e "${BLUE}Step 4/5: Generating RW training data...${NC}"
echo ""

RW_DIR="$TRAJECTORY_DATA_DIR/rw_text"
mkdir -p "$RW_DIR"

python "$PROJECT_ROOT/scripts/generate_planner_training_data.py" \
    "$RESULTS_DIR" \
    --output-dir "$RW_DIR" \
    --exclude-file "$TRAJECTORY_DATA_DIR/test_samples_cot_8b.txt" \
    --rl-data \
    --reward-threshold 3.5 \
    --max-samples 3600

echo -e "${GREEN}✓ RW training data generated${NC}"
echo ""

# ============================================================================
# Step 5: Generate DPO Training Data
# ============================================================================

echo -e "${BLUE}Step 5/5: Generating DPO training data...${NC}"
echo ""

DPO_DIR="$TRAJECTORY_DATA_DIR/dpo_text"
mkdir -p "$DPO_DIR"

python "$PROJECT_ROOT/scripts/training/generate_dpo_pairs.py" \
    --results-dir "$RESULTS_DIR" \
    --output-text "$DPO_DIR" \
    --chosen-threshold 4.0 \
    --rejected-min 2.5 \
    --rejected-max 3.5 \
    --exclude-file "$TRAJECTORY_DATA_DIR/test_samples_cot_8b.txt"

echo -e "${GREEN}✓ DPO training data generated${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All Training Data Generated!${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo "Training data directories:"
echo "  Standard: $STANDARD_DIR"
echo "  RL:       $RL_DIR"
echo "  RW:       $RW_DIR"
echo "  DPO:      $DPO_DIR"
echo ""

echo "Checkpoint directory (to be created during training):"
echo "  $CHECKPOINT_DIR"
echo ""

echo -e "${CYAN}Next steps:${NC}"
echo "1. Train models using existing scripts with trajectory data:"
echo "   bash scripts/training/start_planner_training_text_only.sh \\"
echo "       --data-dir training_data/cot_8b_trajectory \\"
echo "       --checkpoint-dir checkpoints/cot_8b_trajectory"
echo ""
echo "2. Or train manually:"
echo "   cd training/planner_training"
echo "   # Update config files to point to trajectory data directories"
echo "   bash ../../scripts/training/start_planner_training_text_only.sh"
echo ""
echo -e "${GREEN}✓ Ready for training!${NC}"
echo ""

