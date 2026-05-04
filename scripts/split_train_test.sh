#!/bin/bash

# Split 16,000 samples into Train (15,600) and Test (400)
# Creates test_samples_cot_8b.txt with 400 randomly selected sample IDs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR="$PROJECT_ROOT/imageagent_results_16000_cot"
OUTPUT_FILE="$PROJECT_ROOT/training_data/cot_8b/test_samples_cot_8b.txt"
NUM_TEST_SAMPLES=400
RANDOM_SEED=42

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# ============================================================================
# Main Script
# ============================================================================

echo ""
print_color $PURPLE "╔══════════════════════════════════════════════════════════════════════════════╗"
print_color $PURPLE "║                  Train/Test Split for 16K CoT Dataset                        ║"
print_color $PURPLE "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    print_color $RED "❌ Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Count total samples
print_color $CYAN "📊 Counting samples in $RESULTS_DIR..."
TOTAL_SAMPLES=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "image_*" | wc -l)
print_color $GREEN "✓ Found $TOTAL_SAMPLES samples"
echo ""

if [ "$TOTAL_SAMPLES" -lt "$NUM_TEST_SAMPLES" ]; then
    print_color $RED "❌ Error: Not enough samples! Found $TOTAL_SAMPLES, need at least $NUM_TEST_SAMPLES"
    exit 1
fi

# Calculate train samples
NUM_TRAIN_SAMPLES=$((TOTAL_SAMPLES - NUM_TEST_SAMPLES))

print_color $BLUE "Configuration:"
print_color $BLUE "  Total samples: $TOTAL_SAMPLES"
print_color $BLUE "  Test samples: $NUM_TEST_SAMPLES (2.5%)"
print_color $BLUE "  Train samples: $NUM_TRAIN_SAMPLES (97.5%)"
print_color $BLUE "  Random seed: $RANDOM_SEED"
print_color $BLUE "  Output file: $OUTPUT_FILE"
echo ""

# Check if test_samples.txt already exists
if [ -f "$OUTPUT_FILE" ]; then
    print_color $YELLOW "⚠️  Warning: $OUTPUT_FILE already exists!"
    print_color $YELLOW "   This will overwrite the existing split."
    print_color $YELLOW "   If you've already generated training data, you should NOT re-run this."
    echo ""
    read -p "Continue and overwrite? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        print_color $CYAN "Aborted. Using existing split."
        exit 0
    fi
    echo ""
fi

# Get all sample directories
print_color $CYAN "🔍 Collecting all sample directories..."
ALL_SAMPLES=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "image_*" -printf "%f\n" | sort)
SAMPLE_COUNT=$(echo "$ALL_SAMPLES" | wc -l)
print_color $GREEN "✓ Collected $SAMPLE_COUNT sample IDs"
echo ""

# Randomly select test samples with fixed seed
print_color $CYAN "🎲 Randomly selecting $NUM_TEST_SAMPLES test samples (seed=$RANDOM_SEED)..."
echo "$ALL_SAMPLES" | shuf --random-source=<(yes $RANDOM_SEED) | head -n $NUM_TEST_SAMPLES > "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    print_color $RED "❌ Error: Failed to create test split"
    exit 1
fi

TEST_COUNT=$(wc -l < "$OUTPUT_FILE")
print_color $GREEN "✓ Selected $TEST_COUNT test samples"
echo ""

# Display summary
print_color $PURPLE "╔══════════════════════════════════════════════════════════════════════════════╗"
print_color $PURPLE "║                              Split Complete!                                 ║"
print_color $PURPLE "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

print_color $GREEN "✅ Train/Test split created successfully!"
echo ""
print_color $CYAN "📊 Summary:"
print_color $CYAN "   Total samples: $TOTAL_SAMPLES"
print_color $CYAN "   Train samples: $NUM_TRAIN_SAMPLES (97.5%)"
print_color $CYAN "   Test samples: $TEST_COUNT (2.5%)"
print_color $CYAN "   Test IDs saved to: $OUTPUT_FILE"
echo ""

print_color $YELLOW "⚠️  IMPORTANT:"
print_color $YELLOW "   - Do NOT delete or modify test_samples_cot_8b.txt"
print_color $YELLOW "   - All training data generation will exclude these 400 samples"
print_color $YELLOW "   - All evaluation will use ONLY these 400 samples"
print_color $YELLOW "   - This ensures no train/test contamination"
echo ""

# Show first few test samples
print_color $CYAN "📋 First 10 test samples:"
head -n 10 "$OUTPUT_FILE" | while read line; do
    echo "   - $line"
done
echo ""

print_color $GREEN "✅ Ready to proceed with training data generation!"
echo ""

