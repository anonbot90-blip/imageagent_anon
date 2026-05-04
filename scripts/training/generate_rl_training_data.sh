#!/bin/bash

# Generate RL Training Data with Reward Filtering
# Filters training data based on reward scores for high-quality RL training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default values
DEFAULT_THRESHOLD=3.5  # Keep samples with scores >= 3.0 (consistent with RW)
DEFAULT_NUM_DATAPOINTS=20000
DEFAULT_METRIC="overall_quality"
DEFAULT_INPUT_DIR="$PROJECT_ROOT/imageagent_results_40000_no_cot"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/training_data_40000_no_cot/rl"
DEFAULT_EXCLUDE_FILE="$PROJECT_ROOT/test_samples.txt"

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

# Initialize variables
THRESHOLD=$DEFAULT_THRESHOLD
NUM_DATAPOINTS=$DEFAULT_NUM_DATAPOINTS
METRIC=$DEFAULT_METRIC
INPUT_DIR=$DEFAULT_INPUT_DIR
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
EXCLUDE_FILE=$DEFAULT_EXCLUDE_FILE

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold|-t)
            THRESHOLD="$2"
            shift 2
            ;;
        --num-datapoints|-n)
            NUM_DATAPOINTS="$2"
            shift 2
            ;;
        --exclude-file)
            EXCLUDE_FILE="$2"
            shift 2
            ;;
        --metric|-m)
            METRIC="$2"
            shift 2
            ;;
        --input-dir|-i)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            print_color $CYAN "Usage: $0 [OPTIONS]"
            echo ""
            print_color $CYAN "Options:"
            echo "  -t, --threshold FLOAT       Minimum reward score (default: 3.0)"
            echo "  -n, --num-datapoints INT    Max number of samples (default: 1000)"
            echo "  -m, --metric STRING         Reward metric to use (default: overall_quality)"
            echo "  -i, --input-dir PATH        Input directory (default: imageagent_results_5000)"
            echo "  -o, --output-dir PATH       Output directory (default: training_data/rl)"
            echo "  -h, --help                  Show this help message"
            echo ""
            print_color $CYAN "Available metrics:"
            echo "  - overall_quality"
            echo "  - action_plan_quality"
            echo "  - adherence_to_prompt"
            echo "  - adherence_to_plan"
            echo "  - plan_reasoning"
            echo "  - final_image_quality"
            exit 0
            ;;
        *)
            print_color $RED "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
print_color $PURPLE "============================================================================"
print_color $PURPLE "🏆 RL Training Data Generation"
print_color $PURPLE "============================================================================"
echo ""

# Print configuration
print_color $CYAN "📝 Configuration:"
print_color $CYAN "   Input: $INPUT_DIR"
print_color $CYAN "   Output: $OUTPUT_DIR"
print_color $CYAN "   Threshold: $THRESHOLD"
print_color $CYAN "   Max samples: $NUM_DATAPOINTS"
print_color $CYAN "   Metric: $METRIC"
echo ""

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    print_color $RED "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check for reward_scores.json files
REWARD_COUNT=$(find "$INPUT_DIR" -name "reward_scores.json" 2>/dev/null | wc -l)
if [ "$REWARD_COUNT" -eq 0 ]; then
    print_color $RED "❌ Error: No reward_scores.json files found in $INPUT_DIR"
    print_color $YELLOW "💡 Make sure you ran the pipeline with --reward-model flag"
    exit 1
fi

print_color $GREEN "✓ Found $REWARD_COUNT images with reward scores"
echo ""

# Run Python script
print_color $BLUE "🚀 Generating RL training data..."
echo ""

CMD="python \"$PROJECT_ROOT/scripts/generate_planner_training_data.py\" \
    \"$INPUT_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --num-datapoints \"$NUM_DATAPOINTS\" \
    --rl-data \
    --threshold \"$THRESHOLD\" \
    --reward-metric \"$METRIC\""

if [ -n "$EXCLUDE_FILE" ] && [ -f "$EXCLUDE_FILE" ]; then
    CMD="$CMD --exclude-file \"$EXCLUDE_FILE\""
    print_color $CYAN "📝 Excluding test samples from: $EXCLUDE_FILE"
    echo ""
fi

eval $CMD

EXIT_CODE=$?

# Check result
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    print_color $PURPLE "============================================================================"
    print_color $GREEN "✅ RL Training Data Generated Successfully!"
    print_color $PURPLE "============================================================================"
    echo ""
    print_color $CYAN "📁 Output: $OUTPUT_DIR/planner_training_data.json"
    echo ""
    print_color $YELLOW "Next steps:"
    print_color $YELLOW "  1. Review the generated data"
    print_color $YELLOW "  2. Train RL model: ./scripts/start_planner_training_rl.sh"
    print_color $YELLOW "  3. Evaluate: ./scripts/evaluate_planner.sh --checkpoint checkpoints/.../rl/final"
    echo ""
else
    echo ""
    print_color $RED "❌ Failed to generate RL training data (exit code: $EXIT_CODE)"
    print_color $YELLOW "💡 Check the error messages above"
    exit $EXIT_CODE
fi

