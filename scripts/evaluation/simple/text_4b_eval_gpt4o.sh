#!/bin/bash
# Evaluate GPT-4o planner on 4B text-only trajectory samples
# Simple wrapper around evaluate_gpt4o_planner.py

set -e  # Exit on error

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DATA_PATH="$PROJECT_ROOT/training_data/cot_4b_trajectory/full_dataset_for_eval.json"
TEST_SAMPLES_FILE="$PROJECT_ROOT/training_data/simple/cot_4b_trajectory/test_samples_cot_4b.txt"
OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/text_parallel_cot_4b_trajectory"
GPT4O_DIR="$OUTPUT_BASE/gpt4o"

# Action library (NORMAL theme uses 10-action library)
ACTION_LIBRARY="$PROJECT_ROOT/actions/action_library_v2.json"

# Image editor configuration (same as other models)
EDITOR_TYPE="qwen"
HIDREAM_CHECKPOINT="$PROJECT_ROOT/HiDream-E1"
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"

# Default settings
NUM_SAMPLES=2
SAVE_IMAGES="true"
SAVE_PREDICTIONS="true"
NO_GPT_JUDGE="false"
RATE_LIMIT_DELAY=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            GPT4O_DIR="$2"
            shift 2
            ;;
        --sample-ids-file)
            TEST_SAMPLES_FILE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --no-images)
            SAVE_IMAGES="false"
            shift
            ;;
        --no-predictions)
            SAVE_PREDICTIONS="false"
            shift
            ;;
        --no-gpt-judge)
            NO_GPT_JUDGE="true"
            shift
            ;;
        --rate-limit-delay)
            RATE_LIMIT_DELAY="$2"
            shift 2
            ;;
        --model-editor)
            EDITOR_TYPE="$2"
            shift 2
            ;;
        --hidream-checkpoint)
            HIDREAM_CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: bash scripts/evaluation/start_eval_gpt4o_planner_text_4b.sh [OPTIONS]

Evaluate GPT-4o Vision planner on 4B text-only trajectory samples.

Options:
  --data PATH                   Path to evaluation data (default: training_data/cot_8b_trajectory/full_dataset_for_eval.json)
  --output DIR                  Output directory (default: evaluation_results/text_parallel_cot_8b_trajectory/gpt4o)
  --sample-ids-file FILE        File with sample IDs (default: training_data/simple/test_samples_cot_8b.txt)
  --num-samples N               Number of samples to evaluate (default: 2)
  --no-images                   Skip image generation
  --no-predictions              Skip saving predictions
  --rate-limit-delay SECONDS    Delay between API calls (default: 0.5)
  --model-editor {qwen,hidream} Image editor to use (default: qwen)
  --hidream-checkpoint PATH     HiDream checkpoint path (if using hidream)
  -h, --help                    Show this help message

Examples:
  # Test 1: Quick validation (2 samples)
  bash scripts/evaluation/start_eval_gpt4o_planner_text.sh

  # Test 2: Planner-only (10 samples, no images)
  bash scripts/evaluation/start_eval_gpt4o_planner_text.sh --num-samples 10 --no-images

  # Test 3: Full evaluation (20 samples with images)
  bash scripts/evaluation/start_eval_gpt4o_planner_text.sh --num-samples 20

  # Test 4: Complex theme
  bash scripts/evaluation/start_eval_gpt4o_planner_text.sh \\
    --data training_data/complex_cot_8b_trajectory/full_dataset_for_eval.json \\
    --sample-ids-file training_data/complex_cot_8b_trajectory/test_samples_complex_cot_8b.txt \\
    --output evaluation_results/text_parallel_complex_cot_8b_trajectory/gpt4o \\
    --num-samples 5
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  GPT-4O PLANNER EVALUATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Data:           $DATA_PATH"
echo "  Samples File:   $TEST_SAMPLES_FILE"
echo "  Output:         $GPT4O_DIR"
echo "  Num Samples:    $NUM_SAMPLES"
echo "  Save Images:    $SAVE_IMAGES"
echo "  Save Preds:     $SAVE_PREDICTIONS"
echo "  GPT Judge:      $([ "$NO_GPT_JUDGE" = "false" ] && echo "enabled" || echo "disabled")"
echo "  Image Editor:   $EDITOR_TYPE"
echo "  Rate Limit:     ${RATE_LIMIT_DELAY}s"
echo ""
echo "Estimated time: ~$(echo "$NUM_SAMPLES * 3 / 60" | bc -l | xargs printf "%.1f") minutes"
echo "Estimated cost: ~\$$(echo "$NUM_SAMPLES * 0.015" | bc -l | xargs printf "%.2f")"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# CREATE SAMPLE IDS FILE
# ════════════════════════════════════════════════════════════════════════════════

# Create temporary file with first N sample IDs
TEMP_SAMPLE_IDS="/tmp/gpt4o_eval_samples_$$.txt"
head -$NUM_SAMPLES "$TEST_SAMPLES_FILE" > "$TEMP_SAMPLE_IDS"

echo "Sample IDs (first $NUM_SAMPLES):"
cat "$TEMP_SAMPLE_IDS" | nl
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# RUN EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Starting GPT-4o evaluation..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Build Python command
PYTHON_CMD="python scripts/evaluate_gpt4o_planner.py \
    --data \"$DATA_PATH\" \
    --output \"$GPT4O_DIR\" \
    --sample-ids-file \"$TEMP_SAMPLE_IDS\" \
    --action-library \"$ACTION_LIBRARY\" \
    --model-editor \"$EDITOR_TYPE\" \
    --hidream-checkpoint \"$HIDREAM_CHECKPOINT\" \
    --hidream-config \"$HIDREAM_CONFIG\" \
    --rate-limit-delay $RATE_LIMIT_DELAY"

# Add optional flags
if [ "$SAVE_IMAGES" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --save-images"
fi

if [ "$SAVE_PREDICTIONS" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --save-predictions"
fi


# Run evaluation
cd "$PROJECT_ROOT"
eval $PYTHON_CMD

EXIT_CODE=$?

# Cleanup temp file
rm -f "$TEMP_SAMPLE_IDS"

# ════════════════════════════════════════════════════════════════════════════════
# REPORT RESULTS
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ GPT-4O EVALUATION COMPLETE!"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "📂 Results saved to: $GPT4O_DIR"
    echo ""
    echo "📊 View results:"
    echo "   Summary:  cat $GPT4O_DIR/evaluation_summary_all.json"
    echo "   Detailed: cat $GPT4O_DIR/detailed_results_all.json"
    echo "   Samples:  ls $GPT4O_DIR/samples/"
    echo ""
else
    echo "  ❌ GPT-4O EVALUATION FAILED (exit code: $EXIT_CODE)"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Check error messages above for details."
    echo ""
    exit $EXIT_CODE
fi

