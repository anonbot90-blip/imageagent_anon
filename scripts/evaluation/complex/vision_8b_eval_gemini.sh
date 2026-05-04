#!/bin/bash
# Evaluate Gemini 2.5 planner on 8B vision trajectory samples (complex dataset)

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DATA_PATH="$PROJECT_ROOT/training_data/complex/cot_8b_trajectory/full_dataset_for_eval.json"
TEST_SAMPLES_FILE="$PROJECT_ROOT/training_data/complex/cot_8b_trajectory/test_samples_cot_8b.txt"
OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/complex/vision_parallel_cot_8b_trajectory"
GEMINI_DIR="$OUTPUT_BASE/gemini25"
ACTION_LIBRARY="$PROJECT_ROOT/actions/action_library_complex.json"
HIDREAM_CHECKPOINT="$PROJECT_ROOT/HiDream-E1"
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"
EDITOR_TYPE="qwen"
NUM_SAMPLES=199
SAVE_IMAGES="true"
SAVE_PREDICTIONS="true"
RATE_LIMIT_DELAY=0.5

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --output) GEMINI_DIR="$2"; shift 2 ;;
        --no-images) SAVE_IMAGES="false"; shift ;;
        --rate-limit-delay) RATE_LIMIT_DELAY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TEMP_SAMPLE_IDS="/tmp/gemini_eval_complex_vision8b_$$.txt"
head -$NUM_SAMPLES "$TEST_SAMPLES_FILE" > "$TEMP_SAMPLE_IDS"

PYTHON_CMD="python scripts/evaluate_gemini_planner.py \
    --data \"$DATA_PATH\" \
    --output \"$GEMINI_DIR\" \
    --sample-ids-file \"$TEMP_SAMPLE_IDS\" \
    --action-library \"$ACTION_LIBRARY\" \
    --model-editor \"$EDITOR_TYPE\" \
    --hidream-checkpoint \"$HIDREAM_CHECKPOINT\" \
    --hidream-config \"$HIDREAM_CONFIG\" \
    --rate-limit-delay $RATE_LIMIT_DELAY \
    --save-predictions"

if [ "$SAVE_IMAGES" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --save-images"
fi

cd "$PROJECT_ROOT"
eval $PYTHON_CMD
EXIT_CODE=$?
rm -f "$TEMP_SAMPLE_IDS"
exit $EXIT_CODE
