#!/bin/bash
# Run all 4 Gemini 2.5 planner evaluations sequentially in a single screen session.
# Runs text4b → text8b → vision4b → vision8b one at a time to avoid GPU OOM.
#
# Usage: bash scripts/evaluation/simple/master_eval_gemini.sh [num_samples]
#
# Screen: gemini_eval_sequential_simple
# Monitor: screen -r gemini_eval_sequential_simple
# Log:     tail -f logs/gemini_eval/sequential_simple.log

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
NUM_SAMPLES="${1:-199}"
LOG_DIR="$PROJECT_ROOT/logs/gemini_eval"
mkdir -p "$LOG_DIR"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CONDA_INIT="source $CONDA_BASE/etc/profile.d/conda.sh && conda activate img-agent"
CREDS=""  # Set GOOGLE_API_KEY / GEMINI_API_KEY in your environment
LOG_FILE="$LOG_DIR/sequential_simple.log"
SCREEN_NAME="gemini_eval_sequential_simple"

screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true

CMD="$CONDA_INIT && $CREDS && cd $PROJECT_ROOT && (
    echo '=== [1/4] text4b ===' &&
    bash scripts/evaluation/simple/text_4b_eval_gemini.sh --num-samples $NUM_SAMPLES &&
    echo '=== [2/4] text8b ===' &&
    bash scripts/evaluation/simple/text_8b_eval_gemini.sh --num-samples $NUM_SAMPLES &&
    echo '=== [3/4] vision4b ===' &&
    bash scripts/evaluation/simple/vision_4b_eval_gemini.sh --num-samples $NUM_SAMPLES &&
    echo '=== [4/4] vision8b ===' &&
    bash scripts/evaluation/simple/vision_8b_eval_gemini.sh --num-samples $NUM_SAMPLES &&
    echo '=== ALL 4 COMPLETE ==='
) 2>&1 | tee $LOG_FILE"

screen -dmS "$SCREEN_NAME" bash -c "$CMD; exec bash"
echo "✅ Started screen: $SCREEN_NAME"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  screen -r $SCREEN_NAME"
echo "  tail -f $LOG_FILE"
echo ""
echo "After completion, run judge:"
echo "  bash scripts/evaluation/simple/launch_gemini25_judge_screens.sh"
