#!/bin/bash
# Phase C: Run all judges on normal 8B reruns.
# Waits for text_8b and vision_8b inference screens to exit, then launches judges.
# Invokes existing launchers with --skip-existing so 4B configs are no-ops.

set -u

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

echo "[phase-c] Waiting for text8b_eval_normal and vision8b_eval_queued screens..."
while screen -ls | grep -qE "text8b_eval_normal|vision8b_eval_queued"; do
    sleep 300
done
echo "[phase-c] Inference finished at $(date). Starting judges."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate img-agent
# Set required env vars: GOOGLE_API_KEY, GEMINI_API_KEY, AZURE_OPENAI_API_KEY, etc.

# (1) GPT-5.4 + Claude Opus + Gemini 2.5 Flash judges (all 4 normal configs; 4B skip-existing)
echo "[phase-c] Launching launch_judge_screens.sh normal"
bash "$PROJECT_ROOT/scripts/launch_judge_screens.sh" normal

# (2) Gemini 2.5 Pro judge — only 8B configs (paper-critical)
for CFG in text_parallel_cot_8b_trajectory vision_parallel_cot_8b_trajectory; do
    RESULTS_DIR="$PROJECT_ROOT/evaluation_results/normal/$CFG"
    SHORT=$(echo "$CFG" | sed -e 's|text_parallel_cot_8b_trajectory|text8b|' -e 's|vision_parallel_cot_8b_trajectory|vision8b|')
    SCREEN_NAME="judge_gem25pro_${SHORT}_normal"
    LOG_FILE="$PROJECT_ROOT/logs/judge_runs/${SCREEN_NAME}.log"
    mkdir -p "$PROJECT_ROOT/logs/judge_runs"
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
    CMD="source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate img-agent && \
         export LLM_PROXY_GEMINI_MODEL=gemini-2.5-pro && \
         cd $PROJECT_ROOT && \
         python scripts/run_judge_on_results.py \
             --judge proxy_gemini \
             --output-as gemini_25pro \
             --results-dir $RESULTS_DIR \
             --skip-existing \
             --rate-limit-delay 0.3 \
             2>&1 | tee $LOG_FILE"
    screen -dmS "$SCREEN_NAME" bash -c "$CMD; echo 'DONE gem25pro $SHORT'; exec bash"
    echo "[phase-c] Launched $SCREEN_NAME"
done

# (3) Gemini 3.1 Flash Lite judge (all methods, all configs)
echo "[phase-c] Launching launch_judge_gemini31fl.sh normal"
bash "$PROJECT_ROOT/scripts/launch_judge_gemini31fl.sh" normal

echo "[phase-c] All judge launchers fired at $(date)."
echo "[phase-c] When all judge screens finish:"
echo "  - bash $PROJECT_ROOT/scripts/combine_gemini31fl.sh"
echo "  - python $PROJECT_ROOT/scripts/evaluation/generate_multi_judge_latex.py --results-root evaluation_results --out-dir latex/neurips2026/generated_tables"
