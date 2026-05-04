#!/bin/bash
# Master script to run all 4 GPT-4o evaluation variants for COMPLEX_V2 theme
# Runs: 8B text, 4B text, 8B vision, 4B vision

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
NUM_SAMPLES=${1:-2}  # Default: 2 samples

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  GPT-4O MASTER EVALUATION - COMPLEX V2 THEME"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Running 4 variants with $NUM_SAMPLES samples each:"
echo "  1. 8B Text"
echo "  2. 4B Text"
echo "  3. 8B Vision"
echo "  4. 4B Vision"
echo ""

# Run all 4 variants
echo "───────────────────────────────────────────────────────────────────────────"
echo "1/4: Running 8B Text evaluation..."
echo "───────────────────────────────────────────────────────────────────────────"
bash "$PROJECT_ROOT/scripts/evaluation/start_eval_gpt4o_planner_text_complex_v2_8b.sh" --num-samples "$NUM_SAMPLES"

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "2/4: Running 4B Text evaluation..."
echo "───────────────────────────────────────────────────────────────────────────"
bash "$PROJECT_ROOT/scripts/evaluation/start_eval_gpt4o_planner_text_complex_v2_4b.sh" --num-samples "$NUM_SAMPLES"

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "3/4: Running 8B Vision evaluation..."
echo "───────────────────────────────────────────────────────────────────────────"
bash "$PROJECT_ROOT/scripts/evaluation/start_eval_gpt4o_planner_vision_complex_v2_8b.sh" --num-samples "$NUM_SAMPLES"

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "4/4: Running 4B Vision evaluation..."
echo "───────────────────────────────────────────────────────────────────────────"
bash "$PROJECT_ROOT/scripts/evaluation/start_eval_gpt4o_planner_vision_complex_v2_4b.sh" --num-samples "$NUM_SAMPLES"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  ✅ ALL 4 GPT-4O EVALUATIONS COMPLETE (COMPLEX V2 THEME)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  - evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/gpt4o/"
echo "  - evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/gpt4o/"
echo "  - evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/gpt4o/"
echo "  - evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/gpt4o/"
echo ""

