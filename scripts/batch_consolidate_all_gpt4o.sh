#!/bin/bash
# Batch consolidate ALL 12 GPT-4o datasets with 8-model tables
# Output to FINAL directories (consolidated_text / consolidated_vision)

# set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  BATCH CONSOLIDATION - All 12 GPT-4o Datasets (8-Model Tables)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting: $(date)"
echo ""

TOTAL=12
COMPLETED=0
FAILED=0

# Function to run text consolidation
run_text_consolidation() {
    local name="$1"
    local base_dir="$2"
    local script="$3"
    
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "[$((COMPLETED + FAILED + 1))/$TOTAL] Consolidating: $name"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # Build arguments
    local args=(
        "--baseline-dir" "$base_dir/baseline"
        "--edit-only-dir" "$base_dir/edit_only"
        "--standard-text-dir" "$base_dir/standard_text"
        "--rl-text-dir" "$base_dir/rl_text"
        "--rw-text-dir" "$base_dir/rw_text"
        "--dpo-text-dir" "$base_dir/dpo_text"
        "--sw-text-dir" "$base_dir/sw_text"
        "--gpt4o-dir" "$base_dir/gpt4o"
        "--output-dir" "$base_dir/consolidated_text"
    )
    
    if bash "$script" "${args[@]}"; then
        echo ""
        echo "✅ SUCCESS: $name"
        ((COMPLETED++))
    else
        echo ""
        echo "❌ FAILED: $name"
        ((FAILED++))
    fi
    echo ""
}

# Function for vision consolidation
run_vision_consolidation() {
    local name="$1"
    local base_dir="$2"
    local script="$3"
    
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "[$((COMPLETED + FAILED + 1))/$TOTAL] Consolidating: $name"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # Build arguments
    local args=(
        "--baseline-dir" "$base_dir/baseline"
        "--edit-only-dir" "$base_dir/edit_only"
        "--standard-vision-dir" "$base_dir/standard_vision"
        "--rl-vision-dir" "$base_dir/rl_vision"
        "--rw-vision-dir" "$base_dir/rw_vision"
        "--dpo-vision-dir" "$base_dir/dpo_vision"
        "--sw-vision-dir" "$base_dir/sw_vision"
        "--gpt4o-dir" "$base_dir/gpt4o"
        "--output-dir" "$base_dir/consolidated_vision"
    )
    
    if bash "$script" "${args[@]}"; then
        echo ""
        echo "✅ SUCCESS: $name"
        ((COMPLETED++))
    else
        echo ""
        echo "❌ FAILED: $name"
        ((FAILED++))
    fi
    echo ""
}

# ════════════════════════════════════════════════════════════════════════════════
# TEXT 8B DATASETS (3 total)
# ════════════════════════════════════════════════════════════════════════════════

run_text_consolidation \
    "Text Normal 8B" \
    "evaluation_results/text_parallel_cot_8b_trajectory" \
    "scripts/evaluation/consolidate_text_results.sh"

run_text_consolidation \
    "Text Complex 8B" \
    "evaluation_results/text_parallel_complex_cot_8b_trajectory" \
    "scripts/evaluation/complex_theme/consolidate_text_results.sh"

run_text_consolidation \
    "Text Complex_v2 8B" \
    "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory" \
    "scripts/evaluation/consolidate_text_results.sh"

# ════════════════════════════════════════════════════════════════════════════════
# TEXT 4B DATASETS (3 total)
# ════════════════════════════════════════════════════════════════════════════════

run_text_consolidation \
    "Text Normal 4B" \
    "evaluation_results/text_parallel_cot_4b_trajectory" \
    "scripts/evaluation/consolidate_text_results.sh"

run_text_consolidation \
    "Text Complex 4B" \
    "evaluation_results/text_parallel_complex_cot_4b_trajectory" \
    "scripts/evaluation/complex_theme/consolidate_text_results.sh"

run_text_consolidation \
    "Text Complex_v2 4B" \
    "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory" \
    "scripts/evaluation/consolidate_text_results.sh"

# ════════════════════════════════════════════════════════════════════════════════
# VISION 8B DATASETS (3 total)
# ════════════════════════════════════════════════════════════════════════════════

run_vision_consolidation \
    "Vision Normal 8B" \
    "evaluation_results/vision_parallel_cot_8b_trajectory" \
    "scripts/evaluation/consolidate_vision_results.sh"

run_vision_consolidation \
    "Vision Complex 8B" \
    "evaluation_results/vision_parallel_complex_cot_8b_trajectory" \
    "scripts/evaluation/complex_theme/consolidate_vision_results.sh"

run_vision_consolidation \
    "Vision Complex_v2 8B" \
    "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory" \
    "scripts/evaluation/consolidate_vision_results.sh"

# ════════════════════════════════════════════════════════════════════════════════
# VISION 4B DATASETS (3 total)
# ════════════════════════════════════════════════════════════════════════════════

run_vision_consolidation \
    "Vision Normal 4B" \
    "evaluation_results/vision_parallel_cot_4b_trajectory" \
    "scripts/evaluation/consolidate_vision_results.sh"

run_vision_consolidation \
    "Vision Complex 4B" \
    "evaluation_results/vision_parallel_complex_cot_4b_trajectory" \
    "scripts/evaluation/complex_theme/consolidate_vision_results.sh"

run_vision_consolidation \
    "Vision Complex_v2 4B" \
    "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory" \
    "scripts/evaluation/consolidate_vision_results.sh"

# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  BATCH CONSOLIDATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Finished: $(date)"
echo ""
echo "Total datasets:     $TOTAL"
echo "✅ Completed:       $COMPLETED"
echo "❌ Failed:          $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL 12 CONSOLIDATIONS SUCCESSFUL!"
    echo ""
    echo "All 8-model tables are now in:"
    echo "  • evaluation_results/*/consolidated_text/"
    echo "  • evaluation_results/*/consolidated_vision/"
else
    echo "⚠️  Some consolidations failed. Check output above for details."
fi

echo "════════════════════════════════════════════════════════════════════════════════"
