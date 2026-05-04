#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# Master Script: Batch GPT-4o Image Judge Evaluation
# ════════════════════════════════════════════════════════════════════════════════
# 
# PURPOSE:
#   Run GPT-4o image judge evaluation on multiple datasets
#   - Evaluates GPT-4o's own outputs for image quality
#   - Aggregates scores into evaluation_summary_all.json
#   - Automatically skips datasets that already have scores
#
# USAGE:
#   bash scripts/batch_evaluate_gpt4o_image_judge.sh
#
# CUSTOMIZATION:
#   - Comment out datasets you don't want to evaluate
#   - Uncomment datasets you want to re-evaluate
#
# ════════════════════════════════════════════════════════════════════════════════

# set -e  # Exit on error for individual dataset, but continue loop

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# ════════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

DATASETS=(
    # ═══════════════════════════════════════════════════════════════════════════
    # ✅ ALREADY DONE - Uncomment to re-run
    # ═══════════════════════════════════════════════════════════════════════════
    # "text_parallel_cot_8b_trajectory"                    # Normal 8B (DONE)
    # "text_parallel_complex_cot_8b_trajectory"            # Complex 8B (DONE)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ❌ TEXT MODELS - NEEDS EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    # "text_parallel_complex_v2_cot_8b_trajectory"         # Complex_v2 8B ← PRIORITY
    "text_parallel_complex_v2_cot_4b_trajectory"         # Complex_v2 4B
    "text_parallel_complex_cot_4b_trajectory"            # Complex 4B
    "text_parallel_cot_4b_trajectory"                    # Normal 4B
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ❌ VISION MODELS - NEEDS EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    "vision_parallel_cot_8b_trajectory"                  # Vision Normal 8B
    # "vision_parallel_complex_cot_8b_trajectory"          # Vision Complex 8B
    # "vision_parallel_complex_v2_cot_8b_trajectory"       # Vision Complex_v2 8B
    # "vision_parallel_complex_v2_cot_4b_trajectory"       # Vision Complex_v2 4B
    # "vision_parallel_complex_cot_4b_trajectory"          # Vision Complex 4B
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⚠️  RECENTLY COMPLETED - Now included
    # ═══════════════════════════════════════════════════════════════════════════
    # "vision_parallel_cot_4b_trajectory"                  # Vision Normal 4B (96% done, 192/199)
)

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

check_has_image_scores() {
    local gpt4o_dir="$1"
    
    if [ ! -f "$gpt4o_dir/evaluation_summary_all.json" ]; then
        echo "MISSING_FILE"
        return
    fi
    
    python3 << PYEOF
import json
import sys

try:
    with open("$gpt4o_dir/evaluation_summary_all.json") as f:
        data = json.load(f)
    
    if "gpt_image_scores" in data:
        print("HAS_SCORES")
    else:
        print("NO_SCORES")
except Exception as e:
    print("ERROR")
    sys.stderr.write(f"Error checking scores: {e}\n")
PYEOF
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  🚀 Batch GPT-4o Image Judge Evaluation"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Total datasets to process: ${#DATASETS[@]}"
echo ""

PROCESSED=0
SKIPPED=0
ERRORS=0
SUCCESS=0

for dataset in "${DATASETS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📁 Dataset: $dataset"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    GPT4O_DIR="evaluation_results/$dataset/gpt4o"
    
    # Check if directory exists
    if [ ! -d "$GPT4O_DIR" ]; then
        echo "⚠️  GPT-4o directory not found: $GPT4O_DIR"
        echo "   Skipping..."
        ((ERRORS++))
        echo ""
        continue
    fi
    
    # Check if already has scores
    STATUS=$(check_has_image_scores "$GPT4O_DIR")
    
    if [ "$STATUS" == "HAS_SCORES" ]; then
        echo "✅ Already has gpt_image_scores - Skipping"
        ((SKIPPED++))
        echo ""
        continue
    elif [ "$STATUS" == "MISSING_FILE" ]; then
        echo "⚠️  evaluation_summary_all.json not found - Skipping"
        ((ERRORS++))
        echo ""
        continue
    fi
    
    # Run evaluation
    echo ""
    echo "🔍 Step 1/2: Running GPT-4o image judge evaluation..."
    if python scripts/add_gpt4o_image_judge.py "$GPT4O_DIR"; then
        echo "   ✅ Evaluation complete"
    else
        echo "   ❌ Evaluation failed"
        ((ERRORS++))
        echo ""
        continue
    fi
    
    echo ""
    echo "📊 Step 2/2: Aggregating scores to summary..."
    if python scripts/aggregate_gpt4o_image_scores.py "$GPT4O_DIR"; then
        echo "   ✅ Aggregation complete"
        ((SUCCESS++))
    else
        echo "   ❌ Aggregation failed"
        ((ERRORS++))
        echo ""
        continue
    fi
    
    echo ""
    echo "✅ Dataset complete: $dataset"
    echo ""
    
    ((PROCESSED++))
done

# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  📊 BATCH EVALUATION SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Total datasets:        ${#DATASETS[@]}"
echo "  ✅ Successfully evaluated: $SUCCESS"
echo "  ⏭️  Skipped (has scores):  $SKIPPED"
echo "  ❌ Errors/Missing:         $ERRORS"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

if [ $SUCCESS -gt 0 ]; then
    echo ""
    echo "🎉 Successfully evaluated $SUCCESS dataset(s)!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Run consolidation scripts to generate tables"
    echo "   2. Check generated tables for GPT-4o column"
    echo ""
fi

exit 0

