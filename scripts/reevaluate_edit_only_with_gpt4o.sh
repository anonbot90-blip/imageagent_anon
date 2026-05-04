#!/bin/bash
# Re-evaluate Edit-Only baseline with GPT-4o Image Judge (all 12 datasets)
# This replaces Qwen3-VL-8B reward model scores with real GPT-4o scores
# Run in screen: screen -S edit_only_gpt4o bash scripts/reevaluate_edit_only_with_gpt4o.sh

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Log file
LOG_DIR="$PROJECT_ROOT/logs/edit_only_gpt4o"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Edit-Only GPT-4o Re-evaluation (All 12 Datasets)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting at: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "This will evaluate 12 datasets with GPT-4o Image Judge:"
echo "  - Regular: vision/text × 4b/8b (4 datasets)"
echo "  - Complex: vision/text × 4b/8b (4 datasets)"
echo "  - Complex_v2: vision/text × 4b/8b (4 datasets)"
echo ""
echo "Expected time: 4-6 hours (~2,388 GPT-4o API calls)"
echo "Cost estimate: ~\$24"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Start timer
START_TIME=$(date +%s)

# Track statistics
TOTAL_DATASETS=12
COMPLETED_DATASETS=0
FAILED_DATASETS=0

# Function to evaluate one dataset
evaluate_dataset() {
    local dataset=$1
    local results_dir=$2
    local category=$3
    
    local num_samples=$(wc -l < "evaluation_results/$dataset/selected_sample_ids.txt" 2>/dev/null || echo "0")
    
    if [ "$num_samples" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Skipping $dataset: No sample IDs file${NC}"
        return 1
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo -e "${BLUE}[$((COMPLETED_DATASETS + 1))/$TOTAL_DATASETS] $category: $dataset${NC}"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Samples: $num_samples"
    echo "  Results dir: $results_dir"
    echo "  Output: evaluation_results/$dataset/edit_only"
    echo "  Started: $(date)"
    echo ""
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluation/evaluate_edit_only.py \
        --sample-ids-file "evaluation_results/$dataset/selected_sample_ids.txt" \
        --results-dir "$results_dir" \
        --output "evaluation_results/$dataset/edit_only" \
        2>&1 | tee -a "$LOG_FILE" | grep -E "(Evaluating E baseline|✓ Processed|✅ GPT-4o|⚠️|❌)"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $dataset completed successfully${NC}"
        echo "  Finished: $(date)"
        return 0
    else
        echo -e "${RED}❌ $dataset failed (exit code: $exit_code)${NC}"
        echo "  Check log: $LOG_FILE"
        return 1
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# PART 1: Regular Datasets (4 datasets)
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  PART 1/3: Regular Datasets                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Regular vision 4b
if evaluate_dataset "vision_parallel_cot_4b_trajectory" "imageagent_results_16000_cot" "Regular"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Regular vision 8b
if evaluate_dataset "vision_parallel_cot_8b_trajectory" "imageagent_results_16000_cot" "Regular"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Regular text 4b
if evaluate_dataset "text_parallel_cot_4b_trajectory" "imageagent_results_16000_cot" "Regular"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Regular text 8b
if evaluate_dataset "text_parallel_cot_8b_trajectory" "imageagent_results_16000_cot" "Regular"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# ════════════════════════════════════════════════════════════════════════════════
# PART 2: Complex Datasets (4 datasets)
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  PART 2/3: Complex Theme Datasets                                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Complex vision 4b
if evaluate_dataset "vision_parallel_complex_cot_4b_trajectory" "imageagent_results_normal_cot" "Complex"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Complex vision 8b
if evaluate_dataset "vision_parallel_complex_cot_8b_trajectory" "imageagent_results_normal_cot" "Complex"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Complex text 4b
if evaluate_dataset "text_parallel_complex_cot_4b_trajectory" "imageagent_results_normal_cot" "Complex"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Complex text 8b
if evaluate_dataset "text_parallel_complex_cot_8b_trajectory" "imageagent_results_normal_cot" "Complex"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# ════════════════════════════════════════════════════════════════════════════════
# PART 3: Complex V2 Datasets (4 datasets)
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  PART 3/3: Complex V2 Multi-Dimensional Datasets                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Complex_v2 vision 4b
if evaluate_dataset "vision_parallel_complex_v2_cot_4b_trajectory" "imageagent_results_complex_v2_10k_cot" "Complex_v2"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Complex_v2 vision 8b
if evaluate_dataset "vision_parallel_complex_v2_cot_8b_trajectory" "imageagent_results_complex_v2_10k_cot" "Complex_v2"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Complex_v2 text 4b
if evaluate_dataset "text_parallel_complex_v2_cot_4b_trajectory" "imageagent_results_complex_v2_10k_cot" "Complex_v2"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# Complex_v2 text 8b
if evaluate_dataset "text_parallel_complex_v2_cot_8b_trajectory" "imageagent_results_complex_v2_10k_cot" "Complex_v2"; then
    ((COMPLETED_DATASETS++))
else
    ((FAILED_DATASETS++))
fi

# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINS=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  EVALUATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Finished at: $(date)"
echo "Total time: ${HOURS}h ${MINS}m"
echo ""
echo "Results:"
echo "  ✅ Completed: $COMPLETED_DATASETS/$TOTAL_DATASETS datasets"
if [ $FAILED_DATASETS -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $FAILED_DATASETS datasets${NC}"
    echo "  Check log for details: $LOG_FILE"
else
    echo "  🎉 All datasets evaluated successfully!"
fi
echo ""
echo "Next steps:"
echo "  1. Re-run consolidation scripts to update consolidated JSONs"
echo "  2. Regenerate tables in evaluation_results/"
echo "  3. Update consolidated_results/ improvement tables"
echo ""
echo "Log file: $LOG_FILE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $FAILED_DATASETS -gt 0 ]; then
    exit 1
fi

exit 0

