#!/bin/bash
# Re-evaluate Edit-Only baseline with GPT-4o - NORMAL DATASET ONLY
# Run in screen: screen -S edit_regular bash scripts/reevaluate_edit_only_with_gpt4o_regular.sh

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
LOG_FILE="$LOG_DIR/regular_$(date +%Y%m%d_%H%M%S).log"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Edit-Only GPT-4o Re-evaluation - NORMAL DATASET"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting at: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "Datasets: 4 (vision/text × 4b/8b)"
echo "GPU: CUDA_VISIBLE_DEVICES=0"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)
COMPLETED=0
FAILED=0

evaluate_dataset() {
    local dataset=$1
    local results_dir=$2
    
    local num_samples=$(wc -l < "evaluation_results/$dataset/selected_sample_ids.txt" 2>/dev/null || echo "0")
    
    if [ "$num_samples" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Skipping $dataset: No sample IDs file${NC}"
        return 1
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo -e "${BLUE}[$((COMPLETED + 1))/4] $dataset${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Samples: $num_samples | Started: $(date)"
    
    CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/evaluate_edit_only.py \
        --sample-ids-file "evaluation_results/$dataset/selected_sample_ids.txt" \
        --results-dir "$results_dir" \
        --output "evaluation_results/$dataset/edit_only" \
        2>&1 | tee -a "$LOG_FILE" | grep -E "(Evaluating E baseline|✓ Processed|✅ GPT-4o|⚠️|❌)" || true
    
    # if [ ${PIPESTATUS[0]} -eq 0 ]; then
    #     echo -e "${GREEN}✅ Completed: $(date)${NC}"
    #     return 0
    # else
    #     echo -e "${RED}❌ Failed${NC}"
    #     return 1
    # fi
}

# # Vision 4B
# if evaluate_dataset "vision_parallel_cot_4b_trajectory" "imageagent_results_normal_cot_test"; then
#     ((COMPLETED++))
# else
#     ((FAILED++))
# fi

# evaluate_dataset "vision_parallel_cot_4b_trajectory" "imageagent_results_normal_cot_test"

# # Vision 8B
# if evaluate_dataset "vision_parallel_cot_8b_trajectory" "imageagent_results_normal_cot_test"; then
#     ((COMPLETED++))
# else
#     ((FAILED++))
# fi

# evaluate_dataset "vision_parallel_cot_8b_trajectory" "imageagent_results_normal_cot_test"

# # Text 4B
# if evaluate_dataset "text_parallel_cot_4b_trajectory" "imageagent_results_normal_cot_test"; then
#     ((COMPLETED++))
# else
#     ((FAILED++))
# fi

# evaluate_dataset "text_parallel_cot_4b_trajectory" "imageagent_results_normal_cot_test"

# # Text 8B
# if evaluate_dataset "text_parallel_cot_8b_trajectory" "imageagent_results_normal_cot_test"; then
#     ((COMPLETED++))
# else
#     ((FAILED++))
# fi

evaluate_dataset "text_parallel_cot_8b_trajectory" "imageagent_results_normal_cot_test"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINS=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  NORMAL DATASET COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Time: ${HOURS}h ${MINS}m"
echo "  ✅ Completed: $COMPLETED/4"
# [ $FAILED -gt 0 ] && echo -e "  ${RED}❌ Failed: $FAILED${NC}"
# echo "  Log: $LOG_FILE"
# echo "════════════════════════════════════════════════════════════════════════════════"

# [ $FAILED -gt 0 ] && exit 1
# exit 0

