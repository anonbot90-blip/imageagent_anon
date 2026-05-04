#!/bin/bash
# Re-consolidate ALL evaluation results with Edit-Only GPT-4o scores
# Run in screen: screen -S reconsolidate bash scripts/evaluation/reconsolidate_all_with_gpt4o.sh

# set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Log file
LOG_DIR="$PROJECT_ROOT/logs/consolidation"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/reconsolidate_$(date +%Y%m%d_%H%M%S).log"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Re-consolidating ALL datasets with Edit-Only GPT-4o scores"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting at: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)
COMPLETED=0
FAILED=0

# # ════════════════════════════════════════════════════════════════════════════════
# # REGULAR DATASETS (COMMENT OUT THIS SECTION IF ALREADY DONE)
# # ════════════════════════════════════════════════════════════════════════════════

# echo ""
# echo "════════════════════════════════════════════════════════════════════════════════"
# echo "  REGULAR DATASETS"
# echo "════════════════════════════════════════════════════════════════════════════════"
# echo ""

# # REGULAR vision_4b
# echo -e "${BLUE}[1/12] REGULAR vision_4b${NC}"
# bash scripts/evaluation/consolidate_vision_results.sh \
#   --baseline-dir "evaluation_results/vision_parallel_cot_4b_trajectory/baseline" \
#   --edit-only-dir "evaluation_results/vision_parallel_cot_4b_trajectory/edit_only" \
#   --standard-vision-dir "evaluation_results/vision_parallel_cot_4b_trajectory/standard_vision" \
#   --rl-vision-dir "evaluation_results/vision_parallel_cot_4b_trajectory/rl_vision" \
#   --rw-vision-dir "evaluation_results/vision_parallel_cot_4b_trajectory/rw_vision" \
#   --dpo-vision-dir "evaluation_results/vision_parallel_cot_4b_trajectory/dpo_vision" \
#   --sw-vision-dir "evaluation_results/vision_parallel_cot_4b_trajectory/sw_vision" \
#   --gpt4o-dir "evaluation_results/vision_parallel_cot_4b_trajectory/gpt4o" \
  --output-dir "evaluation_results/vision_parallel_cot_4b_trajectory/consolidated_vision" \
#   2>&1 | tee -a "$LOG_FILE"
# ((COMPLETED++))
# echo -e "${GREEN}✅ [1/12] Complete${NC}"
# echo ""

# # REGULAR vision_8b
# echo -e "${BLUE}[2/12] REGULAR vision_8b${NC}"
# bash scripts/evaluation/consolidate_vision_results.sh \
#   --baseline-dir "evaluation_results/vision_parallel_cot_8b_trajectory/baseline" \
#   --edit-only-dir "evaluation_results/vision_parallel_cot_8b_trajectory/edit_only" \
#   --standard-vision-dir "evaluation_results/vision_parallel_cot_8b_trajectory/standard_vision" \
#   --rl-vision-dir "evaluation_results/vision_parallel_cot_8b_trajectory/rl_vision" \
#   --rw-vision-dir "evaluation_results/vision_parallel_cot_8b_trajectory/rw_vision" \
#   --dpo-vision-dir "evaluation_results/vision_parallel_cot_8b_trajectory/dpo_vision" \
#   --sw-vision-dir "evaluation_results/vision_parallel_cot_8b_trajectory/sw_vision" \
#   --gpt4o-dir "evaluation_results/vision_parallel_cot_8b_trajectory/gpt4o" \
  --output-dir "evaluation_results/vision_parallel_cot_8b_trajectory/consolidated_vision" \
#   2>&1 | tee -a "$LOG_FILE"
# ((COMPLETED++))
# echo -e "${GREEN}✅ [2/12] Complete${NC}"
# echo ""

# # REGULAR text_4b
# echo -e "${BLUE}[3/12] REGULAR text_4b${NC}"
# bash scripts/evaluation/consolidate_text_results.sh \
#   --baseline-dir "evaluation_results/text_parallel_cot_4b_trajectory/baseline" \
#   --edit-only-dir "evaluation_results/text_parallel_cot_4b_trajectory/edit_only" \
#   --standard-text-dir "evaluation_results/text_parallel_cot_4b_trajectory/standard_text" \
#   --rl-text-dir "evaluation_results/text_parallel_cot_4b_trajectory/rl_text" \
#   --rw-text-dir "evaluation_results/text_parallel_cot_4b_trajectory/rw_text" \
#   --dpo-text-dir "evaluation_results/text_parallel_cot_4b_trajectory/dpo_text" \
#   --sw-text-dir "evaluation_results/text_parallel_cot_4b_trajectory/sw_text" \
#   --gpt4o-dir "evaluation_results/text_parallel_cot_4b_trajectory/gpt4o" \
  --output-dir "evaluation_results/text_parallel_cot_4b_trajectory/consolidated_text" \
#   2>&1 | tee -a "$LOG_FILE"
# ((COMPLETED++))
# echo -e "${GREEN}✅ [3/12] Complete${NC}"
# echo ""

# # REGULAR text_8b
# echo -e "${BLUE}[4/12] REGULAR text_8b${NC}"
# bash scripts/evaluation/consolidate_text_results.sh \
#   --baseline-dir "evaluation_results/text_parallel_cot_8b_trajectory/baseline" \
#   --edit-only-dir "evaluation_results/text_parallel_cot_8b_trajectory/edit_only" \
#   --standard-text-dir "evaluation_results/text_parallel_cot_8b_trajectory/standard_text" \
#   --rl-text-dir "evaluation_results/text_parallel_cot_8b_trajectory/rl_text" \
#   --rw-text-dir "evaluation_results/text_parallel_cot_8b_trajectory/rw_text" \
#   --dpo-text-dir "evaluation_results/text_parallel_cot_8b_trajectory/dpo_text" \
#   --sw-text-dir "evaluation_results/text_parallel_cot_8b_trajectory/sw_text" \
#   --gpt4o-dir "evaluation_results/text_parallel_cot_8b_trajectory/gpt4o" \
  --output-dir "evaluation_results/text_parallel_cot_8b_trajectory/consolidated_text" \
#   2>&1 | tee -a "$LOG_FILE"
# ((COMPLETED++))
# echo -e "${GREEN}✅ [4/12] Complete${NC}"
# echo ""

# ════════════════════════════════════════════════════════════════════════════════
# COMPLEX DATASETS
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  COMPLEX DATASETS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# COMPLEX vision_4b
echo -e "${BLUE}[5/12] COMPLEX vision_4b${NC}"
bash scripts/evaluation/consolidate_vision_results.sh \
  --baseline-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/edit_only" \
  --standard-vision-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/standard_vision" \
  --rl-vision-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/rl_vision" \
  --rw-vision-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/rw_vision" \
  --dpo-vision-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/dpo_vision" \
  --sw-vision-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/sw_vision" \
  --gpt4o-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/gpt4o" \
  --output-dir "evaluation_results/vision_parallel_complex_cot_4b_trajectory/consolidated_vision" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [5/12] Complete${NC}"
echo ""

# COMPLEX vision_8b
echo -e "${BLUE}[6/12] COMPLEX vision_8b${NC}"
bash scripts/evaluation/consolidate_vision_results.sh \
  --baseline-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/edit_only" \
  --standard-vision-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/standard_vision" \
  --rl-vision-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/rl_vision" \
  --rw-vision-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/rw_vision" \
  --dpo-vision-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/dpo_vision" \
  --sw-vision-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/sw_vision" \
  --gpt4o-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/gpt4o" \
  --output-dir "evaluation_results/vision_parallel_complex_cot_8b_trajectory/consolidated_vision" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [6/12] Complete${NC}"
echo ""

# COMPLEX text_4b
echo -e "${BLUE}[7/12] COMPLEX text_4b${NC}"
bash scripts/evaluation/consolidate_text_results.sh \
  --baseline-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/edit_only" \
  --standard-text-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/standard_text" \
  --rl-text-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/rl_text" \
  --rw-text-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/rw_text" \
  --dpo-text-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/dpo_text" \
  --sw-text-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/sw_text" \
  --gpt4o-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/gpt4o" \
  --output-dir "evaluation_results/text_parallel_complex_cot_4b_trajectory/consolidated_text" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [7/12] Complete${NC}"
echo ""

# COMPLEX text_8b
echo -e "${BLUE}[8/12] COMPLEX text_8b${NC}"
bash scripts/evaluation/consolidate_text_results.sh \
  --baseline-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/edit_only" \
  --standard-text-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/standard_text" \
  --rl-text-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/rl_text" \
  --rw-text-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/rw_text" \
  --dpo-text-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/dpo_text" \
  --sw-text-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/sw_text" \
  --gpt4o-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/gpt4o" \
  --output-dir "evaluation_results/text_parallel_complex_cot_8b_trajectory/consolidated_text" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [8/12] Complete${NC}"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# COMPLEX_V2 DATASETS
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  COMPLEX_V2 DATASETS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# COMPLEX_V2 vision_4b
echo -e "${BLUE}[9/12] COMPLEX_V2 vision_4b${NC}"
bash scripts/evaluation/consolidate_vision_results.sh \
  --baseline-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/edit_only" \
  --standard-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/standard_vision" \
  --rl-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/rl_vision" \
  --rw-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/rw_vision" \
  --dpo-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/dpo_vision" \
  --sw-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/sw_vision" \
  --gpt4o-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/gpt4o" \
  --output-dir "evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory/consolidated_vision" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [9/12] Complete${NC}"
echo ""

# COMPLEX_V2 vision_8b
echo -e "${BLUE}[10/12] COMPLEX_V2 vision_8b${NC}"
bash scripts/evaluation/consolidate_vision_results.sh \
  --baseline-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/edit_only" \
  --standard-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/standard_vision" \
  --rl-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/rl_vision" \
  --rw-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/rw_vision" \
  --dpo-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/dpo_vision" \
  --sw-vision-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/sw_vision" \
  --gpt4o-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/gpt4o" \
  --output-dir "evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory/consolidated_vision" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [10/12] Complete${NC}"
echo ""

# COMPLEX_V2 text_4b
echo -e "${BLUE}[11/12] COMPLEX_V2 text_4b${NC}"
bash scripts/evaluation/consolidate_text_results.sh \
  --baseline-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/edit_only" \
  --standard-text-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/standard_text" \
  --rl-text-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/rl_text" \
  --rw-text-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/rw_text" \
  --dpo-text-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/dpo_text" \
  --sw-text-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/sw_text" \
  --gpt4o-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/gpt4o" \
  --output-dir "evaluation_results/text_parallel_complex_v2_cot_4b_trajectory/consolidated_text" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [11/12] Complete${NC}"
echo ""

# COMPLEX_V2 text_8b
echo -e "${BLUE}[12/12] COMPLEX_V2 text_8b${NC}"
bash scripts/evaluation/consolidate_text_results.sh \
  --baseline-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/baseline" \
  --edit-only-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/edit_only" \
  --standard-text-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/standard_text" \
  --rl-text-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/rl_text" \
  --rw-text-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/rw_text" \
  --dpo-text-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/dpo_text" \
  --sw-text-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/sw_text" \
  --gpt4o-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/gpt4o" \
  --output-dir "evaluation_results/text_parallel_complex_v2_cot_8b_trajectory/consolidated_text" \
  2>&1 | tee -a "$LOG_FILE"
((COMPLETED++))
echo -e "${GREEN}✅ [12/12] Complete${NC}"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINS=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ALL CONSOLIDATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Time: ${HOURS}h ${MINS}m"
echo "  ✅ Completed: $COMPLETED/12"
[ $FAILED -gt 0 ] && echo -e "  ${RED}❌ Failed: $FAILED${NC}"
echo "  Log: $LOG_FILE"
echo "════════════════════════════════════════════════════════════════════════════════"

[ $FAILED -gt 0 ] && exit 1
exit 0

