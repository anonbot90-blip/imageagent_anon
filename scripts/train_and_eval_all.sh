#!/bin/bash

# Simple script to train and evaluate all models
# 1. Train text models (parallel)
# 2. Train vision models (parallel)
# 3. Evaluate text models (parallel)
# 4. Evaluate vision models (parallel)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 TRAIN AND EVALUATE ALL MODELS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# STEP 1: Train Text Models (Parallel)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 STEP 1/4: Training Text Models (Parallel)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$SCRIPT_DIR/training/start_planner_training_text_only.sh"

echo ""
echo "✅ Text models training complete!"
echo ""

# ============================================================================
# STEP 2: Train Vision Models (Parallel)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "👁️  STEP 2/4: Training Vision Models (Parallel)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$SCRIPT_DIR/training/start_planner_training_vision.sh"

echo ""
echo "✅ Vision models training complete!"
echo ""

# # ============================================================================
# # STEP 3: Evaluate Text Models (Parallel)
# # ============================================================================

# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "📊 STEP 3/4: Evaluating Text Models (Parallel)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# bash "$SCRIPT_DIR/evaluation/start_eval_all_planner_text.sh"

# echo ""
# echo "✅ Text models evaluation complete!"
# echo ""

# # ============================================================================
# # STEP 4: Evaluate Vision Models (Parallel)
# # ============================================================================

# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "👁️  STEP 4/4: Evaluating Vision Models (Parallel)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# bash "$SCRIPT_DIR/evaluation/start_eval_all_planner_vision.sh"

# echo ""
# echo "✅ Vision models evaluation complete!"
# echo ""

# # ============================================================================
# # DONE
# # ============================================================================

# echo "════════════════════════════════════════════════════════════════════════════════"
# echo "🎉 ALL TRAINING AND EVALUATION COMPLETE!"
# echo "════════════════════════════════════════════════════════════════════════════════"
# echo ""
# echo "📁 Results:"
# echo "   Text:   evaluation_results/text_parallel_eval/consolidated_text/"
# echo "   Vision: evaluation_results/vision_parallel_eval/consolidated_vision/"
# echo ""

