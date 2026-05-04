#!/bin/bash
# Master Evaluation Script - Runs all 12 evaluation scenarios sequentially
# Each scenario runs 8 models in parallel
#
# Usage: bash run_all_evaluations.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Master Evaluation Script"
echo "  Running all 12 scenarios (NORMAL, SIMPLE, COMPLEX × text/vision × 4B/8B)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# List of all config files
CONFIGS=(
    "configs/normal_text_4b.yaml"
    "configs/normal_text_8b.yaml"
    "configs/normal_vision_4b.yaml"
    "configs/normal_vision_8b.yaml"
    "configs/simple_text_4b.yaml"
    "configs/simple_text_8b.yaml"
    "configs/simple_vision_4b.yaml"
    "configs/simple_vision_8b.yaml"
    "configs/complex_text_4b.yaml"
    "configs/complex_text_8b.yaml"
    "configs/complex_vision_4b.yaml"
    "configs/complex_vision_8b.yaml"
)

TOTAL=${#CONFIGS[@]}
SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

echo "Found $TOTAL evaluation scenarios"
echo ""

# Run each scenario
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NUM=$((i + 1))
    
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Scenario $NUM/$TOTAL: $CONFIG"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    if [ ! -f "$CONFIG" ]; then
        echo "⚠️  Config file not found: $CONFIG"
        echo "   Skipping..."
        echo ""
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Run the scenario
    if bash run_parallel_evaluation.sh --config "$CONFIG"; then
        echo ""
        echo "✅ Scenario $NUM/$TOTAL complete: $CONFIG"
        echo ""
        SUCCESS=$((SUCCESS + 1))
    else
        echo ""
        echo "❌ Scenario $NUM/$TOTAL failed: $CONFIG"
        echo ""
        FAILED=$((FAILED + 1))
    fi
    
    # Pause between scenarios to let GPUs cool down
    if [ $NUM -lt $TOTAL ]; then
        echo "⏸️  Pausing 30 seconds before next scenario..."
        sleep 30
        echo ""
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✅ ALL EVALUATIONS COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  Total scenarios:  $TOTAL"
echo "  Successful:       $SUCCESS"
echo "  Failed:           $FAILED"
echo "  Total time:       ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved in evaluation_results/"
echo ""
