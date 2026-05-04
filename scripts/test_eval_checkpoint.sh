#!/bin/bash
# Test script for --eval-checkpoint feature
# Tests with 4 samples and --eval-checkpoint 2

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
TEST_OUTPUT="$PROJECT_ROOT/evaluation_results/test_eval_checkpoint"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  TESTING --eval-checkpoint FEATURE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Test configuration:"
echo "  Samples: 4"
echo "  Eval checkpoint: 2"
echo "  Expected batches: 2"
echo "  Expected tables: After batch 1 (N=2) and batch 2 (N=4)"
echo ""

# Clean previous test results
rm -rf "$TEST_OUTPUT"
mkdir -p "$TEST_OUTPUT"

echo "Starting test evaluation..."
echo ""

cd "$PROJECT_ROOT"

bash scripts/evaluation/start_eval_all_planner_text.sh \
    --num-samples 4 \
    --eval-checkpoint 2 \
    --output "$TEST_OUTPUT"

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  TEST RESULTS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    echo ""
    
    # Check for checkpoint directories
    if [ -d "$TEST_OUTPUT/consolidated_text/checkpoints" ]; then
        echo "✅ Checkpoints directory exists"
        
        CHECKPOINT_002="$TEST_OUTPUT/consolidated_text/checkpoints/checkpoint_002"
        CHECKPOINT_004="$TEST_OUTPUT/consolidated_text/checkpoints/checkpoint_004"
        
        if [ -d "$CHECKPOINT_002" ]; then
            echo "✅ Checkpoint 002 exists"
            ls "$CHECKPOINT_002"/*.png 2>/dev/null && echo "   - Tables present" || echo "   ⚠️  No tables found"
        else
            echo "❌ Checkpoint 002 missing"
        fi
        
        if [ -d "$CHECKPOINT_004" ]; then
            echo "✅ Checkpoint 004 exists"
            ls "$CHECKPOINT_004"/*.png 2>/dev/null && echo "   - Tables present" || echo "   ⚠️  No tables found"
        else
            echo "❌ Checkpoint 004 missing"
        fi
    else
        echo "❌ Checkpoints directory not found"
    fi
    
    echo ""
    echo "📊 Final consolidated tables:"
    ls -lh "$TEST_OUTPUT/consolidated_text"/*.png 2>/dev/null || echo "   ⚠️  No final tables found"
    
    echo ""
    echo "📂 Test results location:"
    echo "   $TEST_OUTPUT"
    echo ""
    echo "✅ TEST PASSED"
else
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check logs:"
    echo "   ls $TEST_OUTPUT/logs/*.log"
    echo ""
    echo "❌ TEST FAILED"
fi

