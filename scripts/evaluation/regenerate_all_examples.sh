#!/bin/bash
# Regenerate all comparison examples for all 12 datasets with bold, large prompt text

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
ENHANCE_SCRIPT="$PROJECT_ROOT/consolidated_results/scripts/enhance_comparison_images.py"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  REGENERATING ALL COMPARISON EXAMPLES (12 DATASETS)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

COMPLETED=0
TOTAL=12

# Function to regenerate examples for a dataset
regenerate_examples() {
    local dataset=$1
    local description=$2
    
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "[$((COMPLETED+1))/$TOTAL] $description"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    if [ -d "$PROJECT_ROOT/$dataset/examples" ]; then
        python3 "$ENHANCE_SCRIPT" --base-dir "$PROJECT_ROOT/$dataset/examples"
        echo "✅ Completed: $description"
    else
        echo "⚠️  Directory not found: $PROJECT_ROOT/$dataset/examples"
    fi
    
    COMPLETED=$((COMPLETED+1))
    echo ""
}

# REGULAR DATASETS
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          REGULAR DATASETS (4/12)                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

regenerate_examples "consolidated_results/text_4b_improvements" "REGULAR - Text 4B"
regenerate_examples "consolidated_results/text_8b_improvements" "REGULAR - Text 8B"
regenerate_examples "consolidated_results/vision_4b_improvements" "REGULAR - Vision 4B"
regenerate_examples "consolidated_results/vision_8b_improvements" "REGULAR - Vision 8B"

# COMPLEX DATASETS
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          COMPLEX DATASETS (4/12)                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

regenerate_examples "consolidated_results/complex/text_4b_improvements" "COMPLEX - Text 4B"
regenerate_examples "consolidated_results/complex/text_8b_improvements" "COMPLEX - Text 8B"
regenerate_examples "consolidated_results/complex/vision_4b_improvements" "COMPLEX - Vision 4B"
regenerate_examples "consolidated_results/complex/vision_8b_improvements" "COMPLEX - Vision 8B"

# COMPLEX_V2 DATASETS
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         COMPLEX_V2 DATASETS (4/12)                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

regenerate_examples "consolidated_results/complex_v2/text_4b_improvements" "COMPLEX_V2 - Text 4B"
regenerate_examples "consolidated_results/complex_v2/text_8b_improvements" "COMPLEX_V2 - Text 8B"
regenerate_examples "consolidated_results/complex_v2/vision_4b_improvements" "COMPLEX_V2 - Vision 4B"
regenerate_examples "consolidated_results/complex_v2/vision_8b_improvements" "COMPLEX_V2 - Vision 8B"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ ALL COMPARISON EXAMPLES REGENERATED!"
echo "   • All prompt texts are now BOLD and 42pt"
echo "   • Headers remain BOLD and 48pt"
echo "   • Total datasets processed: $COMPLETED"
echo "════════════════════════════════════════════════════════════════════════════════"

