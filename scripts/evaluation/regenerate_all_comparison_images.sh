#!/usr/bin/env bash
#
# Regenerate all 9-way and 10-way comparison images
# Processes all example folders in consolidated_results
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENHANCE_SCRIPT="$PROJECT_ROOT/consolidated_results/scripts/enhance_comparison_images.py"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          Regenerate 9-way and 10-way Comparison Images                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if enhance script exists
if [ ! -f "$ENHANCE_SCRIPT" ]; then
    echo "❌ Error: enhance_comparison_images.py not found at:"
    echo "   $ENHANCE_SCRIPT"
    exit 1
fi

# Process all improvement directories
DATASETS=(
    "text_4b_improvements"
    "text_8b_improvements"
    "vision_4b_improvements"
    "vision_8b_improvements"
)

CATEGORIES=(
    ""  # regular
    "complex"
    "complex_v2"
)

total_success=0
total_failed=0

for category in "${CATEGORIES[@]}"; do
    if [ -z "$category" ]; then
        category_path="$PROJECT_ROOT/consolidated_results"
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "  Processing REGULAR datasets"
        echo "════════════════════════════════════════════════════════════════════════════════"
    else
        category_path="$PROJECT_ROOT/consolidated_results/$category"
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "  Processing $(echo $category | tr '[:lower:]' '[:upper:]') datasets"
        echo "════════════════════════════════════════════════════════════════════════════════"
    fi
    
    for dataset in "${DATASETS[@]}"; do
        examples_dir="$category_path/$dataset/examples"
        
        if [ ! -d "$examples_dir" ]; then
            echo ""
            echo "⚠️  Skipping $dataset ($category): examples directory not found"
            continue
        fi
        
        echo ""
        echo "📦 Processing: $dataset ($category)"
        echo "   Examples dir: $examples_dir"
        
        # Run the enhancement script
        if python3 "$ENHANCE_SCRIPT" --base-dir "$examples_dir"; then
            echo "   ✅ Success"
            ((total_success++))
        else
            echo "   ❌ Failed"
            ((total_failed++))
        fi
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         REGENERATION COMPLETE                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Summary:"
echo "   • Successful: $total_success datasets"
echo "   • Failed: $total_failed datasets"
echo "   • Total: $((total_success + total_failed)) datasets"
echo ""

if [ $total_failed -gt 0 ]; then
    echo "⚠️  Some datasets failed to process"
    exit 1
else
    echo "✅ All datasets processed successfully"
    exit 0
fi

