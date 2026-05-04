#!/usr/bin/env bash
#
# Extract method-specific examples for all 12 datasets
# Then generate 9-way and 10-way comparison images
#

set -uo pipefail  # Removed -e to continue on errors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTRACT_SCRIPT="$PROJECT_ROOT/consolidated_results/scripts/extract_method_examples.py"
ENHANCE_SCRIPT="$PROJECT_ROOT/consolidated_results/scripts/enhance_comparison_images.py"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║             Extract Examples & Generate Comparison Images                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if scripts exist
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "❌ Error: extract_method_examples.py not found"
    exit 1
fi

if [ ! -f "$ENHANCE_SCRIPT" ]; then
    echo "❌ Error: enhance_comparison_images.py not found"
    exit 1
fi

# Number of samples per method
NUM_SAMPLES=10


# Track statistics
total_success=0
total_failed=0

# Function to process with explicit paths
process_with_paths() {
    local improvements_dir=$1
    local eval_dir=$2
    local consolidated_suffix=$3  # "text" or "vision"
    local description=$4
    
    local selected_traj="$improvements_dir/selected_trajectories.json"
    local source_samples="$eval_dir/consolidated_${consolidated_suffix}/samples"
    local output_examples="$improvements_dir/examples"
    
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "📦 $description"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    # Check prerequisites
    if [ ! -f "$selected_traj" ]; then
        echo "   ⚠️  Skipping: selected_trajectories.json not found"
        return 1
    fi
    
    if [ ! -d "$source_samples" ]; then
        echo "   ⚠️  Skipping: consolidated samples not found"
        echo "      Path: $source_samples"
        return 1
    fi
    
    # Step 1: Extract examples
    echo "   [1/2] Extracting examples..."
    if ! python3 "$EXTRACT_SCRIPT" \
        --selected-trajectories "$selected_traj" \
        --source-samples "$source_samples" \
        --output-dir "$output_examples" \
        --num-samples "$NUM_SAMPLES" 2>&1 | grep -E "(📦|✓|⚠️|✅)" | sed 's/^/      /'; then
        echo "      ❌ Failed to extract examples"
        return 1
    fi
    
    # Step 2: Generate comparison images
    echo "   [2/2] Generating 9-way and 10-way comparison images..."
    if ! python3 "$ENHANCE_SCRIPT" \
        --base-dir "$output_examples" 2>&1 | grep -E "(📦|✓|⚠️|✅)" | sed 's/^/      /'; then
        echo "      ❌ Failed to generate comparison images"
        return 1
    fi
    
    echo "   ✅ Complete"
    echo ""
    return 0
}

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  REGULAR DATASETS (4/12)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

process_with_paths "$PROJECT_ROOT/consolidated_results/text_4b_improvements" "$PROJECT_ROOT/evaluation_results/text_parallel_cot_4b_trajectory" "text" "Text 4B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/text_8b_improvements" "$PROJECT_ROOT/evaluation_results/text_parallel_cot_8b_trajectory" "text" "Text 8B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/vision_4b_improvements" "$PROJECT_ROOT/evaluation_results/vision_parallel_cot_4b_trajectory" "vision" "Vision 4B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/vision_8b_improvements" "$PROJECT_ROOT/evaluation_results/vision_parallel_cot_8b_trajectory" "vision" "Vision 8B" && ((total_success++)) || ((total_failed++))

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  COMPLEX DATASETS (4/12)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

process_with_paths "$PROJECT_ROOT/consolidated_results/complex/text_4b_improvements" "$PROJECT_ROOT/evaluation_results/text_parallel_complex_cot_4b_trajectory" "text" "Complex Text 4B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/complex/text_8b_improvements" "$PROJECT_ROOT/evaluation_results/text_parallel_complex_cot_8b_trajectory" "text" "Complex Text 8B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/complex/vision_4b_improvements" "$PROJECT_ROOT/evaluation_results/vision_parallel_complex_cot_4b_trajectory" "vision" "Complex Vision 4B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/complex/vision_8b_improvements" "$PROJECT_ROOT/evaluation_results/vision_parallel_complex_cot_8b_trajectory" "vision" "Complex Vision 8B" && ((total_success++)) || ((total_failed++))

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  COMPLEX_V2 DATASETS (4/12)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

process_with_paths "$PROJECT_ROOT/consolidated_results/complex_v2/text_4b_improvements" "$PROJECT_ROOT/evaluation_results/text_parallel_complex_v2_cot_4b_trajectory" "text" "Complex_v2 Text 4B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/complex_v2/text_8b_improvements" "$PROJECT_ROOT/evaluation_results/text_parallel_complex_v2_cot_8b_trajectory" "text" "Complex_v2 Text 8B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/complex_v2/vision_4b_improvements" "$PROJECT_ROOT/evaluation_results/vision_parallel_complex_v2_cot_4b_trajectory" "vision" "Complex_v2 Vision 4B" && ((total_success++)) || ((total_failed++))
process_with_paths "$PROJECT_ROOT/consolidated_results/complex_v2/vision_8b_improvements" "$PROJECT_ROOT/evaluation_results/vision_parallel_complex_v2_cot_8b_trajectory" "vision" "Complex_v2 Vision 8B" && ((total_success++)) || ((total_failed++))

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         EXTRACTION COMPLETE                                  ║"
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
    echo "✅ All datasets processed successfully!"
    echo "   • Examples extracted with GPT-4o images"
    echo "   • 9-way comparison images generated (without Ground Truth)"
    echo "   • 10-way comparison images generated (with Ground Truth)"
    exit 0
fi

