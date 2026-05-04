#!/bin/bash
# Simple wrapper to generate evaluation summary
# Usage: ./scripts/generate_summary.sh <evaluation_summary.json> [--with-images]

# Check if filename provided
if [ -z "$1" ]; then
    echo "❌ Error: No input file provided"
    echo ""
    echo "Usage: $0 <evaluation_summary.json> [--with-images]"
    echo ""
    echo "Example:"
    echo "  $0 planner_evaluation_results/final_val_5000samples/evaluation_summary_val.json"
    echo "  $0 planner_evaluation_results/final_val_5000samples/evaluation_summary_val.json --with-images"
    exit 1
fi

INPUT_FILE="$1"
GENERATE_IMAGES=false

# Check for --with-images flag
if [ "$2" = "--with-images" ]; then
    GENERATE_IMAGES=true
fi

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: File not found: $INPUT_FILE"
    exit 1
fi

# Get directory and basename
INPUT_DIR=$(dirname "$INPUT_FILE")
OUTPUT_FILE="$INPUT_DIR/FINAL_SUMMARY.md"

echo "=================================================="
echo "  📊 Generating Evaluation Summary"
echo "=================================================="
echo ""
echo "📂 Input:  $INPUT_FILE"
echo "📄 Output: $OUTPUT_FILE"
if [ "$GENERATE_IMAGES" = true ]; then
    echo "🎨 Mode:   With table images"
else
    echo "📝 Mode:   Markdown only"
fi
echo ""

# Build command
CMD="python3 scripts/final_metric_summary.py --results \"$INPUT_FILE\" --output \"$OUTPUT_FILE\""

# Add image generation flag if requested
if [ "$GENERATE_IMAGES" = true ]; then
    CMD="$CMD --generate-images"
fi

# Run Python script
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✅ Summary Generated Successfully!"
    echo "=================================================="
    echo ""
    echo "📖 View the report:"
    echo "   cat $OUTPUT_FILE"
    echo ""
    echo "   or"
    echo ""
    echo "   less $OUTPUT_FILE"
    echo ""
    
    if [ "$GENERATE_IMAGES" = true ]; then
        echo "🖼️  Table images:"
        echo "   $INPUT_DIR/planner_metrics_table.png"
        echo "   $INPUT_DIR/image_metrics_table.png"
        echo "   $INPUT_DIR/overall_summary_table.png"
        echo ""
    fi
else
    echo ""
    echo "❌ Error generating summary"
    exit 1
fi

