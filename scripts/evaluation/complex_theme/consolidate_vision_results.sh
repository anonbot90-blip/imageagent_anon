#!/bin/bash
# Consolidate vision evaluation results from 8 models into unified comparison
# Creates per-sample folders with 9-way comparison (Original | B | E | S | R | RW | D | SW | GT)

# set -e  # Exit on error

# NOTE: Conda environment activation removed - relies on parent shell activation
# # Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate img-agent

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"  # Ensure we're in project root for relative paths
BASELINE_DIR=""
EDIT_ONLY_DIR=""
STANDARD_VISION_DIR=""
RL_VISION_DIR=""
RW_VISION_DIR=""
DPO_VISION_DIR=""
SW_VISION_DIR=""
GPT4O_DIR=""
OUTPUT_DIR=""
CHECKPOINT_INTERVAL=2  # Save checkpoints every N samples

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-dir)
            BASELINE_DIR="$2"
            shift 2
            ;;
        --edit-only-dir)
            EDIT_ONLY_DIR="$2"
            shift 2
            ;;
        --standard-vision-dir)
            STANDARD_VISION_DIR="$2"
            shift 2
            ;;
        --rl-vision-dir)
            RL_VISION_DIR="$2"
            shift 2
            ;;
        --rw-vision-dir)
            RW_VISION_DIR="$2"
            shift 2
            ;;
        --dpo-vision-dir)
            DPO_VISION_DIR="$2"
            shift 2
            ;;
        --sw-vision-dir)
            SW_VISION_DIR="$2"
            shift 2
            ;;
        --gpt4o-dir)
            GPT4O_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: bash scripts/evaluation/consolidate_vision_results.sh [OPTIONS]

Consolidates vision evaluation results from 8 models into a unified comparison structure.

Required Options:
  --baseline-dir DIR           Baseline evaluation results directory
  --edit-only-dir DIR          Edit-Only evaluation results directory
  --standard-vision-dir DIR    Standard Vision evaluation results directory
  --rl-vision-dir DIR          RL Vision evaluation results directory
  --rw-vision-dir DIR          RW Vision evaluation results directory
  --dpo-vision-dir DIR         DPO Vision evaluation results directory
  --sw-vision-dir DIR          SW Vision evaluation results directory
  --gpt4o-dir DIR              GPT-4o Planner evaluation results directory
  --output-dir DIR             Output directory for consolidated results

Examples:
  bash scripts/evaluation/consolidate_vision_results.sh \\
    --baseline-dir evaluation_results/vision_parallel_eval/baseline \\
    --edit-only-dir evaluation_results/vision_parallel_eval/edit_only \\
    --standard-vision-dir evaluation_results/vision_parallel_eval/standard_vision \\
    --rl-vision-dir evaluation_results/vision_parallel_eval/rl_vision \\
    --rw-vision-dir evaluation_results/vision_parallel_eval/rw_vision \\
    --dpo-vision-dir evaluation_results/vision_parallel_eval/dpo_vision \\
    --sw-vision-dir evaluation_results/vision_parallel_eval/sw_vision \\
    --gpt4o-dir evaluation_results/vision_parallel_eval/gpt4o \\
    --output-dir evaluation_results/vision_parallel_eval/consolidated_vision
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$BASELINE_DIR" ] || [ -z "$EDIT_ONLY_DIR" ] || [ -z "$STANDARD_VISION_DIR" ] || [ -z "$RL_VISION_DIR" ] || [ -z "$RW_VISION_DIR" ] || [ -z "$DPO_VISION_DIR" ] || [ -z "$SW_VISION_DIR" ] || [ -z "$GPT4O_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: Missing required arguments"
    echo "Use --help for usage information"
    exit 1
fi

# Validate input directories exist
for dir in "$BASELINE_DIR" "$EDIT_ONLY_DIR" "$STANDARD_VISION_DIR" "$RL_VISION_DIR" "$RW_VISION_DIR" "$DPO_VISION_DIR" "$SW_VISION_DIR" "$GPT4O_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Error: Directory not found: $dir"
        exit 1
    fi
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  CONSOLIDATING VISION EVALUATION RESULTS (8 MODELS)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Input directories:"
echo "  Baseline:         $BASELINE_DIR"
echo "  Edit-Only:        $EDIT_ONLY_DIR"
echo "  Standard Vision:  $STANDARD_VISION_DIR"
echo "  RL Vision:        $RL_VISION_DIR"
echo "  RW Vision:        $RW_VISION_DIR"
echo "  DPO Vision:       $DPO_VISION_DIR"
echo "  SW Vision:        $SW_VISION_DIR"
echo "  GPT-4o:           $GPT4O_DIR"
echo ""
echo "Output directory:"
echo "  Consolidated:     $OUTPUT_DIR"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: CREATE OUTPUT STRUCTURE
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [1/4] Creating output directory structure"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$OUTPUT_DIR/samples"

echo "✅ Output structure created"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: CONSOLIDATE PER-SAMPLE RESULTS
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [2/4] Consolidating per-sample results"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python3 << EOF
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Read arguments from environment
baseline_dir = Path("$BASELINE_DIR")
edit_only_dir = Path("$EDIT_ONLY_DIR")
standard_vision_dir = Path("$STANDARD_VISION_DIR")
rl_vision_dir = Path("$RL_VISION_DIR")
rw_vision_dir = Path("$RW_VISION_DIR")
dpo_vision_dir = Path("$DPO_VISION_DIR")
sw_vision_dir = Path("$SW_VISION_DIR")
gpt4o_dir = Path("$GPT4O_DIR")
output_dir = Path("$OUTPUT_DIR")
checkpoint_interval = $CHECKPOINT_INTERVAL

# Create checkpoints directory
checkpoints_dir = output_dir / "checkpoints"
checkpoints_dir.mkdir(parents=True, exist_ok=True)

# Get sample directories from baseline
baseline_samples = baseline_dir / "samples"
if not baseline_samples.exists():
    print(f"❌ Error: Samples directory not found: {baseline_samples}")
    exit(1)

sample_dirs = sorted([d for d in baseline_samples.iterdir() if d.is_dir()])
print(f"Found {len(sample_dirs)} samples to consolidate")
print(f"Checkpoints will be saved every {checkpoint_interval} samples")
print("")

# Process each sample
for i, baseline_sample_dir in enumerate(sample_dirs, 1):
    sample_id = baseline_sample_dir.name
    print(f"  [{i}/{len(sample_dirs)}] Processing {sample_id}...")
    
    # Create output sample directory
    output_sample_dir = output_dir / "samples" / sample_id
    output_sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Get corresponding directories for other models
    edit_only_sample_dir = edit_only_dir / "samples" / sample_id
    standard_vision_sample_dir = standard_vision_dir / "samples" / sample_id
    rl_vision_sample_dir = rl_vision_dir / "samples" / sample_id
    rw_vision_sample_dir = rw_vision_dir / "samples" / sample_id
    dpo_vision_sample_dir = dpo_vision_dir / "samples" / sample_id
    sw_vision_sample_dir = sw_vision_dir / "samples" / sample_id
    gpt4o_sample_dir = gpt4o_dir / "samples" / sample_id
    # Copy original and ground truth (same for all models)
    for filename in ["original.png", "ground_truth.png"]:
        src = baseline_sample_dir / filename
        if src.exists():
            shutil.copy2(src, output_sample_dir / filename)
    
    # Copy baseline results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = baseline_sample_dir / filename
        if src.exists():
            # Rename predicted_plan.json to plan.json for consistency
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"baseline_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy edit-only results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = edit_only_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"edit_only_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy standard vision results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = standard_vision_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"standard_vision_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy RL vision results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = rl_vision_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"rl_vision_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy RW vision results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = rw_vision_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"rw_vision_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy DPO vision results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = dpo_vision_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"dpo_vision_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy SW vision results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = sw_vision_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan")
            dst = output_sample_dir / f"sw_vision_{dst_name}"
            shutil.copy2(src, dst)
    
    # Copy GPT-4o results
    for filename in ["predicted_edit.png", "predicted_plan.json"]:
        src = gpt4o_sample_dir / filename
        if src.exists():
            dst_name = filename.replace("predicted_plan", "plan").replace("predicted_edit", "gpt4o_predicted_edit")
            dst = output_sample_dir / dst_name
            shutil.copy2(src, dst)
    
    # Get user prompt for this sample
    user_prompt = None
    baseline_plan_file = output_sample_dir / "baseline_plan.json"
    if baseline_plan_file.exists():
        try:
            with open(baseline_plan_file, 'r') as f:
                plan_data = json.load(f)
                # Try both 'overall_instruction' and 'user_prompt' fields
                user_prompt = plan_data.get('overall_instruction', plan_data.get('user_prompt', ''))
        except:
            pass
    
    # Create 10-way comparison (Original | B | E | S | R | RW | D | SW | GPT-4o | GT)
    # [Original | Baseline | E | Standard | RL | RW | DPO | SW | GPT-4o | Ground Truth]
    try:
        labels = ['Original', 'Baseline', 'E', 'Standard', 'RL', 'RW', 'DPO', 'SW', 'GPT-4o', 'Ground Truth']
        paths = [
            output_sample_dir / "original.png",
            output_sample_dir / "baseline_predicted_edit.png",
            output_sample_dir / "edit_only_predicted_edit.png",
            output_sample_dir / "standard_vision_predicted_edit.png",
            output_sample_dir / "rl_vision_predicted_edit.png",
            output_sample_dir / "rw_vision_predicted_edit.png",
            output_sample_dir / "dpo_vision_predicted_edit.png",
            output_sample_dir / "sw_vision_predicted_edit.png",
            output_sample_dir / "gpt4o_predicted_edit.png",
            output_sample_dir / "ground_truth.png"
        ]
        
        # Load images
        loaded_images = []
        target_height = 512
        
        for path, label in zip(paths, labels):
            if path.exists():
                img = Image.open(path).convert('RGB')
                aspect_ratio = img.width / img.height
                new_width = int(target_height * aspect_ratio)
                img = img.resize((new_width, target_height), Image.LANCZOS)
                loaded_images.append(img)
            else:
                # Placeholder if image missing
                placeholder = Image.new('RGB', (target_height, target_height), (200, 200, 200))
                draw = ImageDraw.Draw(placeholder)
                draw.text((target_height//4, target_height//2), "Missing", fill=(100, 100, 100))
                loaded_images.append(placeholder)
        
        # Create canvas with space for labels and prompt
        label_height = 50
        prompt_height = 100 if user_prompt else 20
        total_width = sum(img.width for img in loaded_images) + 20 * (len(loaded_images) - 1)  # 20px gaps
        total_height = target_height + label_height + prompt_height
        
        canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        # Load fonts
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Paste images and add labels
        x_offset = 0
        for img, label in zip(loaded_images, labels):
            # Paste image
            canvas.paste(img, (x_offset, label_height))
            
            # Add label above image (centered)
            bbox = draw.textbbox((0, 0), label, font=font_large)
            text_width = bbox[2] - bbox[0]
            text_x = x_offset + (img.width - text_width) // 2
            draw.text((text_x, 10), label, fill="black", font=font_large)
            
            x_offset += img.width + 20
        
        # Add user prompt at bottom if available
        if user_prompt:
            prompt_y = target_height + label_height + 10
            # Wrap text
            max_chars_per_line = 180
            words = user_prompt.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw prompt lines
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                draw.text((10, prompt_y + i * 20), f"Prompt: {line}" if i == 0 else line, fill="black", font=font_small)
        
        # Save comparison image
        canvas.save(output_sample_dir / "comparison_7way.png")
        
    except Exception as e:
        print(f"    ⚠️  Failed to create comparison image: {e}")
    
    # Save checkpoint every N samples
    if i % checkpoint_interval == 0:
        print(f"    💾 Saving checkpoint at sample {i}...")

print("")
print(f"✅ Consolidated {len(sample_dirs)} samples")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to consolidate per-sample results"
    exit 1
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2.5: SAVE CHECKPOINTS (IF ANY WERE CREATED)
# ════════════════════════════════════════════════════════════════════════════════

# Check if checkpoints directory has any checkpoints
if [ -d "$OUTPUT_DIR/checkpoints" ] && [ "$(ls -A $OUTPUT_DIR/checkpoints 2>/dev/null)" ]; then
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  [2.5/5] Processing intermediate checkpoints"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    # For each checkpoint, generate consolidated metrics and summary
    for checkpoint_dir in "$OUTPUT_DIR"/checkpoints/checkpoint_*; do
        if [ -d "$checkpoint_dir" ]; then
            checkpoint_name=$(basename "$checkpoint_dir")
            echo "📦 Processing $checkpoint_name..."
            
            # Generate metrics for this checkpoint
            python3 << CHECKPOINT_EOF
import json
from pathlib import Path
import sys

baseline_dir = Path("$BASELINE_DIR")
edit_only_dir = Path("$EDIT_ONLY_DIR")
standard_vision_dir = Path("$STANDARD_VISION_DIR")
rl_vision_dir = Path("$RL_VISION_DIR")
rw_vision_dir = Path("$RW_VISION_DIR")
dpo_vision_dir = Path("$DPO_VISION_DIR")
sw_vision_dir = Path("$SW_VISION_DIR")
checkpoint_dir = Path("$checkpoint_dir")

# Load summary results
try:
    baseline_summary = json.load(open(baseline_dir / "evaluation_summary_all.json"))
    edit_only_summary = json.load(open(edit_only_dir / "evaluation_summary_test.json"))
    standard_vision_summary = json.load(open(standard_vision_dir / "evaluation_summary_all.json"))
    rl_vision_summary = json.load(open(rl_vision_dir / "evaluation_summary_all.json"))
    rw_vision_summary = json.load(open(rw_vision_dir / "evaluation_summary_all.json"))
    dpo_vision_summary = json.load(open(dpo_vision_dir / "evaluation_summary_all.json"))
    sw_vision_summary = json.load(open(sw_vision_dir / "evaluation_summary_all.json"))
    gpt4o_summary = json.load(open(gpt4o_dir / "evaluation_summary_all.json"))
    
    # Load detailed results
    baseline_detailed = json.load(open(baseline_dir / "detailed_results_all.json"))
    edit_only_detailed = json.load(open(edit_only_dir / "detailed_results_test.json"))
    standard_vision_detailed = json.load(open(standard_vision_dir / "detailed_results_all.json"))
    rl_vision_detailed = json.load(open(rl_vision_dir / "detailed_results_all.json"))
    rw_vision_detailed = json.load(open(rw_vision_dir / "detailed_results_all.json"))
    dpo_vision_detailed = json.load(open(dpo_vision_dir / "detailed_results_all.json"))
    sw_vision_detailed = json.load(open(sw_vision_dir / "detailed_results_all.json"))
    # Create checkpoint summary
    consolidated_summary = {
        "experiment_type": "vision_7model_checkpoint",
        "checkpoint": "$checkpoint_name",
        "labels": ["Baseline", "Edit-Only", "Standard", "RL", "RW", "DPO", "SW"],
        "baseline": baseline_summary,
        "edit_only": edit_only_summary,
        "standard_vision": standard_vision_summary,
        "rl_vision": rl_vision_summary,
        "rw_vision": rw_vision_summary,
        "dpo_vision": dpo_vision_summary,
        "sw_vision": sw_vision_summary
    }
    
    # Save checkpoint summary
    with open(checkpoint_dir / "consolidated_summary.json", 'w') as f:
        json.dump(consolidated_summary, f, indent=2)
    
    print(f"  ✅ Saved checkpoint metrics")
    
except Exception as e:
    print(f"  ⚠️  Failed to generate checkpoint metrics: {e}")
    sys.exit(0)  # Non-critical, continue
CHECKPOINT_EOF
            
            # Generate summary report for checkpoint
            python scripts/evaluation/final_metric_summary.py \
                --results "$checkpoint_dir/consolidated_summary.json" \
                --output "$checkpoint_dir/FINAL_SUMMARY.md" \
                --generate-images > /dev/null 2>&1 || true
            
            echo "  ✅ Checkpoint $checkpoint_name processed"
            echo ""
        fi
    done
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: CONSOLIDATE METRICS
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [3/5] Consolidating final metrics"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python3 << EOF
import json
from pathlib import Path
import numpy as np

# Read arguments from environment
baseline_dir = Path("$BASELINE_DIR")
edit_only_dir = Path("$EDIT_ONLY_DIR")
standard_vision_dir = Path("$STANDARD_VISION_DIR")
rl_vision_dir = Path("$RL_VISION_DIR")
rw_vision_dir = Path("$RW_VISION_DIR")
dpo_vision_dir = Path("$DPO_VISION_DIR")
sw_vision_dir = Path("$SW_VISION_DIR")
gpt4o_dir = Path("$GPT4O_DIR")
output_dir = Path("$OUTPUT_DIR")

# Load summary results
baseline_summary = json.load(open(baseline_dir / "evaluation_summary_all.json"))
edit_only_summary = json.load(open(edit_only_dir / "evaluation_summary_test.json"))
standard_vision_summary = json.load(open(standard_vision_dir / "evaluation_summary_all.json"))
rl_vision_summary = json.load(open(rl_vision_dir / "evaluation_summary_all.json"))
rw_vision_summary = json.load(open(rw_vision_dir / "evaluation_summary_all.json"))
dpo_vision_summary = json.load(open(dpo_vision_dir / "evaluation_summary_all.json"))
sw_vision_summary = json.load(open(sw_vision_dir / "evaluation_summary_all.json"))
# Load detailed results
baseline_detailed = json.load(open(baseline_dir / "detailed_results_all.json"))
edit_only_detailed = json.load(open(edit_only_dir / "detailed_results_test.json"))
standard_vision_detailed = json.load(open(standard_vision_dir / "detailed_results_all.json"))
rl_vision_detailed = json.load(open(rl_vision_dir / "detailed_results_all.json"))
rw_vision_detailed = json.load(open(rw_vision_dir / "detailed_results_all.json"))
dpo_vision_detailed = json.load(open(dpo_vision_dir / "detailed_results_all.json"))
sw_vision_detailed = json.load(open(sw_vision_dir / "detailed_results_all.json"))
gpt4o_summary = json.load(open(gpt4o_dir / "evaluation_summary_all.json"))
gpt4o_detailed = json.load(open(gpt4o_dir / "detailed_results_all.json"))

# Create consolidated summary
consolidated_summary = {
    "experiment_type": "vision_8model",
    "num_samples": baseline_summary.get("num_samples", 0),
    "labels": ["Baseline", "Edit-Only", "Standard", "RL", "RW", "DPO", "SW", "GPT-4o"],
    "baseline": baseline_summary,
    "edit_only": edit_only_summary,
    "standard_vision": standard_vision_summary,
    "rl_vision": rl_vision_summary,
    "rw_vision": rw_vision_summary,
    "dpo_vision": dpo_vision_summary,
    "sw_vision": sw_vision_summary,
    "gpt4o": gpt4o_summary
}

# Create consolidated detailed results
consolidated_detailed = {
    "experiment_type": "vision_8model",
    "num_samples": len(baseline_detailed),
    "samples": []
}

# Merge per-sample results
for b_sample in baseline_detailed:
    sample_id = b_sample["sample_id"]
    
    # Find corresponding samples in other models
    e_sample = next((s for s in edit_only_detailed if s["sample_id"] == sample_id), None)
    sv_sample = next((s for s in standard_vision_detailed if s["sample_id"] == sample_id), None)
    rv_sample = next((s for s in rl_vision_detailed if s["sample_id"] == sample_id), None)
    rw_sample = next((s for s in rw_vision_detailed if s["sample_id"] == sample_id), None)
    sw_sample = next((s for s in sw_vision_detailed if s["sample_id"] == sample_id), None)
    dpo_sample = next((s for s in dpo_vision_detailed if s["sample_id"] == sample_id), None)
    gpt4o_sample = next((s for s in gpt4o_detailed if s["sample_id"] == sample_id), None)
    
    merged_sample = {
        "sample_id": sample_id,
        "user_prompt": b_sample.get("user_prompt", "N/A"),
        "baseline": b_sample,
        "edit_only": e_sample if e_sample else {},
        "standard_vision": sv_sample if sv_sample else {},
        "rl_vision": rv_sample if rv_sample else {},
        "rw_vision": rw_sample if rw_sample else {},
        "dpo_vision": dpo_sample if dpo_sample else {},
        "sw_vision": sw_sample if sw_sample else {},
        "gpt4o": gpt4o_sample if gpt4o_sample else {}
    }
    
    consolidated_detailed["samples"].append(merged_sample)

# Save consolidated results
with open(output_dir / "consolidated_summary.json", 'w') as f:
    json.dump(consolidated_summary, f, indent=2)

with open(output_dir / "consolidated_detailed.json", 'w') as f:
    json.dump(consolidated_detailed, f, indent=2)

print("✅ Metrics consolidated")
print(f"   Summary: {output_dir / 'consolidated_summary.json'}")
print(f"   Detailed: {output_dir / 'consolidated_detailed.json'}")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed to consolidate metrics"
    exit 1
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [4/5] Generating summary report"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python scripts/evaluation/final_metric_summary.py --results "$OUTPUT_DIR/consolidated_summary.json" --output "$OUTPUT_DIR/FINAL_SUMMARY.md"

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Failed to generate summary report (non-critical)"
else
    echo "✅ Summary report generated"
    echo "   Markdown: $OUTPUT_DIR/FINAL_SUMMARY.md"
fi

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5: GENERATE COMPARISON TABLES (NEW STYLED)
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [5/5] Generating Comparison Tables (New Styled)"
echo "════════════════════════════════════════════════════════════════════════════════"

echo ""
echo "📊 Generating all comparison tables with new styling..."
python scripts/evaluation/generate_comparison_tables.py \
    --results "$OUTPUT_DIR/consolidated_summary.json" \
    --output-dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ All comparison tables generated:"
    echo "   • Planner Metrics (N/A for Edit-Only)"
    echo "   • Image Metrics (incl. CLIP Score)"
    echo "   • GPT-4o Action Judge (11 metrics)"
    echo "   • GPT-4o Image Quality (7 metrics, N/A for Edit-Only)"
    echo "   Style: Blue/white with grey GPT-4o column"
else
    echo "⚠️  Warning: Failed to generate comparison tables (non-critical)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✅ CONSOLIDATION COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Consolidated results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Output structure:"
echo "   samples/                      # Per-sample folders"
echo "   ├── sample_001/"
echo "   │   ├── original.png"
echo "   │   ├── ground_truth.png"
echo "   │   ├── baseline_predicted_edit.png"
echo "   │   ├── baseline_plan.json"
echo "   │   ├── edit_only_predicted_edit.png"
echo "   │   ├── edit_only_prompt.txt"
echo "   │   ├── standard_vision_predicted_edit.png"
echo "   │   ├── standard_vision_plan.json"
echo "   │   ├── rl_vision_predicted_edit.png"
echo "   │   ├── rl_vision_plan.json"
echo "   │   ├── rw_vision_predicted_edit.png"
echo "   │   ├── rw_vision_plan.json"
echo "   │   ├── dpo_vision_predicted_edit.png"
echo "   │   ├── dpo_vision_plan.json"
echo "   │   ├── sw_vision_predicted_edit.png"
echo "   │   ├── sw_vision_plan.json"
echo "   │   └── comparison_7way.png   # 7-way comparison (Original | B | E | S | R | RW | D | SW | GT)"
echo "   ├── ..."
echo "   consolidated_summary.json     # Aggregated metrics (all 8 models)"
echo "   consolidated_detailed.json    # Per-sample results"
echo "   FINAL_SUMMARY.md              # Markdown report"
echo "   planner_metrics_table.png     # Visual tables (B | E | S | R | RW | D | SW | GPT-4o)"
echo "   image_metrics_table.png       # Objective metrics + GPT-4o scores"
echo "   overall_summary_table.png"
echo ""

