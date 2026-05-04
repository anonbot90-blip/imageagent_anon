#!/bin/bash
# Parallel Evaluation Script - Runs 8 models in parallel for ONE scenario
# Reads configuration from YAML file and launches existing evaluation scripts
# Usage: bash run_parallel_evaluation.sh --config configs/normal_text_4b.yaml

set -e  # Exit on error

# ════════════════════════════════════════════════════════════════════════════════
# PARSE ARGUMENTS
# ════════════════════════════════════════════════════════════════════════════════

CONFIG_FILE=""
NUM_SAMPLES_OVERRIDE=""
CHECKPOINT_INTERVAL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES_OVERRIDE="$2"
            shift 2
            ;;
        --checkpoint-interval)
            CHECKPOINT_INTERVAL_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: --config required"
    echo "Usage: bash run_parallel_evaluation.sh --config configs/normal_text_4b.yaml [--num-samples N] [--checkpoint-interval M]"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════════
# PARSE YAML CONFIG
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Parallel Model Evaluation"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Config: $CONFIG_FILE"
echo ""

# Parse YAML using Python
eval $(python3 << PYEOF
import yaml

config_file = "$CONFIG_FILE"
with open(config_file) as f:
    config = yaml.safe_load(f)

# Print bash variables
print(f"NUM_SAMPLES={config['evaluation']['num_samples']}")
print(f"CHECKPOINT_INTERVAL={config['evaluation']['checkpoint_interval']}")
print(f"CATEGORY={config['evaluation']['category']}")
print(f"MODEL_SIZE={config['evaluation']['model_size']}")
print(f"MODALITY={config['evaluation']['modality']}")
print(f"RESULTS_DIR={config['data']['results_dir']}")
print(f"TRAJECTORY_DIR={config['data']['trajectory_dir']}")
print(f"TEST_SAMPLES_FILE={config['data']['test_samples_file']}")
print(f"OUTPUT_DIR={config['data']['output_dir']}")
print(f"ACTION_LIBRARY={config['action_library']}")
print(f"EDITOR_TYPE={config['image_editor']['type']}")
print(f"EDITOR_CHECKPOINT={config['image_editor'].get('checkpoint', '')}")
print(f"EDITOR_CONFIG={config['image_editor'].get('config', '')}")
print(f"GPT_ENABLED={config['gpt_judge']['enabled']}")
print(f"GPT_ACTION_JUDGE={config['gpt_judge']['action_judge']}")

# Model configs
for model_id, model_config in config['models'].items():
    prefix = model_id.upper().replace('-', '_')
    print(f"{prefix}_NAME=\"{model_config['name']}\"")
    print(f"{prefix}_CHECKPOINT=\"{model_config['checkpoint']}\"")
    print(f"{prefix}_GPU={model_config['gpu'] if model_config['gpu'] is not None else ''}")
    print(f"{prefix}_TYPE=\"{model_config['type']}\"")

PYEOF
)

# Apply command-line overrides
if [ -n "$NUM_SAMPLES_OVERRIDE" ]; then
    NUM_SAMPLES=$NUM_SAMPLES_OVERRIDE
fi

if [ -n "$CHECKPOINT_INTERVAL_OVERRIDE" ]; then
    CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL_OVERRIDE
fi

# Set project root
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

# Full paths
RESULTS_DIR="$PROJECT_ROOT/$RESULTS_DIR"
TRAJECTORY_DIR="$PROJECT_ROOT/$TRAJECTORY_DIR"
TEST_SAMPLES_FILE="$PROJECT_ROOT/$TEST_SAMPLES_FILE"
OUTPUT_BASE="$PROJECT_ROOT/$OUTPUT_DIR"
ACTION_LIBRARY="$PROJECT_ROOT/$ACTION_LIBRARY"

# Data path
DATA_PATH="$TRAJECTORY_DIR/full_dataset_for_eval.json"

# Log directory
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

# Model output directories
BASELINE_DIR="$OUTPUT_BASE/baseline"
EDIT_ONLY_DIR="$OUTPUT_BASE/edit_only"
STANDARD_DIR="$OUTPUT_BASE/standard_text"
RL_DIR="$OUTPUT_BASE/rl_text"
RW_DIR="$OUTPUT_BASE/rw_text"
DPO_DIR="$OUTPUT_BASE/dpo_text"
SW_DIR="$OUTPUT_BASE/sw_text"
GPT4O_DIR="$OUTPUT_BASE/gpt4o"
CONSOLIDATED_DIR="$OUTPUT_BASE/consolidated_text"

# ════════════════════════════════════════════════════════════════════════════════
# DISPLAY CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

echo "Configuration:"
echo "  Category:    $CATEGORY"
echo "  Model Size:  $MODEL_SIZE"
echo "  Modality:    $MODALITY"
echo "  Samples:     $NUM_SAMPLES"
echo "  Checkpoint:  Every $CHECKPOINT_INTERVAL samples"
echo ""
echo "GPU Assignments:"
echo "  Baseline:    GPU $BASELINE_GPU"
echo "  Edit-Only:   GPU $EDIT_ONLY_GPU"
echo "  Standard:    GPU $STANDARD_GPU"
echo "  RL:          GPU $RL_GPU"
echo "  RW:          GPU $RW_GPU"
echo "  DPO:         GPU $DPO_GPU"
echo "  SW:          GPU $SW_GPU"
echo "  GPT-4o:      API"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: PREPARE EVALUATION DATASET
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [1/3] Preparing evaluation data"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$OUTPUT_BASE"

# Create full evaluation dataset if needed
if [ ! -f "$DATA_PATH" ]; then
    echo "📦 Creating full evaluation dataset..."
    
    python3 << PYEOF
import json
from pathlib import Path
from tqdm import tqdm

results_dir = Path("$RESULTS_DIR")
output_file = Path("$DATA_PATH")
output_file.parent.mkdir(parents=True, exist_ok=True)

samples = []
subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
print(f"Found {len(subdirs)} sample directories")

for subdir in tqdm(subdirs, desc="Processing"):
    action_plan_file = subdir / "action_plan.json"
    if action_plan_file.exists():
        try:
            with open(action_plan_file) as f:
                action_plan = json.load(f)
            
            sample = {
                "sample_id": subdir.name,
                "original_image_path": f"{results_dir.name}/{subdir.name}/original.png",
                "edited_image_path": f"{results_dir.name}/{subdir.name}/edited.png",
                "analysis_path": f"{results_dir.name}/{subdir.name}/analysis.json",
                "overall_instruction": action_plan.get("overall_instruction", ""),
                "user_prompt": action_plan.get("user_prompt", ""),
                "target_action_plan": action_plan,
                "metadata": {
                    "source_dir": f"{results_dir.name}/{subdir.name}",
                    "folder_name": subdir.name
                }
            }
            samples.append(sample)
        except:
            pass

output_data = {
    "version": "3.0-trajectory",
    "description": "Full dataset for trajectory-based evaluation",
    "total_samples": len(samples),
    "rl_filtered": False,
    "samples": samples
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Created {output_file} with {len(samples)} samples")
PYEOF
fi

# Select sample IDs
echo "📝 Selecting $NUM_SAMPLES samples..."

python3 << PYEOF
import json
import random
from pathlib import Path

data_path = Path("$DATA_PATH")
output_file = Path("$OUTPUT_BASE/selected_sample_ids.txt")
test_samples_file = Path("$TEST_SAMPLES_FILE")
num_samples = $NUM_SAMPLES

with open(data_path) as f:
    data = json.load(f)

if test_samples_file.exists():
    with open(test_samples_file) as f:
        test_sample_ids = [line.strip() for line in f if line.strip()]
    sample_ids = test_sample_ids[:num_samples]
else:
    all_sample_ids = [s["sample_id"] for s in data["samples"]]
    sample_ids = all_sample_ids[:num_samples]

with open(output_file, 'w') as f:
    for sid in sample_ids:
        f.write(f"{sid}\n")

print(f"✓ Selected {len(sample_ids)} samples")
PYEOF

echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: PARALLEL MODEL EVALUATION (BATCHED)
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [2/3] Running parallel evaluation"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $CHECKPOINT_INTERVAL -gt 0 ]; then
    NUM_BATCHES=$(( ($NUM_SAMPLES + $CHECKPOINT_INTERVAL - 1) / $CHECKPOINT_INTERVAL ))
    echo "Batching enabled: $NUM_BATCHES batches of $CHECKPOINT_INTERVAL samples"
else
    NUM_BATCHES=1
    echo "No batching: Processing all samples at once"
fi

echo ""

# Process samples in batches
for batch_num in $(seq 1 $NUM_BATCHES); do
    # Calculate sample range
    START_IDX=$(( ($batch_num - 1) * $CHECKPOINT_INTERVAL ))
    END_IDX=$(( $START_IDX + $CHECKPOINT_INTERVAL - 1 ))
    if [ $END_IDX -ge $NUM_SAMPLES ]; then
        END_IDX=$(( $NUM_SAMPLES - 1 ))
    fi
    
    BATCH_SIZE=$(( $END_IDX - $START_IDX + 1 ))
    BATCH_IDS_FILE="$OUTPUT_BASE/batch_${batch_num}_sample_ids.txt"
    
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  Batch $batch_num/$NUM_BATCHES: Samples $START_IDX-$END_IDX ($BATCH_SIZE samples)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    # Extract sample IDs for this batch
    tail -n +$(($START_IDX + 1)) "$OUTPUT_BASE/selected_sample_ids.txt" | head -n $BATCH_SIZE > "$BATCH_IDS_FILE"
    
    echo ""
    echo "🚀 Launching 8 models in parallel..."
    echo ""
    
    # Launch all 8 models in parallel using unified evaluator
    
    # 1. Baseline
    (CUDA_VISIBLE_DEVICES=$BASELINE_GPU python scripts/evaluation/evaluate_model.py \
        --model-type baseline \
        --checkpoint "$BASELINE_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$BASELINE_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/baseline.log" 2>&1) &
    BASELINE_PID=$!
    echo "  [1/8] Baseline:     PID $BASELINE_PID (GPU $BASELINE_GPU)"
    
    # 2. Edit-Only
    (CUDA_VISIBLE_DEVICES=$EDIT_ONLY_GPU python scripts/evaluation/evaluate_model.py \
        --model-type edit_only \
        --results-dir "$RESULTS_DIR" \
        --output "$EDIT_ONLY_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/edit_only.log" 2>&1) &
    EDIT_ONLY_PID=$!
    echo "  [2/8] Edit-Only:    PID $EDIT_ONLY_PID (GPU $EDIT_ONLY_GPU)"
    
    # 3. Standard
    (CUDA_VISIBLE_DEVICES=$STANDARD_GPU python scripts/evaluation/evaluate_model.py \
        --model-type standard \
        --checkpoint "$STANDARD_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$STANDARD_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/standard_text.log" 2>&1) &
    STANDARD_PID=$!
    echo "  [3/8] Standard:     PID $STANDARD_PID (GPU $STANDARD_GPU)"
    
    # 4. RL
    (CUDA_VISIBLE_DEVICES=$RL_GPU python scripts/evaluation/evaluate_model.py \
        --model-type rl \
        --checkpoint "$RL_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RL_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/rl_text.log" 2>&1) &
    RL_PID=$!
    echo "  [4/8] RL:           PID $RL_PID (GPU $RL_GPU)"
    
    # 5. RW
    (CUDA_VISIBLE_DEVICES=$RW_GPU python scripts/evaluation/evaluate_model.py \
        --model-type rw \
        --checkpoint "$RW_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RW_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/rw_text.log" 2>&1) &
    RW_PID=$!
    echo "  [5/8] RW:           PID $RW_PID (GPU $RW_GPU)"
    
    # 6. DPO
    (CUDA_VISIBLE_DEVICES=$DPO_GPU python scripts/evaluation/evaluate_model.py \
        --model-type dpo \
        --checkpoint "$DPO_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$DPO_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/dpo_text.log" 2>&1) &
    DPO_PID=$!
    echo "  [6/8] DPO:          PID $DPO_PID (GPU $DPO_GPU)"
    
    # 7. SW
    (CUDA_VISIBLE_DEVICES=$SW_GPU python scripts/evaluation/evaluate_model.py \
        --model-type sw \
        --checkpoint "$SW_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$SW_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --gpu 0 \
        --save-images \
        >> "$LOG_DIR/sw_text.log" 2>&1) &
    SW_PID=$!
    echo "  [7/8] SW:           PID $SW_PID (GPU $SW_GPU)"
    
    # 8. GPT-4o
    (python scripts/evaluation/evaluate_model.py \
        --model-type gpt4o \
        --data "$DATA_PATH" \
        --output "$GPT4O_DIR" \
        --sample-ids-file "$BATCH_IDS_FILE" \
        --editor-type "$EDITOR_TYPE" \
        --action-library "$ACTION_LIBRARY" \
        --gpu none \
        --save-images \
        >> "$LOG_DIR/gpt4o.log" 2>&1) &
    GPT4O_PID=$!
    echo "  [8/8] GPT-4o:       PID $GPT4O_PID (API)"
    
    echo ""
    echo "⏳ Waiting for all 8 models to complete..."
    
    # Wait for all processes
    wait $BASELINE_PID
    echo "  ✅ Baseline complete"
    wait $EDIT_ONLY_PID
    echo "  ✅ Edit-Only complete"
    wait $STANDARD_PID
    echo "  ✅ Standard complete"
    wait $RL_PID
    echo "  ✅ RL complete"
    wait $RW_PID
    echo "  ✅ RW complete"
    wait $DPO_PID
    echo "  ✅ DPO complete"
    wait $SW_PID
    echo "  ✅ SW complete"
    wait $GPT4O_PID
    echo "  ✅ GPT-4o complete"
    
    echo ""
    echo "✅ Batch $batch_num/$NUM_BATCHES complete"
    echo ""
    
    # Generate checkpoint tables if not the last batch
    if [ $batch_num -lt $NUM_BATCHES ] && [ $CHECKPOINT_INTERVAL -gt 0 ]; then
        SAMPLES_SO_FAR=$(( $END_IDX + 1 ))
        CHECKPOINT_NUM=$(printf "%03d" $SAMPLES_SO_FAR)
        CHECKPOINT_DIR="$CONSOLIDATED_DIR/checkpoints/checkpoint_$CHECKPOINT_NUM"
        mkdir -p "$CHECKPOINT_DIR"
        
        echo "📊 Generating checkpoint tables ($CHECKPOINT_NUM samples)..."
        
        # Consolidate results
        python3 << PYEOF
import json
from pathlib import Path

consolidated_dir = Path("$CONSOLIDATED_DIR")
checkpoint_dir = Path("$CHECKPOINT_DIR")
consolidated_dir.mkdir(parents=True, exist_ok=True)

# Model directories
model_dirs = {
    "baseline": Path("$BASELINE_DIR"),
    "edit_only": Path("$EDIT_ONLY_DIR"),
    "standard_text": Path("$STANDARD_DIR"),
    "rl_text": Path("$RL_DIR"),
    "rw_text": Path("$RW_DIR"),
    "dpo_text": Path("$DPO_DIR"),
    "sw_text": Path("$SW_DIR"),
    "gpt4o": Path("$GPT4O_DIR")
}

# Model display names
model_labels = {
    "baseline": "Baseline",
    "edit_only": "Edit-Only",
    "standard_text": "Standard",
    "rl_text": "RL",
    "rw_text": "RW",
    "dpo_text": "DPO",
    "sw_text": "SW",
    "gpt4o": "GPT-4o"
}

# Collect summaries
consolidated = {}
labels = []
for model_name, model_dir in model_dirs.items():
    summary_file = model_dir / "evaluation_summary_all.json"
    if summary_file.exists():
        with open(summary_file) as f:
            consolidated[model_name] = json.load(f)
        labels.append(model_labels[model_name])

# Add labels array for final_metric_summary.py
consolidated["labels"] = labels

# Save consolidated summary
with open(checkpoint_dir / "consolidated_summary.json", 'w') as f:
    json.dump(consolidated, f, indent=2)

print(f"✓ Saved checkpoint summary to {checkpoint_dir}")
PYEOF
        
        # Generate tables
        python scripts/evaluation/generate_comparison_tables.py \
            --results "$CHECKPOINT_DIR/consolidated_summary.json" \
            --output-dir "$CHECKPOINT_DIR"
        
        echo "  ✓ Tables saved to $CHECKPOINT_DIR"
        echo ""
    fi
done

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: FINAL CONSOLIDATION & TABLES
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [3/6] Generating final tables"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$CONSOLIDATED_DIR"

# Consolidate all results
python3 << PYEOF
import json
from pathlib import Path

consolidated_dir = Path("$CONSOLIDATED_DIR")

model_dirs = {
    "baseline": Path("$BASELINE_DIR"),
    "edit_only": Path("$EDIT_ONLY_DIR"),
    "standard_text": Path("$STANDARD_DIR"),
    "rl_text": Path("$RL_DIR"),
    "rw_text": Path("$RW_DIR"),
    "dpo_text": Path("$DPO_DIR"),
    "sw_text": Path("$SW_DIR"),
    "gpt4o": Path("$GPT4O_DIR")
}

# Model display names
model_labels = {
    "baseline": "Baseline",
    "edit_only": "Edit-Only",
    "standard_text": "Standard",
    "rl_text": "RL",
    "rw_text": "RW",
    "dpo_text": "DPO",
    "sw_text": "SW",
    "gpt4o": "GPT-4o"
}

consolidated = {}
labels = []
for model_name, model_dir in model_dirs.items():
    summary_file = model_dir / "evaluation_summary_all.json"
    if summary_file.exists():
        with open(summary_file) as f:
            consolidated[model_name] = json.load(f)
        labels.append(model_labels[model_name])

# Add labels array for final_metric_summary.py
consolidated["labels"] = labels

with open(consolidated_dir / "consolidated_summary.json", 'w') as f:
    json.dump(consolidated, f, indent=2)

print(f"✓ Consolidated {len(consolidated) - 1} models")  # -1 for labels
PYEOF

# Generate final tables
echo "📊 Generating final comparison tables..."

python scripts/evaluation/generate_comparison_tables.py \
    --results "$CONSOLIDATED_DIR/consolidated_summary.json" \
    --output-dir "$CONSOLIDATED_DIR"

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: PER-SAMPLE CONSOLIDATION
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [4/6] Consolidating per-sample results"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python scripts/evaluation/consolidate_per_sample_results.py \
    --baseline-dir "$BASELINE_DIR" \
    --edit-only-dir "$EDIT_ONLY_DIR" \
    --standard-text-dir "$STANDARD_DIR" \
    --rl-text-dir "$RL_DIR" \
    --rw-text-dir "$RW_DIR" \
    --dpo-text-dir "$DPO_DIR" \
    --sw-text-dir "$SW_DIR" \
    --gpt4o-dir "$GPT4O_DIR" \
    --output-dir "$CONSOLIDATED_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to consolidate per-sample results"
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5: CREATE CONSOLIDATED_DETAILED.JSON
# ════════════════════════════════════════════════════════════════════════════════
# Note: This is already handled by consolidate_per_sample_results.py in Step 4
# This step is documented here for clarity but no action is needed

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6: GENERATE FINAL_SUMMARY.MD
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [6/6] Generating final summary report"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python scripts/evaluation/final_metric_summary.py \
    --results "$CONSOLIDATED_DIR/consolidated_summary.json" \
    --output "$CONSOLIDATED_DIR/FINAL_SUMMARY.md"

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Failed to generate summary report (non-critical)"
else
    echo "✅ Summary report generated"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✅ EVALUATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Tables:"
echo "  • $CONSOLIDATED_DIR/image_metrics_table.png"
echo "  • $CONSOLIDATED_DIR/planner_metrics_table.png"
echo "  • $CONSOLIDATED_DIR/gpt4o_action_judge_table.png"
echo "  • $CONSOLIDATED_DIR/gpt4o_image_quality_table.png"
echo ""
echo "Per-Sample Results:"
echo "  • $CONSOLIDATED_DIR/samples/ (per-sample folders with comparison images)"
echo "  • $CONSOLIDATED_DIR/consolidated_detailed.json"
echo "  • $CONSOLIDATED_DIR/FINAL_SUMMARY.md"
echo ""

