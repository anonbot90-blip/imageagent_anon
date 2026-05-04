#!/bin/bash
# Parallel evaluation for Complex V2 8B text-only models
# Runs 7 models in parallel on 7 different GPUs (B, E, S, R, RW, D, SW)

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
RESULTS_DIR="$PROJECT_ROOT/imageagent_results_complex_v2_10k_cot_10k_cot"
DATA_PATH="$PROJECT_ROOT/training_data/complex_v2_cot_8b_trajectory/full_dataset_for_eval.json"
TEST_SAMPLES_FILE="$PROJECT_ROOT/training_data/complex/cot_8b_trajectory/test_samples_cot_8b.txt"
OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/complex/text_parallel_cot_8b_trajectory"
NUM_SAMPLES=4
EVAL_CHECKPOINT=2

BASELINE_DIR="$OUTPUT_BASE/baseline"
EDIT_ONLY_DIR="$OUTPUT_BASE/edit_only"
STANDARD_TEXT_DIR="$OUTPUT_BASE/standard_text"
RL_TEXT_DIR="$OUTPUT_BASE/rl_text"
RW_TEXT_DIR="$OUTPUT_BASE/rw_text"
DPO_TEXT_DIR="$OUTPUT_BASE/dpo_text"
SW_TEXT_DIR="$OUTPUT_BASE/sw_text"
GPT4O_DIR="$OUTPUT_BASE/gpt4o"
CONSOLIDATED_DIR="$OUTPUT_BASE/consolidated_text"

BASELINE_GPU=7
EDIT_ONLY_GPU=1
STANDARD_TEXT_GPU=2
RL_TEXT_GPU=3
RW_TEXT_GPU=4
DPO_TEXT_GPU=5
SW_TEXT_GPU=6

BASELINE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"
STANDARD_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex/cot_8b_trajectory/standard/final"
RL_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex/cot_8b_trajectory/rl/final"
RW_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex/cot_8b_trajectory/rw/final"
DPO_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex/cot_8b_trajectory/dpo/final"
SW_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex/cot_8b_trajectory/sw/final"

EDITOR_TYPE="qwen"
HIDREAM_CHECKPOINT="$PROJECT_ROOT/HiDream-E1"
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"
USE_GPT_ACTION_JUDGE=""
export USE_GPT_ACTION_JUDGE

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --eval-checkpoint) EVAL_CHECKPOINT="$2"; shift 2 ;;
        --use-gpt-judge-action) USE_GPT_ACTION_JUDGE="--use-gpt-judge-action"; shift ;;
        --baseline-gpu) BASELINE_GPU="$2"; shift 2 ;;
        --edit-only-gpu) EDIT_ONLY_GPU="$2"; shift 2 ;;
        --standard-text-gpu) STANDARD_TEXT_GPU="$2"; shift 2 ;;
        --rl-text-gpu) RL_TEXT_GPU="$2"; shift 2 ;;
        --rw-text-gpu) RW_TEXT_GPU="$2"; shift 2 ;;
        --dpo-text-gpu) DPO_TEXT_GPU="$2"; shift 2 ;;
        --sw-text-gpu) SW_TEXT_GPU="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  TEXT PARALLEL EVALUATION - COMPLEX V2 8B MODELS (TRAJECTORY-BASED)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Evaluating 7 models in parallel (Complex V2 - 8B):"
echo "  [GPU $BASELINE_GPU] Baseline (Qwen3-VL-8B-Instruct)"
echo "  [GPU $EDIT_ONLY_GPU] Edit-Only (Direct Editing)"
echo "  [GPU $STANDARD_TEXT_GPU] Standard Text"
echo "  [GPU $RL_TEXT_GPU] RL Text"
echo "  [GPU $RW_TEXT_GPU] RW Text"
echo "  [GPU $DPO_TEXT_GPU] DPO Text"
echo "  [GPU $SW_TEXT_GPU] SW Text"
echo ""
echo "Test samples: $NUM_SAMPLES (from trajectory-based split)"
echo "Batch evaluation: Every $EVAL_CHECKPOINT samples"
echo ""

mkdir -p "$BASELINE_DIR" "$EDIT_ONLY_DIR" "$STANDARD_TEXT_DIR" "$RL_TEXT_DIR" "$RW_TEXT_DIR" "$DPO_TEXT_DIR" "$SW_TEXT_DIR" "$CONSOLIDATED_DIR"
mkdir -p "$OUTPUT_BASE/logs"

cd "$PROJECT_ROOT"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate img-agent

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: CREATE FULL EVALUATION DATASET
# ════════════════════════════════════════════════════════════════════════════════

# Create full evaluation dataset if it doesn't exist
if [ ! -f "$DATA_PATH" ]; then
    echo "📦 Creating full evaluation dataset from $RESULTS_DIR..."
    echo ""
    
python3 << EOF
import json
from pathlib import Path
from tqdm import tqdm

# Create full evaluation dataset from results directory
results_dir = Path("$RESULTS_DIR")
output_file = Path("$DATA_PATH")

print("Creating full evaluation dataset from all samples...")
samples = []

subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
print(f"Found {len(subdirs)} sample directories")

for subdir in tqdm(subdirs, desc="Processing"):
    action_plan_file = subdir / "action_plan.json"
    if action_plan_file.exists():
        try:
            with open(action_plan_file) as f:
                action_plan = json.load(f)
            
            # Extract overall_instruction from action_plan
            overall_instruction = action_plan.get("overall_instruction", "")
            
            sample = {
                "sample_id": subdir.name,
                "original_image_path": f"{results_dir.name}/{subdir.name}/original.png",
                "edited_image_path": f"{results_dir.name}/{subdir.name}/edited.png",
                "analysis_path": f"{results_dir.name}/{subdir.name}/analysis.json",
                "overall_instruction": overall_instruction,
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
    "version": "3.0-trajectory-v2",
    "description": "Full dataset for Complex V2 trajectory-based evaluation",
    "total_samples": len(samples),
    "rl_filtered": False,
    "samples": samples
}

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Created full evaluation dataset: {output_file}")
print(f"   Total samples: {len(samples)}")
EOF

    if [ $? -ne 0 ]; then
        echo "❌ Failed to create full evaluation dataset"
        exit 1
    fi
    echo ""
else
    echo "✅ Using existing full evaluation dataset: $DATA_PATH"
    echo ""
fi

# Select N sample IDs for evaluation
if [ ! -f "$OUTPUT_BASE/selected_sample_ids.txt" ]; then
    echo "📋 Selecting $NUM_SAMPLES random sample IDs from test set..."
    if [ -f "$TEST_SAMPLES_FILE" ]; then
        # If test samples file exists, use it
        head -n "$NUM_SAMPLES" "$TEST_SAMPLES_FILE" > "$OUTPUT_BASE/selected_sample_ids.txt"
    else
        echo "❌ ERROR: Test samples file not found: $TEST_SAMPLES_FILE"
        exit 1
    fi
    echo "✅ Selected $NUM_SAMPLES samples"
    echo ""
else
    echo "✅ Using existing sample IDs: $OUTPUT_BASE/selected_sample_ids.txt"
    echo ""
fi

LOG_DIR="$OUTPUT_BASE/logs"

# Determine batching strategy
if [ $EVAL_CHECKPOINT -gt 0 ] && [ $EVAL_CHECKPOINT -le $NUM_SAMPLES ]; then
    # BATCHED EVALUATION with incremental table generation
    echo "🔄 Batched evaluation mode: Generating tables after every $EVAL_CHECKPOINT samples"
    echo ""
    
    # Split sample IDs into batches
    TOTAL_SAMPLES=$(wc -l < "$OUTPUT_BASE/selected_sample_ids.txt")
    CURRENT_SAMPLE=0
    BATCH_NUM=0
    
    while [ $CURRENT_SAMPLE -lt $TOTAL_SAMPLES ]; do
        BATCH_NUM=$((BATCH_NUM + 1))
        BATCH_START=$((CURRENT_SAMPLE + 1))
        BATCH_END=$((CURRENT_SAMPLE + EVAL_CHECKPOINT))
        if [ $BATCH_END -gt $TOTAL_SAMPLES ]; then
            BATCH_END=$TOTAL_SAMPLES
        fi
        BATCH_SIZE=$((BATCH_END - BATCH_START + 1))
        
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "  Batch $BATCH_NUM: Samples $BATCH_START-$BATCH_END ($BATCH_SIZE samples)"
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo ""
        
        # Create batch-specific sample IDs file
        BATCH_IDS_FILE="$OUTPUT_BASE/batch_${BATCH_NUM}_sample_ids.txt"
        sed -n "${BATCH_START},${BATCH_END}p" "$OUTPUT_BASE/selected_sample_ids.txt" > "$BATCH_IDS_FILE"
        
        echo "🚀 Starting parallel evaluations for batch $BATCH_NUM..."
        echo ""
        
        # Launch all 8 models in parallel for this batch (7 on GPUs + GPT-4o via API)
        echo "  [GPU $BASELINE_GPU] Baseline model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$BASELINE_GPU python scripts/evaluate_planner_batch.py \
            --checkpoint "$BASELINE_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$BASELINE_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/baseline.log" 2>&1) &
        BASELINE_PID=$!
        
        echo "  [GPU $EDIT_ONLY_GPU] Edit-Only baseline (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$EDIT_ONLY_GPU python scripts/evaluation/evaluate_edit_only.py \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --results-dir "$RESULTS_DIR" \
            --output "$EDIT_ONLY_DIR" \
            >> "$LOG_DIR/edit_only.log" 2>&1) &
        EDIT_ONLY_PID=$!
        
        echo "  [GPU $STANDARD_TEXT_GPU] Standard Text model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$STANDARD_TEXT_GPU python scripts/evaluate_planner_batch.py \
            --checkpoint "$STANDARD_TEXT_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$STANDARD_TEXT_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/standard_text.log" 2>&1) &
        STANDARD_TEXT_PID=$!
        
        echo "  [GPU $RL_TEXT_GPU] RL Text model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RL_TEXT_GPU python scripts/evaluate_planner_batch.py \
            --checkpoint "$RL_TEXT_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$RL_TEXT_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/rl_text.log" 2>&1) &
        RL_TEXT_PID=$!
        
        echo "  [GPU $RW_TEXT_GPU] RW Text model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RW_TEXT_GPU python scripts/evaluate_planner_batch.py \
            --checkpoint "$RW_TEXT_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$RW_TEXT_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/rw_text.log" 2>&1) &
        RW_TEXT_PID=$!
        
        echo "  [GPU $DPO_TEXT_GPU] DPO Text model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$DPO_TEXT_GPU python scripts/evaluate_planner_batch.py \
            --checkpoint "$DPO_TEXT_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$DPO_TEXT_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/dpo_text.log" 2>&1) &
        DPO_TEXT_PID=$!
        
        echo "  [GPU $SW_TEXT_GPU] SW Text model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$SW_TEXT_GPU python scripts/evaluate_planner_batch.py \
            --checkpoint "$SW_TEXT_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$SW_TEXT_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/sw_text.log" 2>&1) &
        SW_TEXT_PID=$!
        
        echo "  [No GPU] GPT-4o Planner (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && python scripts/evaluate_gpt4o_planner.py \
            --data "$DATA_PATH" \
            --output "$GPT4O_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            --rate-limit-delay 0.5 \
            >> "$LOG_DIR/gpt4o.log" 2>&1) &
            # Note: GPT-4o uses API calls, no GPU required
        GPT4O_PID=$!
        
        echo ""
        echo "⏳ Waiting for batch $BATCH_NUM to complete (8 models)..."
        wait $BASELINE_PID $EDIT_ONLY_PID $STANDARD_TEXT_PID $RL_TEXT_PID $RW_TEXT_PID $DPO_TEXT_PID $SW_TEXT_PID
        echo "✅ Batch $BATCH_NUM complete!"
        echo ""
        
        # Generate consolidated table after this batch
        CUMULATIVE_SAMPLES=$BATCH_END
        echo "📊 Generating consolidated results table (after batch $BATCH_NUM)..."
        bash "$PROJECT_ROOT/scripts/evaluation/consolidate_text_results.sh" \
            --baseline-dir "$BASELINE_DIR" \
            --edit-only-dir "$EDIT_ONLY_DIR" \
            --standard-text-dir "$STANDARD_TEXT_DIR" \
            --rl-text-dir "$RL_TEXT_DIR" \
            --rw-text-dir "$RW_TEXT_DIR" \
            --dpo-text-dir "$DPO_TEXT_DIR" \
            --sw-text-dir "$SW_TEXT_DIR" \
            --gpt4o-dir "$GPT4O_DIR" \
            --output-dir "$CONSOLIDATED_DIR"
        
        if [ $? -eq 0 ]; then
            echo "✅ Tables updated (N=$CUMULATIVE_SAMPLES)"
            
            # Save checkpoint tables
            CHECKPOINT_DIR="$CONSOLIDATED_DIR/checkpoints/checkpoint_$(printf "%03d" $CUMULATIVE_SAMPLES)"
            mkdir -p "$CHECKPOINT_DIR"
            if cp "$CONSOLIDATED_DIR"/*.png "$CONSOLIDATED_DIR"/*.json "$CONSOLIDATED_DIR"/*.md "$CHECKPOINT_DIR/" 2>/dev/null; then
                echo "   📁 Checkpoint saved: checkpoint_$(printf "%03d" $CUMULATIVE_SAMPLES)/"
            else
                echo "   ⚠️  Warning: Could not copy all files to checkpoint"
            fi
        else
            echo "⚠️  Table generation failed (continuing evaluation)"
        fi
        
        echo ""
        
        CURRENT_SAMPLE=$BATCH_END
    done
    
else
    # SINGLE-SHOT EVALUATION (all samples at once)
    echo "🚀 Starting parallel evaluations (single batch)..."
    echo ""
    
    # Launch all 7 models in parallel
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$BASELINE_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$BASELINE_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$BASELINE_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/baseline.log" 2>&1) &
    BASELINE_PID=$!
    
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$EDIT_ONLY_GPU python scripts/evaluation/evaluate_edit_only.py \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --results-dir "$RESULTS_DIR" \
        --output "$EDIT_ONLY_DIR" \
        > "$LOG_DIR/edit_only.log" 2>&1) &
    EDIT_ONLY_PID=$!
    
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$STANDARD_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$STANDARD_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$STANDARD_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/standard_text.log" 2>&1) &
    STANDARD_TEXT_PID=$!
    
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RL_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$RL_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RL_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/rl_text.log" 2>&1) &
    RL_TEXT_PID=$!
    
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RW_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$RW_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RW_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/rw_text.log" 2>&1) &
    RW_TEXT_PID=$!
    
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$DPO_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$DPO_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$DPO_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/dpo_text.log" 2>&1) &
    DPO_TEXT_PID=$!
    
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$SW_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$SW_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$SW_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/sw_text.log" 2>&1) &
    SW_TEXT_PID=$!
        
        echo "  [No GPU] GPT-4o Planner (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && python scripts/evaluate_gpt4o_planner.py \
            --data "$DATA_PATH" \
            --output "$GPT4O_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --save-images \
            --save-predictions \
            --rate-limit-delay 0.5 \
            >> "$LOG_DIR/gpt4o.log" 2>&1) &
            # Note: GPT-4o uses API calls, no GPU required
        GPT4O_PID=$!
    
    echo "⏳ Waiting for all 7 models to complete..."
    wait $BASELINE_PID $EDIT_ONLY_PID $STANDARD_TEXT_PID $RL_TEXT_PID $RW_TEXT_PID $DPO_TEXT_PID $SW_TEXT_PID
    echo "✅ All evaluations complete!"
    echo ""
    
    # Generate final consolidated table
    echo "📊 Generating consolidated results table..."
    bash "$PROJECT_ROOT/scripts/evaluation/consolidate_text_results.sh" \
        --baseline-dir "$BASELINE_DIR" \
        --edit-only-dir "$EDIT_ONLY_DIR" \
        --standard-text-dir "$STANDARD_TEXT_DIR" \
        --rl-text-dir "$RL_TEXT_DIR" \
        --rw-text-dir "$RW_TEXT_DIR" \
        --dpo-text-dir "$DPO_TEXT_DIR" \
        --sw-text-dir "$SW_TEXT_DIR" \
            --gpt4o-dir "$GPT4O_DIR" \
        --output-dir "$CONSOLIDATED_DIR"
    echo "✅ Table generated"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ Evaluation Complete - Complex V2 8B Text Models"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Results: $CONSOLIDATED_DIR"
