#!/bin/bash
# Parallel evaluation for TRAJECTORY-BASED 4B baseline, standard, RL, RW, DPO, SW, and GPT-4o text-only models - 4B
# TEXT-ONLY TRAJECTORY EXPERIMENT (4B): Runs 8 models in PARALLEL (7 on GPUs + 1 via API)
# Ensures ALL models evaluate the SAME samples for fair comparison

set -e  # Exit on error

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - 4B MODEL
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
RESULTS_DIR="$PROJECT_ROOT/imageagent_results_normal_cot_test"
DATA_PATH="$PROJECT_ROOT/training_data/normal/cot_4b_trajectory/full_dataset_for_eval.json"
TEST_SAMPLES_FILE="$PROJECT_ROOT/training_data/normal/cot_4b_trajectory/test_samples_cot_4b.txt"
OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/normal/text_parallel_cot_4b_trajectory"
NUM_SAMPLES=4  # Evaluate on low-quality test samples (trajectory-based split)
EVAL_CHECKPOINT=2  # Generate tables after every N samples (0 = only at end)

# Output directories for each model
BASELINE_DIR="$OUTPUT_BASE/baseline"
EDIT_ONLY_DIR="$OUTPUT_BASE/edit_only"
STANDARD_TEXT_DIR="$OUTPUT_BASE/standard_text"
RL_TEXT_DIR="$OUTPUT_BASE/rl_text"
RW_TEXT_DIR="$OUTPUT_BASE/rw_text"
DPO_TEXT_DIR="$OUTPUT_BASE/dpo_text"
SW_TEXT_DIR="$OUTPUT_BASE/sw_text"
GPT4O_DIR="$OUTPUT_BASE/gpt4o"
CONSOLIDATED_DIR="$OUTPUT_BASE/consolidated_text"

# GPU assignments
BASELINE_GPU=7
EDIT_ONLY_GPU=1
STANDARD_TEXT_GPU=2
RL_TEXT_GPU=3
RW_TEXT_GPU=4
DPO_TEXT_GPU=5
SW_TEXT_GPU=6

# Model checkpoints - TRAJECTORY-BASED 4B
BASELINE_CHECKPOINT="Qwen/Qwen3-VL-4B-Instruct"
EDIT_ONLY_CHECKPOINT="none"  # E baseline is inference-only, no checkpoint
STANDARD_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/normal/cot_4b_trajectory/standard/final"
RL_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/normal/cot_4b_trajectory/rl/final"
RW_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/normal/cot_4b_trajectory/rw/final"
DPO_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/normal/cot_4b_trajectory/dpo/final"
SW_TEXT_CHECKPOINT="$PROJECT_ROOT/checkpoints/normal/cot_4b_trajectory/sw/final"

# Image editor configuration
EDITOR_TYPE="qwen"  # "qwen" (Qwen-Image-Edit) or "hidream" (HiDream-E1)
HIDREAM_CHECKPOINT="$PROJECT_ROOT/HiDream-E1"  # Only used if EDITOR_TYPE=hidream
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"

# GPT-4o Action Judge flag (NEW)
USE_GPT_ACTION_JUDGE=""
export USE_GPT_ACTION_JUDGE  # Export so it's visible in subshells

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            BASELINE_DIR="$OUTPUT_BASE/baseline"
            EDIT_ONLY_DIR="$OUTPUT_BASE/edit_only"
            STANDARD_TEXT_DIR="$OUTPUT_BASE/standard_text"
            RL_TEXT_DIR="$OUTPUT_BASE/rl_text"
            RW_TEXT_DIR="$OUTPUT_BASE/rw_text"
            DPO_TEXT_DIR="$OUTPUT_BASE/dpo_text"
            SW_TEXT_DIR="$OUTPUT_BASE/sw_text"
            CONSOLIDATED_DIR="$OUTPUT_BASE/consolidated_text"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --eval-checkpoint)
            EVAL_CHECKPOINT="$2"
            shift 2
            ;;
        --use-gpt-judge-action)
            USE_GPT_ACTION_JUDGE="--use-gpt-judge-action"
            shift
            ;;
        --baseline-gpu)
            BASELINE_GPU="$2"
            shift 2
            ;;
        --edit-only-gpu)
            EDIT_ONLY_GPU="$2"
            shift 2
            ;;
        --standard-text-gpu)
            STANDARD_TEXT_GPU="$2"
            shift 2
            ;;
        --rl-text-gpu)
            RL_TEXT_GPU="$2"
            shift 2
            ;;
        --rw-text-gpu)
            RW_TEXT_GPU="$2"
            shift 2
            ;;
        --dpo-text-gpu)
            DPO_TEXT_GPU="$2"
            shift 2
            ;;
        --sw-text-gpu)
            SW_TEXT_GPU="$2"
            shift 2
            ;;
        --baseline-checkpoint)
            BASELINE_CHECKPOINT="$2"
            shift 2
            ;;
        --standard-text-checkpoint)
            STANDARD_TEXT_CHECKPOINT="$2"
            shift 2
            ;;
        --rl-text-checkpoint)
            RL_TEXT_CHECKPOINT="$2"
            shift 2
            ;;
        --rw-text-checkpoint)
            RW_TEXT_CHECKPOINT="$2"
            shift 2
            ;;
        --dpo-text-checkpoint)
            DPO_TEXT_CHECKPOINT="$2"
            shift 2
            ;;
        --sw-text-checkpoint)
            SW_TEXT_CHECKPOINT="$2"
            shift 2
            ;;
        --hidream-checkpoint)
            HIDREAM_CHECKPOINT="$2"
            shift 2
            ;;
        --hidream-config)
            HIDREAM_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: bash scripts/evaluation/start_eval_all_planner_text_trajectory_4b.sh [OPTIONS]

TRAJECTORY-BASED 4B TEXT-ONLY EXPERIMENT (PARALLEL): Evaluates baseline, edit-only, standard, RL, RW, DPO, and SW 
text-only models trained with trajectory-based approach on the SAME samples using 7 GPUs + 1 API-based model (GPT-4o) in parallel.

Options:
  --data PATH                   Path to evaluation data (default: training_data/normal/cot_4b_trajectory/full_dataset_for_eval.json)
  --output PATH                 Output base directory (default: evaluation_results/normal/text_parallel_cot_4b_trajectory)
  --num-samples N               Number of samples to evaluate (default: from test set)
  --eval-checkpoint N           Generate tables after every N samples (default: 0 = only at end)
  --baseline-gpu ID             GPU for baseline (default: 0)
  --edit-only-gpu ID            GPU for edit-only (default: 1)
  --standard-text-gpu ID        GPU for standard text (default: 2)
  --rl-text-gpu ID              GPU for RL text (default: 3)
  --rw-text-gpu ID              GPU for RW text (default: 4)
  --dpo-text-gpu ID             GPU for DPO text (default: 5)
  --sw-text-gpu ID              GPU for SW text (default: 6)
  --baseline-checkpoint P       Baseline checkpoint (default: Qwen/Qwen3-VL-4B-Instruct)
  --standard-text-checkpoint P  Standard Text checkpoint (default: checkpoints/normal/cot_4b_trajectory/standard/final)
  --rl-text-checkpoint P        RL Text checkpoint (default: checkpoints/normal/cot_4b_trajectory/rl/final)
  --rw-text-checkpoint P        RW Text checkpoint (default: checkpoints/normal/cot_4b_trajectory/rw/final)
  --dpo-text-checkpoint P       DPO Text checkpoint (default: checkpoints/normal/cot_4b_trajectory/dpo/final)
  --sw-text-checkpoint P        SW Text checkpoint (default: checkpoints/normal/cot_4b_trajectory/sw/final)
  --hidream-checkpoint P        HiDream checkpoint for image generation (default: HiDream-E1)
  --hidream-config P            HiDream config file (default: training/config/training_config.yaml)
  -h, --help                    Show this help message

Examples:
  # Evaluate test samples on GPUs 0-6 (parallel trajectory experiment)
  bash scripts/evaluation/start_eval_all_planner_text_trajectory_4b.sh

  # Evaluate custom number of samples on GPUs 3-9
  bash scripts/evaluation/start_eval_all_planner_text_trajectory_4b.sh --num-samples 100 --baseline-gpu 3 --edit-only-gpu 4 --standard-text-gpu 5 --rl-text-gpu 6 --rw-text-gpu 7 --dpo-text-gpu 8 --sw-text-gpu 9
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

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  PARALLEL TRAJECTORY-BASED 4B TEXT-ONLY MODEL EVALUATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Data:        $DATA_PATH"
echo "  Output:      $OUTPUT_BASE"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Eval Checkpoint: $EVAL_CHECKPOINT"
if [ $EVAL_CHECKPOINT -gt 0 ]; then
    NUM_BATCHES=$(( ($NUM_SAMPLES + $EVAL_CHECKPOINT - 1) / $EVAL_CHECKPOINT ))
    echo "  Batching:    Yes ($NUM_BATCHES batches of $EVAL_CHECKPOINT samples)"
else
    echo "  Batching:    No (tables generated only at end)"
fi
echo ""
echo "GPU Assignments:"
echo "  Baseline:         GPU $BASELINE_GPU"
echo "  Edit-Only:        GPU $EDIT_ONLY_GPU"
echo "  Standard Text:    GPU $STANDARD_TEXT_GPU"
echo "  RL Text:          GPU $RL_TEXT_GPU"
echo "  RW Text:          GPU $RW_TEXT_GPU"
echo "  DPO Text:         GPU $DPO_TEXT_GPU"
echo "  SW Text:          GPU $SW_TEXT_GPU"
echo ""
echo "Checkpoints (TRAJECTORY-BASED 4B):"
echo "  Baseline:         $BASELINE_CHECKPOINT"
echo "  Edit-Only:        $EDIT_ONLY_CHECKPOINT"
echo "  Standard Text:    $STANDARD_TEXT_CHECKPOINT"
echo "  RL Text:          $RL_TEXT_CHECKPOINT"
echo "  RW Text:          $RW_TEXT_CHECKPOINT"
echo "  DPO Text:         $DPO_TEXT_CHECKPOINT"
echo "  SW Text:          $SW_TEXT_CHECKPOINT"
echo "  HiDream:          $HIDREAM_CHECKPOINT"
echo "  HiDream Cfg:      $HIDREAM_CONFIG"
echo ""
echo "Models to evaluate: 8 (Baseline, Edit-Only, Standard, RL, RW, DPO, SW, GPT-4o Planner)"
echo "Execution mode: PARALLEL (7× faster than sequential)"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: PRE-SELECT SAMPLE IDs
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [1/3] Preparing evaluation data (trajectory-based)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

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
    "version": "3.0-trajectory",
    "description": "Full dataset for trajectory-based evaluation",
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

# Check if test samples file exists
if [ ! -f "$TEST_SAMPLES_FILE" ]; then
    echo "❌ Error: Test samples file not found: $TEST_SAMPLES_FILE"
    echo "   Please run the trajectory pipeline to create the train/test split first."
    exit 1
fi

# Copy test samples to evaluation directory
cp "$TEST_SAMPLES_FILE" "$OUTPUT_BASE/selected_sample_ids.txt"

# Count and display
NUM_TEST_SAMPLES=$(wc -l < "$TEST_SAMPLES_FILE")
echo "✅ Using $NUM_TEST_SAMPLES trajectory-based test samples"
echo ""
echo "Sample IDs (all):"
cat "$TEST_SAMPLES_FILE" | nl
echo ""


echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: LAUNCH PARALLEL EVALUATIONS (WITH OPTIONAL BATCHING)
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [2/3] Launching parallel evaluations"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Create log directory
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

# Activate conda environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1; then
    conda activate img-agent
    echo "✅ Activated img-agent environment"
else
    echo "⚠️  Warning: conda not found, using system Python"
fi

echo ""

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
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
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
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
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
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
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
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
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
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
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
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
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
        echo ""
        
        # Wait for all processes in this batch
        wait $BASELINE_PID
        BASELINE_EXIT=$?
        
        wait $EDIT_ONLY_PID
        EDIT_ONLY_EXIT=$?
        
        wait $STANDARD_TEXT_PID
        STANDARD_TEXT_EXIT=$?
        
        wait $RL_TEXT_PID
        RL_TEXT_EXIT=$?
        
        wait $RW_TEXT_PID
        RW_TEXT_EXIT=$?
        
        wait $DPO_TEXT_PID
        DPO_TEXT_EXIT=$?
        
        wait $SW_TEXT_PID
        SW_TEXT_EXIT=$?
        
        wait $GPT4O_PID
        GPT4O_EXIT=$?
        
        # Check batch results
        ALL_SUCCESS=true
        if [ $BASELINE_EXIT -ne 0 ]; then
            echo "❌ Baseline batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/baseline.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $EDIT_ONLY_EXIT -ne 0 ]; then
            echo "❌ Edit-Only batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/edit_only.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $STANDARD_TEXT_EXIT -ne 0 ]; then
            echo "❌ Standard Text batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/standard_text.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $RL_TEXT_EXIT -ne 0 ]; then
            echo "❌ RL Text batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/rl_text.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $RW_TEXT_EXIT -ne 0 ]; then
            echo "❌ RW Text batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/rw_text.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $DPO_TEXT_EXIT -ne 0 ]; then
            echo "❌ DPO Text batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/dpo_text.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $SW_TEXT_EXIT -ne 0 ]; then
            echo "❌ SW Text batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/sw_text.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        if [ $GPT4O_EXIT -ne 0 ]; then
            echo "❌ GPT-4o Planner batch $BATCH_NUM: FAILED"
            echo ""
            echo "🔴 ERROR LOG (last 20 lines):"
            tail -20 "$LOG_DIR/gpt4o.log" 2>/dev/null || echo "Log file not found"
            echo ""
            ALL_SUCCESS=false
        fi
        
        if [ "$ALL_SUCCESS" = false ]; then
            echo ""
            echo "═════════════════════════════════════════════════════════════════════"
            echo "❌ BATCH $BATCH_NUM FAILED - ERROR LOGS SHOWN ABOVE"
            echo "═════════════════════════════════════════════════════════════════════"
            echo "Full logs available in: $LOG_DIR/"
            echo ""
            exit 1
        fi
        
        echo "✅ Batch $BATCH_NUM complete ($BATCH_END/$TOTAL_SAMPLES samples evaluated)"
        echo ""
        
        # Generate tables after this batch
        CUMULATIVE_SAMPLES=$BATCH_END
        echo "📊 Generating tables for cumulative $CUMULATIVE_SAMPLES samples..."
        
        bash "$PROJECT_ROOT/scripts/evaluation/complex_theme/consolidate_text_results.sh" \
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
            if cp "$CONSOLIDATED_DIR"/*.png "$CONSOLIDATED_DIR"/*.json "$CONSOLIDATED_DIR"/*.md "$CHECKPOINT_DIR/" 2>&1; then
                echo "   Checkpoint saved: $CHECKPOINT_DIR"
            else
                echo "   ⚠️  Warning: Could not copy all files to checkpoint"
            fi
        else
            echo "⚠️  Table generation failed (continuing evaluation)"
        fi
        
        echo ""
        
        CURRENT_SAMPLE=$BATCH_END
    done
    
    echo "✅ All batches complete!"
    
else
    # NON-BATCHED EVALUATION (original behavior)
    echo "🚀 Starting parallel evaluations (single batch)..."
    echo ""

    # Launch Baseline (GPU 0) in background
    echo "  [GPU $BASELINE_GPU] Baseline model..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$BASELINE_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$BASELINE_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$BASELINE_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --save-images \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-predictions \
        > "$LOG_DIR/baseline.log" 2>&1) &
    BASELINE_PID=$!
    
    # Launch Edit-Only in background
    echo "  [GPU $EDIT_ONLY_GPU] Edit-Only baseline..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$EDIT_ONLY_GPU python scripts/evaluate_edit_only.py \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --results-dir "$RESULTS_DIR" \
        --output "$EDIT_ONLY_DIR" \
        > "$LOG_DIR/edit_only.log" 2>&1) &
    EDIT_ONLY_PID=$!
    
    # Launch Standard Text (GPU 1) in background
    echo "  [GPU $STANDARD_TEXT_GPU] Standard Text model (trajectory)..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$STANDARD_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$STANDARD_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$STANDARD_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --save-images \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-predictions \
        > "$LOG_DIR/standard_text.log" 2>&1) &
    STANDARD_TEXT_PID=$!
    
    # Launch RL Text (GPU 2) in background
    echo "  [GPU $RL_TEXT_GPU] RL Text model (trajectory)..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RL_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$RL_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RL_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --save-images \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-predictions \
        > "$LOG_DIR/rl_text.log" 2>&1) &
    RL_TEXT_PID=$!
    
    # Launch RW Text (GPU 3) in background
    echo "  [GPU $RW_TEXT_GPU] RW Text model (trajectory)..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RW_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$RW_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RW_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --save-images \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-predictions \
        > "$LOG_DIR/rw_text.log" 2>&1) &
    RW_TEXT_PID=$!
    
    # Launch DPO Text (GPU 4) in background
    echo "  [GPU $DPO_TEXT_GPU] DPO Text model (trajectory)..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$DPO_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$DPO_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$DPO_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --save-images \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-predictions \
        > "$LOG_DIR/dpo_text.log" 2>&1) &
    DPO_TEXT_PID=$!
    
    # Launch SW Text in background
    echo "  [GPU $SW_TEXT_GPU] SW Text model (trajectory)..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$SW_TEXT_GPU python scripts/evaluate_planner_batch.py \
        --checkpoint "$SW_TEXT_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$SW_TEXT_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --save-images \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --save-predictions \
        > "$LOG_DIR/sw_text.log" 2>&1) &
    SW_TEXT_PID=$!
    
    echo ""
    echo "✅ All 8 evaluations launched in parallel (7 on GPUs + GPT-4o via API)"
    echo ""
    echo "Process IDs:"
    echo "  Baseline:         PID $BASELINE_PID (GPU $BASELINE_GPU)"
    echo "  Edit-Only:        PID $EDIT_ONLY_PID (GPU $EDIT_ONLY_GPU)"
    echo "  Standard Text:    PID $STANDARD_TEXT_PID (GPU $STANDARD_TEXT_GPU)"
    echo "  RL Text:          PID $RL_TEXT_PID (GPU $RL_TEXT_GPU)"
    echo "  RW Text:          PID $RW_TEXT_PID (GPU $RW_TEXT_GPU)"
    echo "  DPO Text:         PID $DPO_TEXT_PID (GPU $DPO_TEXT_GPU)"
    echo "  SW Text:          PID $SW_TEXT_PID (GPU $SW_TEXT_GPU)"
    echo ""
    echo "Logs:"
    echo "  Baseline:         tail -f $LOG_DIR/baseline.log"
    echo "  Edit-Only:        tail -f $LOG_DIR/edit_only.log"
    echo "  Standard Text:    tail -f $LOG_DIR/standard_text.log"
    echo "  RL Text:          tail -f $LOG_DIR/rl_text.log"
    echo "  RW Text:          tail -f $LOG_DIR/rw_text.log"
    echo "  DPO Text:         tail -f $LOG_DIR/dpo_text.log"
    echo "  SW Text:          tail -f $LOG_DIR/sw_text.log"
    echo ""
    echo "⏳ Waiting for all evaluations to complete..."
    echo ""
    
    # Wait for all processes and capture exit codes
    wait $BASELINE_PID
    BASELINE_EXIT=$?
    
    wait $EDIT_ONLY_PID
    EDIT_ONLY_EXIT=$?
    
    wait $STANDARD_TEXT_PID
    STANDARD_TEXT_EXIT=$?
    
    wait $RL_TEXT_PID
    RL_TEXT_EXIT=$?
    
    wait $RW_TEXT_PID
    RW_TEXT_EXIT=$?
    
    wait $DPO_TEXT_PID
    DPO_TEXT_EXIT=$?
    
    wait $SW_TEXT_PID
    SW_TEXT_EXIT=$?
    
    # Check results
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Evaluation Results"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    ALL_SUCCESS=true
    
    if [ $BASELINE_EXIT -eq 0 ]; then
        echo "✅ Baseline evaluation: SUCCESS"
    else
        echo "❌ Baseline evaluation: FAILED (exit code: $BASELINE_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $EDIT_ONLY_EXIT -eq 0 ]; then
        echo "✅ Edit-Only evaluation: SUCCESS"
    else
        echo "❌ Edit-Only evaluation: FAILED (exit code: $EDIT_ONLY_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $STANDARD_TEXT_EXIT -eq 0 ]; then
        echo "✅ Standard Text evaluation: SUCCESS"
    else
        echo "❌ Standard Text evaluation: FAILED (exit code: $STANDARD_TEXT_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $RL_TEXT_EXIT -eq 0 ]; then
        echo "✅ RL Text evaluation: SUCCESS"
    else
        echo "❌ RL Text evaluation: FAILED (exit code: $RL_TEXT_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $RW_TEXT_EXIT -eq 0 ]; then
        echo "✅ RW Text evaluation: SUCCESS"
    else
        echo "❌ RW Text evaluation: FAILED (exit code: $RW_TEXT_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $DPO_TEXT_EXIT -eq 0 ]; then
        echo "✅ DPO Text evaluation: SUCCESS"
    else
        echo "❌ DPO Text evaluation: FAILED (exit code: $DPO_TEXT_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $SW_TEXT_EXIT -eq 0 ]; then
        echo "✅ SW Text evaluation: SUCCESS"
    else
        echo "❌ SW Text evaluation: FAILED (exit code: $SW_TEXT_EXIT)"
        ALL_SUCCESS=false
    fi

echo ""

if [ "$ALL_SUCCESS" = false ]; then
    echo "❌ Some evaluations failed. Check logs in $LOG_DIR/"
    exit 1
fi

# Close batching if/else block
fi

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: CONSOLIDATE RESULTS
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [3/3] Consolidating results into unified comparison"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

bash "$PROJECT_ROOT/scripts/evaluation/complex_theme/consolidate_text_results.sh" \
    --baseline-dir "$BASELINE_DIR" \
    --edit-only-dir "$EDIT_ONLY_DIR" \
    --standard-text-dir "$STANDARD_TEXT_DIR" \
    --rl-text-dir "$RL_TEXT_DIR" \
    --rw-text-dir "$RW_TEXT_DIR" \
    --dpo-text-dir "$DPO_TEXT_DIR" \
    --sw-text-dir "$SW_TEXT_DIR" \
            --gpt4o-dir "$GPT4O_DIR" \
    --output-dir "$CONSOLIDATED_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Consolidation failed"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✅ ALL TRAJECTORY-BASED 4B EVALUATIONS COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Results:"
echo "   Baseline:         $BASELINE_DIR"
echo "   Edit-Only:        $EDIT_ONLY_DIR"
echo "   Standard Text:    $STANDARD_TEXT_DIR"
echo "   RL Text:          $RL_TEXT_DIR"
echo "   RW Text:          $RW_TEXT_DIR"
echo "   DPO Text:         $DPO_TEXT_DIR"
echo "   SW Text:          $SW_TEXT_DIR"
echo "   Consolidated:     $CONSOLIDATED_DIR"
echo ""
echo "📊 View consolidated results:"
echo "   Summary:          cat $CONSOLIDATED_DIR/consolidated_summary.json"
echo "   Detailed:         cat $CONSOLIDATED_DIR/consolidated_detailed.json"
echo "   Markdown Report:  cat $CONSOLIDATED_DIR/FINAL_SUMMARY.md"
echo "   9-way images:     ls $CONSOLIDATED_DIR/samples/*/comparison_9way.png"
echo ""
echo "Models evaluated (TRAJECTORY-BASED 4B):"
echo "   1. Baseline (Qwen3-VL-4B-Instruct zero-shot)"
echo "   2. Edit-Only (Direct Qwen-Image-Edit, no planning)"
echo "   3. Standard Text-Only (trajectory-sampled training)"
echo "   4. RL Text-Only (trajectory-filtered training >= 3.0)"
echo "   5. RW Text-Only (trajectory-weighted training >= 3.5)"
echo "   6. DPO Text-Only (trajectory-preference pairs)"
echo "   7. SW Text-Only (trajectory z-score weighted training)"
echo ""
echo "Execution time saved: ~85% (parallel vs sequential)"
echo ""

