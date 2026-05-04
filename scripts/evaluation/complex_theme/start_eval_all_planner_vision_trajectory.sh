#!/bin/bash
# Parallel evaluation for TRAJECTORY-BASED baseline, standard, RL, RW, and DPO vision models
# VISION TRAJECTORY EXPERIMENT: Runs 7 models in PARALLEL on 7 different GPUs
# Ensures ALL models evaluate the SAME samples for fair comparison

# set -e  # Exit on error

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
RESULTS_DIR="$PROJECT_ROOT/imageagent_results_normal_cot"
DATA_PATH="$PROJECT_ROOT/training_data/complex_cot_8b_trajectory/full_dataset_for_eval.json"
TEST_SAMPLES_FILE="$PROJECT_ROOT/training_data/complex_cot_8b_trajectory/test_samples_complex_cot_8b.txt"
OUTPUT_BASE="$PROJECT_ROOT/evaluation_results/vision_parallel_complex_cot_8b_trajectory"
NUM_SAMPLES=10  # Evaluate on 200 low-quality test samples
EVAL_CHECKPOINT=2  # Generate tables after every N samples (0 = only at end)

# Output directories for each model
BASELINE_DIR="$OUTPUT_BASE/baseline"
EDIT_ONLY_DIR="$OUTPUT_BASE/edit_only"
STANDARD_VISION_DIR="$OUTPUT_BASE/standard_vision"
RL_VISION_DIR="$OUTPUT_BASE/rl_vision"
RW_VISION_DIR="$OUTPUT_BASE/rw_vision"
DPO_VISION_DIR="$OUTPUT_BASE/dpo_vision"
SW_VISION_DIR="$OUTPUT_BASE/sw_vision"
CONSOLIDATED_DIR="$OUTPUT_BASE/consolidated_vision"

# GPU assignments
BASELINE_GPU=7
EDIT_ONLY_GPU=4
STANDARD_VISION_GPU=5
RL_VISION_GPU=6
RW_VISION_GPU=7
DPO_VISION_GPU=0
SW_VISION_GPU=1

# Model checkpoints
BASELINE_CHECKPOINT="Qwen/Qwen3-VL-4B-Instruct"
EDIT_ONLY_CHECKPOINT="none"  # E baseline is inference-only, no checkpoint
STANDARD_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex_cot_8b_trajectory/vision/standard/final"
RL_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex_cot_8b_trajectory/vision/rl/final"
RW_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex_cot_8b_trajectory/vision/rw/final"
DPO_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex_cot_8b_trajectory/vision/dpo/final"
SW_VISION_CHECKPOINT="$PROJECT_ROOT/checkpoints/complex_cot_8b_trajectory/vision/sw/final"

# Image editor configuration
EDITOR_TYPE="qwen"  # "qwen" (Qwen-Image-Edit) or "hidream" (HiDream-E1)

# HiDream checkpoint for end-to-end image generation
HIDREAM_CHECKPOINT="$PROJECT_ROOT/training/hidream_training/checkpoints/final"
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"

# GPT-4o Action Judge flag (NEW)
USE_GPT_JUDGE=""

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
            STANDARD_VISION_DIR="$OUTPUT_BASE/standard_vision"
            RL_VISION_DIR="$OUTPUT_BASE/rl_vision"
            RW_VISION_DIR="$OUTPUT_BASE/rw_vision"
            DPO_VISION_DIR="$OUTPUT_BASE/dpo_vision"
            CONSOLIDATED_DIR="$OUTPUT_BASE/consolidated_vision"
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
        --use-gpt-judge)
            USE_GPT_JUDGE="--use-gpt-judge"
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
        --standard-vision-gpu)
            STANDARD_VISION_GPU="$2"
            shift 2
            ;;
        --rl-vision-gpu)
            RL_VISION_GPU="$2"
            shift 2
            ;;
        --rw-vision-gpu)
            RW_VISION_GPU="$2"
            shift 2
            ;;
        --dpo-vision-gpu)
            DPO_VISION_GPU="$2"
            shift 2
            ;;
        --sw-vision-gpu)
            SW_VISION_GPU="$2"
            shift 2
            ;;
        --baseline-checkpoint)
            BASELINE_CHECKPOINT="$2"
            shift 2
            ;;
        --standard-vision-checkpoint)
            STANDARD_VISION_CHECKPOINT="$2"
            shift 2
            ;;
        --rl-vision-checkpoint)
            RL_VISION_CHECKPOINT="$2"
            shift 2
            ;;
        --rw-vision-checkpoint)
            RW_VISION_CHECKPOINT="$2"
            shift 2
            ;;
        --dpo-vision-checkpoint)
            DPO_VISION_CHECKPOINT="$2"
            shift 2
            ;;
        --sw-vision-checkpoint)
            SW_VISION_CHECKPOINT="$2"
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
Usage: bash scripts/evaluation/start_eval_all_planner_vision.sh [OPTIONS]

VISION EXPERIMENT (PARALLEL): Evaluates baseline, standard, RL, RW, and DPO vision 
models on the SAME samples using 5 GPUs in parallel.

Options:
  --data PATH                   Path to evaluation data (default: training_data/standard/planner_training_data.json)
  --output PATH                 Output base directory (default: evaluation_results/vision_parallel_eval)
  --num-samples N               Number of samples to evaluate (default: 50)
  --baseline-gpu ID             GPU for baseline (default: 3)
  --standard-vision-gpu ID      GPU for standard vision (default: 4)
  --rl-vision-gpu ID            GPU for RL vision (default: 5)
  --rw-vision-gpu ID            GPU for RW vision (default: 6)
  --dpo-vision-gpu ID           GPU for DPO vision (default: 7)
  --baseline-checkpoint P       Baseline checkpoint (default: Qwen/Qwen3-VL-4B-Instruct)
  --standard-vision-checkpoint P Standard Vision checkpoint (default: checkpoints/standard/final)
  --rl-vision-checkpoint P      RL Vision checkpoint (default: checkpoints/rl/final)
  --rw-vision-checkpoint P      RW Vision checkpoint (default: checkpoints/rw/final)
  --dpo-vision-checkpoint P     DPO Vision checkpoint (default: checkpoints/dpo/final)
  --hidream-checkpoint P        HiDream checkpoint for image generation (default: training/hidream_training/checkpoints/final)
  --hidream-config P            HiDream config file (default: training/config/training_config.yaml)
  -h, --help                    Show this help message

Examples:
  # Evaluate 50 samples on GPUs 3-7 (parallel vision experiment)
  bash scripts/evaluation/start_eval_all_planner_vision.sh

  # Evaluate 100 samples on custom GPUs
  bash scripts/evaluation/start_eval_all_planner_vision.sh --num-samples 50 --baseline-gpu 3 --standard-vision-gpu 4 --rl-vision-gpu 5

  # Use custom data path
  bash scripts/evaluation/start_eval_all_planner_vision.sh --data training_data/rl/planner_training_data.json
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
echo "  PARALLEL VISION MODEL EVALUATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Data:        $DATA_PATH"
echo "  Output:      $OUTPUT_BASE"
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
echo "  Standard Vision:  GPU $STANDARD_VISION_GPU"
echo "  RL Vision:        GPU $RL_VISION_GPU"
echo "  RW Vision:        GPU $RW_VISION_GPU"
echo "  DPO Vision:       GPU $DPO_VISION_GPU"
echo "  SW Vision:        GPU $SW_VISION_GPU"
echo ""
echo "Checkpoints:"
echo "  Baseline:         $BASELINE_CHECKPOINT"
echo "  Edit-Only:        $EDIT_ONLY_CHECKPOINT"
echo "  Standard Vision:  $STANDARD_VISION_CHECKPOINT"
echo "  RL Vision:        $RL_VISION_CHECKPOINT"
echo "  RW Vision:        $RW_VISION_CHECKPOINT"
echo "  DPO Vision:       $DPO_VISION_CHECKPOINT"
echo "  SW Vision:        $SW_VISION_CHECKPOINT"
echo "  HiDream:          $HIDREAM_CHECKPOINT"
echo "  HiDream Cfg:      $HIDREAM_CONFIG"
echo ""
echo "Models to evaluate: 7 (Baseline, Edit-Only, Standard, RL, RW, DPO, SW Vision)"
echo "Execution mode: PARALLEL (5× faster than sequential)"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: PRE-SELECT SAMPLE IDs
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  [1/3] Preparing evaluation data"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Check if test samples file exists
if [ ! -f "$TEST_SAMPLES_FILE" ]; then
    echo "❌ Error: Test samples file not found: $TEST_SAMPLES_FILE"
    echo "   Please run the text pipeline first to create the train/test split."
    exit 1
fi

# Copy test samples to evaluation directory
cp "$TEST_SAMPLES_FILE" "$OUTPUT_BASE/selected_sample_ids.txt"

# Count and display
NUM_TEST_SAMPLES=$(wc -l < "$TEST_SAMPLES_FILE")
echo "✅ Using $NUM_TEST_SAMPLES low-quality test samples (score < 2.0)"
echo ""
echo "Sample IDs (first 10):"
head -10 "$TEST_SAMPLES_FILE" | nl
if [ $NUM_TEST_SAMPLES -gt 10 ]; then
    echo "  ... and $((NUM_TEST_SAMPLES - 10)) more"
fi
echo ""

# Update NUM_SAMPLES to match actual test set size
NUM_SAMPLES=$NUM_TEST_SAMPLES

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
        
        # Launch all 7 models in parallel for this batch
        echo "  [GPU $BASELINE_GPU] Baseline model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$BASELINE_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
            --checkpoint "$BASELINE_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$BASELINE_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --hidream-config "$HIDREAM_CONFIG" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/baseline.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
        BASELINE_PID=$!
        
        echo "  [GPU $EDIT_ONLY_GPU] Edit-Only baseline (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$EDIT_ONLY_GPU $PYTHON_BIN scripts/evaluation/evaluate_edit_only.py \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --results-dir "$RESULTS_DIR" \
            --output "$EDIT_ONLY_DIR" \
            >> "$LOG_DIR/edit_only.log" 2>&1) &
        EDIT_ONLY_PID=$!
        
        echo "  [GPU $STANDARD_VISION_GPU] Standard Vision model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$STANDARD_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
            --checkpoint "$STANDARD_VISION_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$STANDARD_VISION_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --hidream-config "$HIDREAM_CONFIG" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/standard_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
        STANDARD_VISION_PID=$!
        
        echo "  [GPU $RL_VISION_GPU] RL Vision model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RL_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
            --checkpoint "$RL_VISION_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$RL_VISION_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --hidream-config "$HIDREAM_CONFIG" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/rl_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
        RL_VISION_PID=$!
        
        echo "  [GPU $RW_VISION_GPU] RW Vision model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RW_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
            --checkpoint "$RW_VISION_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$RW_VISION_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --hidream-config "$HIDREAM_CONFIG" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/rw_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
        RW_VISION_PID=$!
        
        echo "  [GPU $DPO_VISION_GPU] DPO Vision model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$DPO_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
            --checkpoint "$DPO_VISION_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$DPO_VISION_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --hidream-config "$HIDREAM_CONFIG" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/dpo_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
        DPO_VISION_PID=$!
        
        echo "  [GPU $SW_VISION_GPU] SW Vision model (batch $BATCH_NUM)..."
        (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$SW_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
            --checkpoint "$SW_VISION_CHECKPOINT" \
            --data "$DATA_PATH" \
            --output "$SW_VISION_DIR" \
            --sample-ids-file "$BATCH_IDS_FILE" \
            --split all \
            --model-editor "$EDITOR_TYPE" \
            --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
            --hidream-config "$HIDREAM_CONFIG" \
            --save-images \
            --save-predictions \
            >> "$LOG_DIR/sw_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
        SW_VISION_PID=$!
        
        echo ""
        echo "⏳ Waiting for batch $BATCH_NUM to complete..."
        echo ""
        
        # Wait for all processes in this batch
        wait $BASELINE_PID
        BASELINE_EXIT=$?
        
        wait $EDIT_ONLY_PID
        EDIT_ONLY_EXIT=$?
        
        wait $STANDARD_VISION_PID
        STANDARD_VISION_EXIT=$?
        
        wait $RL_VISION_PID
        RL_VISION_EXIT=$?
        
        wait $RW_VISION_PID
        RW_VISION_EXIT=$?
        
        wait $DPO_VISION_PID
        DPO_VISION_EXIT=$?
        
        wait $SW_VISION_PID
        SW_VISION_EXIT=$?
        
        # Check batch results
        ALL_SUCCESS=true
        if [ $BASELINE_EXIT -ne 0 ]; then
            echo "❌ Baseline batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        if [ $EDIT_ONLY_EXIT -ne 0 ]; then
            echo "❌ Edit-Only batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        if [ $STANDARD_VISION_EXIT -ne 0 ]; then
            echo "❌ Standard Vision batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        if [ $RL_VISION_EXIT -ne 0 ]; then
            echo "❌ RL Vision batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        if [ $RW_VISION_EXIT -ne 0 ]; then
            echo "❌ RW Vision batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        if [ $DPO_VISION_EXIT -ne 0 ]; then
            echo "❌ DPO Vision batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        if [ $SW_VISION_EXIT -ne 0 ]; then
            echo "❌ SW Vision batch $BATCH_NUM: FAILED"
            ALL_SUCCESS=false
        fi
        
        if [ "$ALL_SUCCESS" = false ]; then
            echo "❌ Batch $BATCH_NUM failed. Check logs in $LOG_DIR/"
            exit 1
        fi
        
        echo "✅ Batch $BATCH_NUM complete ($BATCH_END/$TOTAL_SAMPLES samples evaluated)"
        echo ""
        
        # Memory cleanup for consolidation
        echo "🧹 Cleaning up model processes..."
        kill -9 $BASELINE_PID $EDIT_ONLY_PID $STANDARD_VISION_PID $RL_VISION_PID $RW_VISION_PID $DPO_VISION_PID $SW_VISION_PID 2>/dev/null || true
        
        echo "⏳ Waiting for memory cleanup (30 seconds)..."
        sleep 30
        
        # Generate tables after this batch
        CUMULATIVE_SAMPLES=$BATCH_END
        echo "📊 Generating tables for cumulative $CUMULATIVE_SAMPLES samples..."
        
        # Export conda environment variables so they propagate to subshell
        export PATH
        export CONDA_DEFAULT_ENV
        export CONDA_PREFIX
        export CONDA_SHLVL
        export CONDA_PYTHON_EXE
        export CONDA_EXE
        
        bash "$PROJECT_ROOT/scripts/evaluation/complex_theme/consolidate_vision_results.sh" \
            --baseline-dir "$BASELINE_DIR" \
            --edit-only-dir "$EDIT_ONLY_DIR" \
            --standard-vision-dir "$STANDARD_VISION_DIR" \
            --rl-vision-dir "$RL_VISION_DIR" \
            --rw-vision-dir "$RW_VISION_DIR" \
            --dpo-vision-dir "$DPO_VISION_DIR" \
            --sw-vision-dir "$SW_VISION_DIR" \
            --output-dir "$CONSOLIDATED_DIR"
        
        if [ $? -eq 0 ]; then
            echo "✅ Tables updated (N=$CUMULATIVE_SAMPLES)"
            
            # Save checkpoint tables
            CHECKPOINT_DIR="$CONSOLIDATED_DIR/checkpoints/checkpoint_$(printf "%03d" $CUMULATIVE_SAMPLES)"
            mkdir -p "$CHECKPOINT_DIR"
            if cp "$CONSOLIDATED_DIR"/*.png "$CONSOLIDATED_DIR"/*.json "$CONSOLIDATED_DIR"/*.md "$CHECKPOINT_DIR/" 2>&1; then
                # Verify PNG files were copied
                PNG_COUNT=$(ls "$CHECKPOINT_DIR"/*.png 2>&1 | wc -l)
                echo "   Checkpoint saved: $CHECKPOINT_DIR (${PNG_COUNT} PNG files)"
            else
                echo "   ⚠️  Warning: Could not copy all files to checkpoint"
            fi
        else
            echo "⚠️  Table generation failed (check output above)"
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
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$BASELINE_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
        --checkpoint "$BASELINE_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$BASELINE_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/baseline.log" 2>&1) &
        # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
    BASELINE_PID=$!
    
    # Launch Edit-Only in background
    echo "  [GPU $EDIT_ONLY_GPU] Edit-Only baseline..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$EDIT_ONLY_GPU $PYTHON_BIN scripts/evaluate_edit_only.py \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --results-dir "$RESULTS_DIR" \
        --output "$EDIT_ONLY_DIR" \
        > "$LOG_DIR/edit_only.log" 2>&1) &
    EDIT_ONLY_PID=$!
    
    # Launch Standard Vision (GPU 1) in background
    echo "  [GPU $STANDARD_VISION_GPU] Standard Vision model..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$STANDARD_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
        --checkpoint "$STANDARD_VISION_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$STANDARD_VISION_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/standard_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
    STANDARD_VISION_PID=$!
    
    # Launch RL Vision (GPU 2) in background
    echo "  [GPU $RL_VISION_GPU] RL Vision model..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RL_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
        --checkpoint "$RL_VISION_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RL_VISION_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/rl_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
    RL_VISION_PID=$!
    
    # Launch RW Vision (GPU 3) in background
    echo "  [GPU $RW_VISION_GPU] RW Vision model..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$RW_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
        --checkpoint "$RW_VISION_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$RW_VISION_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/rw_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
    RW_VISION_PID=$!
    
    # Launch DPO Vision (GPU 4) in background
    echo "  [GPU $DPO_VISION_GPU] DPO Vision model..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$DPO_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
        --checkpoint "$DPO_VISION_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$DPO_VISION_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/dpo_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
    DPO_VISION_PID=$!
    
    # Launch SW Vision in background
    echo "  [GPU $SW_VISION_GPU] SW Vision model..."
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=$SW_VISION_GPU $PYTHON_BIN scripts/evaluate_planner_batch.py \
        --checkpoint "$SW_VISION_CHECKPOINT" \
        --data "$DATA_PATH" \
        --output "$SW_VISION_DIR" \
        --sample-ids-file "$OUTPUT_BASE/selected_sample_ids.txt" \
        --split all \
        --model-editor "$EDITOR_TYPE" \
        --hidream-checkpoint "$HIDREAM_CHECKPOINT" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        > "$LOG_DIR/sw_vision.log" 2>&1) &
            # Note: GPT-4o judge is hardcoded to always run in evaluate_planner_batch.py (line 410)
    SW_VISION_PID=$!
    
    echo ""
    echo "✅ All 7 evaluations launched in parallel"
    echo ""
    echo "Process IDs:"
    echo "  Baseline:         PID $BASELINE_PID (GPU $BASELINE_GPU)"
    echo "  Edit-Only:        PID $EDIT_ONLY_PID (GPU $EDIT_ONLY_GPU)"
    echo "  Standard Vision:    PID $STANDARD_VISION_PID (GPU $STANDARD_VISION_GPU)"
    echo "  RL Vision:          PID $RL_VISION_PID (GPU $RL_VISION_GPU)"
    echo "  RW Vision:          PID $RW_VISION_PID (GPU $RW_VISION_GPU)"
    echo "  DPO Vision:         PID $DPO_VISION_PID (GPU $DPO_VISION_GPU)"
    echo "  SW Vision:          PID $SW_VISION_PID (GPU $SW_VISION_GPU)"
    echo ""
    echo "Logs:"
    echo "  Baseline:         tail -f $LOG_DIR/baseline.log"
    echo "  Edit-Only:        tail -f $LOG_DIR/edit_only.log"
    echo "  Standard Vision:    tail -f $LOG_DIR/standard_vision.log"
    echo "  RL Vision:          tail -f $LOG_DIR/rl_vision.log"
    echo "  RW Vision:          tail -f $LOG_DIR/rw_vision.log"
    echo "  DPO Vision:         tail -f $LOG_DIR/dpo_vision.log"
    echo "  SW Vision:          tail -f $LOG_DIR/sw_vision.log"
    echo ""
    echo "⏳ Waiting for all evaluations to complete..."
    echo ""
    
    # Wait for all processes and capture exit codes
    wait $BASELINE_PID
    BASELINE_EXIT=$?
    
    wait $EDIT_ONLY_PID
    EDIT_ONLY_EXIT=$?
    
    wait $STANDARD_VISION_PID
    STANDARD_VISION_EXIT=$?
    
    wait $RL_VISION_PID
    RL_VISION_EXIT=$?
    
    wait $RW_VISION_PID
    RW_VISION_EXIT=$?
    
    wait $DPO_VISION_PID
    DPO_VISION_EXIT=$?
    
    wait $SW_VISION_PID
    SW_VISION_EXIT=$?
    
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
    
    if [ $STANDARD_VISION_EXIT -eq 0 ]; then
        echo "✅ Standard Vision evaluation: SUCCESS"
    else
        echo "❌ Standard Vision evaluation: FAILED (exit code: $STANDARD_VISION_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $RL_VISION_EXIT -eq 0 ]; then
        echo "✅ RL Vision evaluation: SUCCESS"
    else
        echo "❌ RL Vision evaluation: FAILED (exit code: $RL_VISION_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $RW_VISION_EXIT -eq 0 ]; then
        echo "✅ RW Vision evaluation: SUCCESS"
    else
        echo "❌ RW Vision evaluation: FAILED (exit code: $RW_VISION_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $DPO_VISION_EXIT -eq 0 ]; then
        echo "✅ DPO Vision evaluation: SUCCESS"
    else
        echo "❌ DPO Vision evaluation: FAILED (exit code: $DPO_VISION_EXIT)"
        ALL_SUCCESS=false
    fi
    
    if [ $SW_VISION_EXIT -eq 0 ]; then
        echo "✅ SW Vision evaluation: SUCCESS"
    else
        echo "❌ SW Vision evaluation: FAILED (exit code: $SW_VISION_EXIT)"
        ALL_SUCCESS=false
    fi
    
    echo ""
    
    if [ "$ALL_SUCCESS" = false ]; then
        echo "❌ Some evaluations failed. Check logs in $LOG_DIR/"
        exit 1
    fi
    
    # ════════════════════════════════════════════════════════════════════════════════
    # STEP 3: CONSOLIDATE RESULTS
    # ════════════════════════════════════════════════════════════════════════════════
    
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  [3/3] Consolidating results into unified comparison"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    bash "$PROJECT_ROOT/scripts/evaluation/complex_theme/consolidate_vision_results.sh" \
        --baseline-dir "$BASELINE_DIR" \
        --edit-only-dir "$EDIT_ONLY_DIR" \
        --standard-vision-dir "$STANDARD_VISION_DIR" \
        --rl-vision-dir "$RL_VISION_DIR" \
        --rw-vision-dir "$RW_VISION_DIR" \
        --dpo-vision-dir "$DPO_VISION_DIR" \
        --sw-vision-dir "$SW_VISION_DIR" \
        --output-dir "$CONSOLIDATED_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Consolidation failed"
        exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✅ ALL EVALUATIONS COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Results:"
echo "   Baseline:         $BASELINE_DIR"
echo "   Standard Vision:  $STANDARD_VISION_DIR"
echo "   RL Vision:        $RL_VISION_DIR"
echo "   RW Vision:        $RW_VISION_DIR"
echo "   DPO Vision:       $DPO_VISION_DIR"
echo "   Consolidated:     $CONSOLIDATED_DIR"
echo ""
echo "📊 View consolidated results:"
echo "   Summary:          cat $CONSOLIDATED_DIR/consolidated_summary.json"
echo "   Detailed:         cat $CONSOLIDATED_DIR/consolidated_detailed.json"
echo "   Markdown Report:  cat $CONSOLIDATED_DIR/FINAL_SUMMARY.md"
echo "   7-way images:     ls $CONSOLIDATED_DIR/samples/*/comparison_7way.png"
echo ""
echo "Models evaluated:"
echo "   1. Baseline (Qwen3-VL-4B-Instruct zero-shot)"
echo "   2. Standard Vision (vision-language trained)"
echo "   3. RL Vision (vision-language + RL filtering >= 4.0)"
echo "   4. RW Vision (vision-language + reward-weighted samples)"
echo "   5. DPO Vision (vision-language + preference pairs)"
echo ""
echo "Execution time saved: ~80% (parallel vs sequential)"
echo ""

