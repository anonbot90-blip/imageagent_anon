#!/bin/bash
################################################################################
# Chain-of-Thought (CoT) Full Pipeline Test
#
# This script:
# 1. Generates 5 CoT training samples with reasoning
# 2. Duplicates them to 100 samples for proper training
# 3. Trains all 4 model variants (Standard, RL, RW, DPO) text-only
# 4. Evaluates each model
# 5. Compares CoT vs non-CoT results
#
# All outputs go to separate directories to avoid touching existing data
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_color() {
    echo -e "${1}${2}${NC}"
}

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# CoT-specific directories
COT_DATA_DIR="training_data/cot_test"
COT_CHECKPOINT_DIR="checkpoints/cot_test"
COT_EVAL_DIR="evaluation_results/cot_test"

# Evaluation configuration
NUM_TEST_SAMPLES=2  # Small test for quick validation

print_color $CYAN "╔════════════════════════════════════════════════════════════════╗"
print_color $CYAN "║    Chain-of-Thought Action Planner - Full Pipeline Test       ║"
print_color $CYAN "╚════════════════════════════════════════════════════════════════╝"
echo ""
print_color $BLUE "📁 Data Directory: $COT_DATA_DIR"
print_color $BLUE "💾 Checkpoint Directory: $COT_CHECKPOINT_DIR"
print_color $BLUE "📊 Evaluation Directory: $COT_EVAL_DIR"
echo ""

# ################################################################################
# # Step 1: Generate 5 CoT Training Samples
# ################################################################################

# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# print_color $PURPLE "STEP 1: Generate 5 NEW CoT Samples (Running Full Pipeline)"
# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# echo ""

# print_color $YELLOW "🚀 Running ImageAgent pipeline to generate 5 samples with CoT reasoning..."
# print_color $CYAN "   This will use the CoT-enabled planner"
# echo ""

# # Set output directory for new CoT samples
# COT_RESULTS_DIR="imageagent_results_cot_test"

# # Run the ImageAgent pipeline with 5 samples
# bash scripts/run_imageagent1.sh \
#     --num-samples 5 \
#     --output-dir "$COT_RESULTS_DIR"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ Generated 5 new CoT samples in $COT_RESULTS_DIR"
# else
#     print_color $RED "❌ Failed to generate CoT samples"
#     exit 1
# fi
# echo ""

# # Generate training data from these 5 samples (reads action_plan.json with reasoning)
# print_color $YELLOW "🔄 Converting pipeline results to training data JSON..."
# python scripts/generate_planner_training_data.py \
#     "$COT_RESULTS_DIR" \
#     --output-dir "$COT_DATA_DIR/text" \
#     --num-datapoints 5

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ Generated training data from 5 CoT samples"
# else
#     print_color $RED "❌ Failed to generate training data"
#     exit 1
# fi
# echo ""

# ################################################################################
# # Step 1B: Generate Training Data from Existing CoT Results
# ################################################################################

# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# print_color $PURPLE "STEP 1: Generate Training Data from Existing CoT Samples"
# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# echo ""

# print_color $YELLOW "🔄 Converting existing imageagent_results_cot_test to training data..."
# python scripts/generate_planner_training_data.py \
#     "imageagent_results_cot_test" \
#     --output-dir "$COT_DATA_DIR/text" \
#     --num-datapoints 5

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ Generated training data from existing CoT samples"
# else
#     print_color $RED "❌ Failed to generate training data"
#     exit 1
# fi
# echo ""

# ################################################################################
# # Step 2: Duplicate to 100 Samples
# ################################################################################

# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# print_color $PURPLE "STEP 2: Duplicate 5 Samples to 100 Samples"
# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# echo ""

# print_color $YELLOW "📋 Creating 100 training samples (20 copies of each)..."

# python << 'EOF'
# import json
# from pathlib import Path
# import copy

# # Load the 5 samples
# data_file = Path("training_data/cot_test/text/planner_training_data.json")
# with open(data_file, 'r') as f:
#     data = json.load(f)

# # Extract samples array
# samples = data.get('samples', data) if isinstance(data, dict) else data
# print(f"Loaded {len(samples)} original samples")

# # Duplicate each sample 20 times (5 * 20 = 100)
# duplicated = []
# for i, sample in enumerate(samples):
#     for copy_idx in range(20):
#         # Create deep copy with unique ID
#         new_sample = copy.deepcopy(sample)
#         new_sample['id'] = f"{sample['id']}_copy{copy_idx}"
#         if 'metadata' not in new_sample:
#             new_sample['metadata'] = {}
#         new_sample['metadata']['is_duplicate'] = True
#         new_sample['metadata']['original_id'] = sample['id']
#         new_sample['metadata']['copy_number'] = copy_idx
#         duplicated.append(new_sample)

# print(f"Created {len(duplicated)} total samples")

# # Update data structure
# data['samples'] = duplicated
# data['total_samples'] = len(duplicated)

# # Save duplicated dataset
# with open(data_file, 'w') as f:
#     json.dump(data, f, indent=2)

# print(f"✓ Saved to {data_file}")
# EOF

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ Created 100 training samples"
# else
#     print_color $RED "❌ Failed to duplicate samples"
#     exit 1
# fi
# echo ""

# # Verify
# NUM_SAMPLES=$(jq '. | length' "$COT_DATA_DIR/text/planner_training_data.json")
# print_color $CYAN "📊 Total training samples: $NUM_SAMPLES"
# echo ""

# ################################################################################
# # Step 3: Train Models (Text-Only for Speed)
# ################################################################################

# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# print_color $PURPLE "STEP 3: Train All 4 Model Variants with CoT"
# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# echo ""

# print_color $YELLOW "📝 Training will use config files for all parameters"
# print_color $CYAN "  - Using torchrun for distributed training (8 GPUs)"
# print_color $CYAN "  - Config files: training/planner_training/planner_config_*.yaml"
# echo ""

# ################################################################################
# # 3.1: Train Standard Model
# ################################################################################

# print_color $BLUE "──────────────────────────────────────────────────────────────"
# print_color $BLUE "3.1: Training Standard CoT Model"
# print_color $BLUE "──────────────────────────────────────────────────────────────"
# echo ""

# print_color $YELLOW "🚀 Starting Standard model training..."

# torchrun --nproc_per_node=8 --master_port=29502 training/planner_training/train_planner_text_only.py \
#     --data-path "$COT_DATA_DIR/text/planner_training_data.json" \
#     --output-dir "$COT_CHECKPOINT_DIR/standard_text"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ Standard model training complete"
# else
#     print_color $RED "❌ Standard model training failed"
# fi
# echo ""

# ################################################################################
# # 3.2: Train RL Model
# ################################################################################

# print_color $BLUE "──────────────────────────────────────────────────────────────"
# print_color $BLUE "3.2: Training RL CoT Model (Standard + RL fine-tuning)"
# print_color $BLUE "──────────────────────────────────────────────────────────────"
# echo ""

# print_color $YELLOW "🚀 Starting RL model training..."

# # RL training starts from Standard checkpoint
# torchrun --nproc_per_node=8 --master_port=29503 training/planner_training/train_planner_rl_text.py \
#     --data-path "$COT_DATA_DIR/text/planner_training_data.json" \
#     --output-dir "$COT_CHECKPOINT_DIR/rl_text"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ RL model training complete"
# else
#     print_color $RED "❌ RL model training failed"
# fi
# echo ""

# ################################################################################
# # 3.3: Generate RW Data and Train RW Model
# ################################################################################

# print_color $BLUE "──────────────────────────────────────────────────────────────"
# print_color $BLUE "3.3: Training RW CoT Model (Reward-Weighted)"
# print_color $BLUE "──────────────────────────────────────────────────────────────"
# echo ""

# print_color $YELLOW "📊 Generating RW training data (with reward scores)..."

# # Generate RW data from existing imageagent_results_cot_test (has reward scores)
# python scripts/generate_planner_training_data.py \
#     "imageagent_results_cot_test" \
#     --output-dir "$COT_DATA_DIR/rw_text" \
#     --rl-data \
#     --threshold 0.0 \
#     --num-datapoints 5

# # Duplicate to 100
# python << 'EOF'
# import json
# from pathlib import Path
# import copy

# data_file = Path("training_data/cot_test/rw_text/planner_training_data.json")
# with open(data_file, 'r') as f:
#     data = json.load(f)

# samples = data.get('samples', data) if isinstance(data, dict) else data
# print(f"Loaded {len(samples)} RW samples")

# duplicated = []
# for sample in samples:
#     for copy_idx in range(20):
#         new_sample = copy.deepcopy(sample)
#         new_sample['id'] = f"{sample['id']}_copy{copy_idx}"
#         if 'metadata' not in new_sample:
#             new_sample['metadata'] = {}
#         new_sample['metadata']['is_duplicate'] = True
#         new_sample['metadata']['copy_number'] = copy_idx
#         duplicated.append(new_sample)

# # Update structure
# data['samples'] = duplicated
# data['total_samples'] = len(duplicated)

# with open(data_file, 'w') as f:
#     json.dump(data, f, indent=2)

# print(f"✓ Created {len(duplicated)} RW samples")
# EOF

# print_color $GREEN "✓ RW training data ready"
# echo ""

# print_color $YELLOW "🚀 Starting RW model training..."

# torchrun --nproc_per_node=8 --master_port=29504 training/planner_training/train_planner_rw_text.py \
#     --data-path "$COT_DATA_DIR/rw_text/planner_training_data.json" \
#     --output-dir "$COT_CHECKPOINT_DIR/rw_text"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ RW model training complete"
# else
#     print_color $RED "❌ RW model training failed"
# fi
# echo ""

# ################################################################################
# # 3.4: Generate DPO Data and Train DPO Model
# ################################################################################

# print_color $BLUE "──────────────────────────────────────────────────────────────"
# print_color $BLUE "3.4: Training DPO CoT Model (Preference Optimization)"
# print_color $BLUE "──────────────────────────────────────────────────────────────"
# echo ""

# print_color $YELLOW "📊 Generating DPO preference pairs..."

# # Note: DPO needs preference pairs (chosen/rejected) with score_margin
# python << 'EOF'
# import json
# from pathlib import Path
# import copy

# # Load standard samples
# standard_file = Path("training_data/cot_test/text/planner_training_data.json")
# with open(standard_file, 'r') as f:
#     data = json.load(f)

# samples = data.get('samples', data) if isinstance(data, dict) else data

# # Create DPO pairs (chosen=original with reasoning, rejected=without reasoning)
# dpo_samples = []
# for sample in samples[:5]:  # Use 5 unique samples only
#     # Keep original as chosen
#     chosen_plan = copy.deepcopy(sample['target_action_plan'])
    
#     # Create rejected by removing detailed reasoning
#     rejected_plan = copy.deepcopy(chosen_plan)
#     if 'reasoning' in rejected_plan:
#         rejected_plan['reasoning'] = "Basic edit needed."
    
#     dpo_sample = {
#         'id': sample['id'],
#         'image_path': sample['image_path'],
#         'analysis_path': sample.get('analysis_path', ''),
#         'user_prompt': sample['user_prompt'],
#         'chosen_plan': chosen_plan,  # Fixed: use chosen_plan not chosen_action_plan
#         'rejected_plan': rejected_plan,  # Fixed: use rejected_plan not rejected_action_plan
#         'score_margin': 1.5,  # Synthetic margin for training
#         'metadata': copy.deepcopy(sample.get('metadata', {}))
#     }
#     dpo_samples.append(dpo_sample)

# print(f"Created {len(dpo_samples)} base DPO pairs")

# # Duplicate to 100
# duplicated = []
# for sample in dpo_samples:
#     for copy_idx in range(20):
#         new_sample = copy.deepcopy(sample)
#         new_sample['id'] = f"{sample['id']}_dpo_copy{copy_idx}"
#         if 'metadata' not in new_sample:
#             new_sample['metadata'] = {}
#         new_sample['metadata']['is_duplicate'] = True
#         new_sample['metadata']['copy_number'] = copy_idx
#         duplicated.append(new_sample)

# # Save with proper structure
# dpo_dir = Path("training_data/cot_test/dpo_text")
# dpo_dir.mkdir(parents=True, exist_ok=True)

# dpo_data = {
#     "version": "3.0",
#     "description": "DPO preference pairs for planner training",
#     "total_samples": len(duplicated),
#     "samples": duplicated
# }

# with open(dpo_dir / "planner_training_data_dpo.json", 'w') as f:
#     json.dump(dpo_data, f, indent=2)

# print(f"✓ Created {len(duplicated)} DPO preference pairs with score_margin")
# EOF

# print_color $GREEN "✓ DPO training data ready"
# echo ""

# print_color $YELLOW "🚀 Starting DPO model training..."

# torchrun --nproc_per_node=8 --master_port=29505 training/planner_training/train_planner_dpo_text.py \
#     --data-path "$COT_DATA_DIR/dpo_text/planner_training_data_dpo.json" \
#     --output-dir "$COT_CHECKPOINT_DIR/dpo_text"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ DPO model training complete"
# else
#     print_color $RED "❌ DPO model training failed"
# fi
# echo ""

# ################################################################################
# # Step 4: Evaluate All Models
# ################################################################################

# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# print_color $PURPLE "STEP 4: Evaluate All CoT Models"
# print_color $PURPLE "═══════════════════════════════════════════════════════════════"
# echo ""

# # Use validation split from training data
# NUM_TEST_SAMPLES=2  # Small test for quick validation

# print_color $YELLOW "📝 Evaluating on $NUM_TEST_SAMPLES validation samples..."
# echo ""

# # Evaluate each model
# for model_type in standard_text rl_text rw_text dpo_text; do
#     if [ ! -d "$COT_CHECKPOINT_DIR/$model_type/final" ]; then
#         print_color $YELLOW "⚠️  Skipping $model_type (not trained)"
#         continue
#     fi
    
#     print_color $BLUE "──────────────────────────────────────────────────────────────"
#     print_color $BLUE "Evaluating: $model_type"
#     print_color $BLUE "──────────────────────────────────────────────────────────────"
#     echo ""
    
#     python scripts/evaluate_planner.py \
#         --checkpoint "$COT_CHECKPOINT_DIR/$model_type" \
#         --data "$COT_DATA_DIR/text/planner_training_data.json" \
#         --output "$COT_EVAL_DIR/$model_type" \
#         --split val \
#         --num_samples $NUM_TEST_SAMPLES \
#         --save-predictions \
#         --use-gpt-judge
    
#     if [ $? -eq 0 ]; then
#         print_color $GREEN "✓ Evaluation complete for $model_type"
#     else
#         print_color $YELLOW "⚠️  Evaluation failed for $model_type (continuing...)"
#     fi
#     echo ""
# done

# ################################################################################
# # PART 2: VISION MODELS
# ################################################################################

# print_color $PURPLE "╔════════════════════════════════════════════════════════════════╗"
# print_color $PURPLE "║                  PART 2: VISION MODELS                         ║"
# print_color $PURPLE "╚════════════════════════════════════════════════════════════════╝"
# echo ""

# ################################################################################
# # 5.0: Generate Vision Embeddings
# ################################################################################

# print_color $BLUE "══════════════════════════════════════════════════════════════"
# print_color $BLUE "5.0: Generating Vision Embeddings (Cached Training)"
# print_color $BLUE "══════════════════════════════════════════════════════════════"
# echo ""

# # Check if embeddings exist
# if [ ! -f "$COT_DATA_DIR/vision/vision_embeddings.h5" ]; then
#     print_color $YELLOW "🔄 Pre-computing vision embeddings..."
    
#     # First, copy the training data JSON to vision directory
#     mkdir -p "$COT_DATA_DIR/vision"
#     cp "$COT_DATA_DIR/text/planner_training_data.json" "$COT_DATA_DIR/vision/"
    
#     # Then compute embeddings
#     python scripts/precompute_image_embeddings.py \
#         --data-path "$COT_DATA_DIR/vision/planner_training_data.json" \
#         --output-dir "$COT_DATA_DIR/vision"
    
#     print_color $GREEN "✓ Vision embeddings generated"
# else
#     print_color $GREEN "✓ Vision embeddings already exist"
    
#     # Make sure the JSON file exists too
#     if [ ! -f "$COT_DATA_DIR/vision/planner_training_data.json" ]; then
#         print_color $YELLOW "Copying training data JSON to vision directory..."
#         cp "$COT_DATA_DIR/text/planner_training_data.json" "$COT_DATA_DIR/vision/"
#     fi
# fi
# echo ""

# ################################################################################
# # 5.1: Train Standard Vision Model
# ################################################################################

# print_color $BLUE "══════════════════════════════════════════════════════════════"
# print_color $BLUE "5.1: Training Standard CoT Vision Model"
# print_color $BLUE "══════════════════════════════════════════════════════════════"
# echo ""

# torchrun --nproc_per_node=8 --master_port=29506 training/planner_training/train_planner_cached.py \
#     --data-path "$COT_DATA_DIR/vision/planner_training_data.json" \
#     --embeddings-path "$COT_DATA_DIR/vision/vision_embeddings.h5" \
#     --output-dir "$COT_CHECKPOINT_DIR/standard_vision"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ Standard vision model training complete"
# else
#     print_color $YELLOW "⚠️  Standard vision model training failed"
# fi
# echo ""

# ################################################################################
# # 5.2: Train RL Vision Model
# ################################################################################

# print_color $BLUE "══════════════════════════════════════════════════════════════"
# print_color $BLUE "5.2: Training RL CoT Vision Model"
# print_color $BLUE "══════════════════════════════════════════════════════════════"
# echo ""

# torchrun --nproc_per_node=8 --master_port=29507 training/planner_training/train_planner_rl_cached.py \
#     --data-path "$COT_DATA_DIR/vision/planner_training_data.json" \
#     --embeddings-path "$COT_DATA_DIR/vision/vision_embeddings.h5" \
#     --output-dir "$COT_CHECKPOINT_DIR/rl_vision"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ RL vision model training complete"
# else
#     print_color $YELLOW "⚠️  RL vision model training failed"
# fi
# echo ""

# ################################################################################
# # 5.3: Prepare RW Vision Data and Train
# ################################################################################

# print_color $BLUE "══════════════════════════════════════════════════════════════"
# print_color $BLUE "5.3: Training RW CoT Vision Model"
# print_color $BLUE "══════════════════════════════════════════════════════════════"
# echo ""

# # Prepare RW vision data if needed
# if [ ! -f "$COT_DATA_DIR/rw_vision/planner_training_data.json" ]; then
#     print_color $YELLOW "Preparing RW vision data..."
#     mkdir -p "$COT_DATA_DIR/rw_vision"
#     cp "$COT_DATA_DIR/rw_text/planner_training_data.json" "$COT_DATA_DIR/rw_vision/"
#     cp "$COT_DATA_DIR/vision/vision_embeddings.h5" "$COT_DATA_DIR/rw_vision/"
#     cp "$COT_DATA_DIR/vision/embeddings_manifest.json" "$COT_DATA_DIR/rw_vision/"
#     print_color $GREEN "✓ RW vision data prepared (copied embeddings + manifest)"
# fi

# torchrun --nproc_per_node=8 --master_port=29508 training/planner_training/train_planner_rw_cached.py \
#     --data-path "$COT_DATA_DIR/rw_vision/planner_training_data.json" \
#     --embeddings-path "$COT_DATA_DIR/rw_vision/vision_embeddings.h5" \
#     --output-dir "$COT_CHECKPOINT_DIR/rw_vision"

# if [ $? -eq 0 ]; then
#     print_color $GREEN "✓ RW vision model training complete"
# else
#     print_color $YELLOW "⚠️  RW vision model training failed"
# fi
# echo ""

################################################################################
# 5.4: Prepare DPO Vision Data and Train
################################################################################

print_color $BLUE "══════════════════════════════════════════════════════════════"
print_color $BLUE "5.4: Training DPO CoT Vision Model"
print_color $BLUE "══════════════════════════════════════════════════════════════"
echo ""

# Prepare DPO vision data if needed
if [ ! -f "$COT_DATA_DIR/dpo_vision/planner_training_data_dpo.json" ]; then
    print_color $YELLOW "Preparing DPO vision data..."
    mkdir -p "$COT_DATA_DIR/dpo_vision"
    cp "$COT_DATA_DIR/dpo_text/planner_training_data_dpo.json" "$COT_DATA_DIR/dpo_vision/"
    cp "$COT_DATA_DIR/vision/vision_embeddings.h5" "$COT_DATA_DIR/dpo_vision/"
    cp "$COT_DATA_DIR/vision/embeddings_manifest.json" "$COT_DATA_DIR/dpo_vision/"
    print_color $GREEN "✓ DPO vision data prepared (copied embeddings + manifest)"
fi

torchrun --nproc_per_node=8 --master_port=29509 training/planner_training/train_planner_dpo_cached.py \
    --data-path "$COT_DATA_DIR/dpo_vision/planner_training_data_dpo.json" \
    --embeddings-path "$COT_DATA_DIR/dpo_vision/vision_embeddings.h5" \
    --output-dir "$COT_CHECKPOINT_DIR/dpo_vision"

if [ $? -eq 0 ]; then
    print_color $GREEN "✓ DPO vision model training complete"
else
    print_color $YELLOW "⚠️  DPO vision model training failed"
fi
echo ""

################################################################################
# 5.5: Evaluate All Vision Models
################################################################################

print_color $BLUE "══════════════════════════════════════════════════════════════"
print_color $BLUE "5.5: Evaluating All Vision Models"
print_color $BLUE "══════════════════════════════════════════════════════════════"
echo ""

for model_type in standard_vision rl_vision rw_vision dpo_vision; do
    if [ ! -d "$COT_CHECKPOINT_DIR/$model_type/final" ]; then
        print_color $YELLOW "⚠️  Skipping $model_type (not trained)"
        continue
    fi
    
    print_color $CYAN "Evaluating: $model_type"
    
    python scripts/evaluate_planner.py \
        --checkpoint "$COT_CHECKPOINT_DIR/$model_type" \
        --data "$COT_DATA_DIR/vision/planner_training_data.json" \
        --output "$COT_EVAL_DIR/$model_type" \
        --split val \
        --num_samples $NUM_TEST_SAMPLES \
        --save-predictions \
        --use-gpt-judge
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "✓ Evaluation complete for $model_type"
    else
        print_color $YELLOW "⚠️  Evaluation failed for $model_type"
    fi
    echo ""
done

print_color $GREEN "✓ Vision model evaluation complete"
echo ""

################################################################################
# Step 5 (Optional): End-to-End Evaluation with Image Generation
################################################################################

print_color $PURPLE "═══════════════════════════════════════════════════════════════"
print_color $PURPLE "STEP 5: End-to-End Evaluation with HiDream"
print_color $PURPLE "═══════════════════════════════════════════════════════════════"
echo ""

# HiDream configuration
HIDREAM_MODEL="$PROJECT_ROOT/HiDream-E1"  # Base model (untrained)
HIDREAM_CONFIG="$PROJECT_ROOT/training/config/training_config.yaml"

print_color $CYAN "🖼️  Generating actual room images using HiDream-E1..."
print_color $CYAN "   Model: Base HiDream-E1 (untrained)"
print_color $CYAN "   Samples: $NUM_TEST_SAMPLES validation samples"
print_color $CYAN "   Time estimate: ~2-3 minutes per sample"
echo ""

# Only evaluate text models (vision would be same images, just different planners)
for model_type in standard_text rl_text rw_text dpo_text; do
    if [ ! -d "$COT_CHECKPOINT_DIR/$model_type/final" ]; then
        print_color $YELLOW "⚠️  Skipping $model_type (not trained)"
        continue
    fi
    
    print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_color $CYAN "Generating images with $model_type planner..."
    print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    python scripts/evaluate_planner.py \
        --checkpoint "$COT_CHECKPOINT_DIR/$model_type" \
        --data "$COT_DATA_DIR/text/planner_training_data.json" \
        --output "$COT_EVAL_DIR/${model_type}_e2e" \
        --split val \
        --num_samples $NUM_TEST_SAMPLES \
        --hidream-checkpoint "$HIDREAM_MODEL" \
        --hidream-config "$HIDREAM_CONFIG" \
        --save-images \
        --save-predictions \
        --use-gpt-judge
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "✓ Images generated for $model_type"
        print_color $CYAN "   Output: $COT_EVAL_DIR/${model_type}_e2e/samples/"
    else
        print_color $YELLOW "⚠️  Image generation failed for $model_type"
    fi
    echo ""
done

print_color $GREEN "✅ End-to-end evaluation complete!"
print_color $CYAN "   Generated images saved in: $COT_EVAL_DIR/*_e2e/samples/"
echo ""

################################################################################
# Step 5.1: Consolidate E2E Results
################################################################################

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "Consolidating E2E results..."
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash scripts/evaluation/consolidate_text_results.sh \
    --baseline-dir "$COT_EVAL_DIR/standard_text_e2e" \
    --standard-text-dir "$COT_EVAL_DIR/standard_text_e2e" \
    --rl-text-dir "$COT_EVAL_DIR/rl_text_e2e" \
    --rw-text-dir "$COT_EVAL_DIR/rw_text_e2e" \
    --dpo-text-dir "$COT_EVAL_DIR/dpo_text_e2e" \
    --output-dir "$COT_EVAL_DIR/consolidated_e2e"

if [ $? -eq 0 ]; then
    print_color $GREEN "✅ E2E results consolidated!"
    print_color $CYAN "   Summary: $COT_EVAL_DIR/consolidated_e2e/FINAL_SUMMARY.md"
    print_color $CYAN "   Tables: $COT_EVAL_DIR/consolidated_e2e/*.png"
else
    print_color $YELLOW "⚠️  Consolidation failed (continuing...)"
fi
echo ""

################################################################################
# Step 6: Final Results Summary
################################################################################

print_color $PURPLE "═══════════════════════════════════════════════════════════════"
print_color $PURPLE "STEP 6: CoT Test Results Summary"
print_color $PURPLE "═══════════════════════════════════════════════════════════════"
echo ""

print_color $CYAN "📊 Results Location:"
echo ""
print_color $GREEN "✓ Evaluation Results:"
print_color $CYAN "  - Text models: $COT_EVAL_DIR/{standard,rl,rw,dpo}_text/"
print_color $CYAN "  - Vision models: $COT_EVAL_DIR/{standard,rl,rw,dpo}_vision/"

# Check if E2E evaluation was run
if [ -d "$COT_EVAL_DIR/standard_text_e2e" ]; then
    print_color $CYAN "  - E2E with images: $COT_EVAL_DIR/{standard,rl,rw,dpo}_text_e2e/"
fi

# Check if consolidation was run
if [ -f "$COT_EVAL_DIR/consolidated_e2e/FINAL_SUMMARY.md" ]; then
    print_color $GREEN "  - Consolidated Report: $COT_EVAL_DIR/consolidated_e2e/FINAL_SUMMARY.md"
    print_color $CYAN "  - Visual Tables: $COT_EVAL_DIR/consolidated_e2e/*.png"
fi
echo ""

print_color $YELLOW "Note: Consolidation skipped for small test set (use full pipeline for consolidated reports)"
echo ""

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "TEXT MODEL RESULTS (2 validation samples)"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for model_type in standard_text rl_text rw_text dpo_text; do
    gpt_file="$COT_EVAL_DIR/$model_type/gpt4o_action_scores_val.json"
    summary_file="$COT_EVAL_DIR/$model_type/evaluation_summary_val.json"
    
    if [ -f "$gpt_file" ]; then
        print_color $BLUE "┌─ $model_type ─────────────────────────────────────────────┐"
        
        # GPT Judge Scores
        print_color $YELLOW "│ GPT-4o Judge Scores:"
        jq -r '.aggregated_scores | "│   Relevance:    \(.relevance_mean)/10  (σ=\(.relevance_std))\n│   Completeness: \(.completeness_mean)/10  (σ=\(.completeness_std))\n│   Efficiency:   \(.efficiency_mean)/10  (σ=\(.efficiency_std))\n│   Correctness:  \(.correctness_mean)/10  (σ=\(.correctness_std))\n│   Overall:      \(.overall_score_mean)/10  (σ=\(.overall_score_std))"' "$gpt_file" 2>/dev/null | while IFS= read -r line; do
            print_color $CYAN "$line"
        done
        echo "│"
        
        # Planner Metrics
        if [ -f "$summary_file" ]; then
            print_color $YELLOW "│ Planner Metrics:"
            jq -r '.aggregated_planner_metrics | "│   Action F1:         \(.planner_action_f1_mean) (\(.planner_action_f1_min)-\(.planner_action_f1_max))\n│   Action IOU:        \(.planner_action_iou_mean) (\(.planner_action_iou_min)-\(.planner_action_iou_max))\n│   Valid JSON:        \(.planner_valid_json_mean * 100)%\n│   Avg Actions:       \(.planner_num_predicted_actions_mean) (GT: \(.planner_num_ground_truth_actions_mean))"' "$summary_file" 2>/dev/null | while IFS= read -r line; do
                print_color $CYAN "$line"
            done
        fi
        
        print_color $BLUE "└───────────────────────────────────────────────────────────┘"
        echo ""
    else
        print_color $RED "❌ $model_type: No evaluation results"
        echo ""
    fi
done

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "VISION MODEL RESULTS (2 validation samples)"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

vision_found=false
for model_type in standard_vision rl_vision rw_vision dpo_vision; do
    gpt_file="$COT_EVAL_DIR/$model_type/gpt4o_action_scores_val.json"
    summary_file="$COT_EVAL_DIR/$model_type/evaluation_summary_val.json"
    
    if [ -f "$gpt_file" ]; then
        vision_found=true
        print_color $BLUE "┌─ $model_type ─────────────────────────────────────────────┐"
        
        # GPT Judge Scores
        print_color $YELLOW "│ GPT-4o Judge Scores:"
        jq -r '.aggregated_scores | "│   Relevance:    \(.relevance_mean)/10  (σ=\(.relevance_std))\n│   Completeness: \(.completeness_mean)/10  (σ=\(.completeness_std))\n│   Efficiency:   \(.efficiency_mean)/10  (σ=\(.efficiency_std))\n│   Correctness:  \(.correctness_mean)/10  (σ=\(.correctness_std))\n│   Overall:      \(.overall_score_mean)/10  (σ=\(.overall_score_std))"' "$gpt_file" 2>/dev/null | while IFS= read -r line; do
            print_color $CYAN "$line"
        done
        echo "│"
        
        # Planner Metrics
        if [ -f "$summary_file" ]; then
            print_color $YELLOW "│ Planner Metrics:"
            jq -r '.aggregated_planner_metrics | "│   Action F1:         \(.planner_action_f1_mean) (\(.planner_action_f1_min)-\(.planner_action_f1_max))\n│   Action IOU:        \(.planner_action_iou_mean) (\(.planner_action_iou_min)-\(.planner_action_iou_max))\n│   Valid JSON:        \(.planner_valid_json_mean * 100)%\n│   Avg Actions:       \(.planner_num_predicted_actions_mean) (GT: \(.planner_num_ground_truth_actions_mean))"' "$summary_file" 2>/dev/null | while IFS= read -r line; do
                print_color $CYAN "$line"
            done
        fi
        
        print_color $BLUE "└───────────────────────────────────────────────────────────┘"
        echo ""
    fi
done

if [ "$vision_found" = false ]; then
    print_color $YELLOW "⚠️  No vision models were evaluated (training may have failed)"
    echo ""
fi

################################################################################
# Cleanup
################################################################################

print_color $YELLOW "🧹 Cleaning up temporary directories..."
rm -rf "$TMP_5_SAMPLES" "$COT_DATA_DIR/tmp_5_samples_rw"
print_color $GREEN "✓ Cleanup complete"
echo ""

################################################################################
# Summary
################################################################################

print_color $CYAN "╔════════════════════════════════════════════════════════════════╗"
print_color $CYAN "║                   CoT Pipeline Test Complete!                  ║"
print_color $CYAN "╚════════════════════════════════════════════════════════════════╝"
echo ""
print_color $GREEN "✓ Data: $COT_DATA_DIR"
print_color $GREEN "✓ Checkpoints: $COT_CHECKPOINT_DIR"
print_color $GREEN "✓ Evaluation: $COT_EVAL_DIR"
echo ""
print_color $BLUE "📝 To compare with baseline (non-CoT) results:"
print_color $CYAN "  - CoT results: $COT_EVAL_DIR/<model>/"
print_color $CYAN "  - Baseline results: evaluation_results/text_parallel_eval_40000/"
echo ""
print_color $PURPLE "🎉 Chain-of-Thought implementation test complete!"

