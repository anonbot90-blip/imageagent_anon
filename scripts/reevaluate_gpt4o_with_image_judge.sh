#!/bin/bash
# Re-evaluate GPT-4o planner outputs with GPT-4o image judge
# Adds GPT-4o image quality scores to GPT-4o's detailed_results_all.json

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Re-evaluating GPT-4o with GPT-4o Image Judge"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Dataset to evaluate
DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: bash $0 <dataset_dir>"
    echo ""
    echo "Example:"
    echo "  bash $0 evaluation_results/text_parallel_cot_8b_trajectory/gpt4o"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Directory not found: $DATASET_DIR"
    exit 1
fi

echo "📂 Dataset: $DATASET_DIR"
echo ""

# Run Python script to add GPT-4o image judge scores
python3 << PYTHON_SCRIPT
import sys
import json
from pathlib import Path
from training.evaluation.gpt_judge import GPT4oJudge
from PIL import Image

dataset_dir = Path("$DATASET_DIR")
detailed_file = dataset_dir / "detailed_results_all.json"

if not detailed_file.exists():
    print(f"❌ Error: {detailed_file} not found")
    sys.exit(1)

# Load detailed results
with open(detailed_file, 'r') as f:
    detailed_results = json.load(f)

print(f"✅ Loaded {len(detailed_results)} samples")
print("")

# Initialize GPT-4o judge
print("🔧 Initializing GPT-4o Image Judge...")
gpt_judge = GPT4oJudge()
print("✅ GPT-4o judge initialized")
print("")

# Process each sample
for i, sample in enumerate(detailed_results, 1):
    sample_id = sample['sample_id']
    print(f"  [{i}/{len(detailed_results)}] Processing {sample_id}...", end=" ")
    
    # Get sample directory
    sample_dir = dataset_dir / "samples" / sample_id
    
    # Check if already has gpt_judge scores
    if 'image_metrics' in sample and 'gpt_judge_overall' in sample['image_metrics']:
        print("⏭️  Already has GPT-4o judge scores")
        continue
    
    # Get images
    original_img = sample_dir / "original.png"
    edited_img = sample_dir / "predicted_edit.png"
    
    if not original_img.exists() or not edited_img.exists():
        print("⚠️  Missing images")
        continue
    
    # Get user prompt
    user_prompt = sample.get('user_prompt', '')
    
    # Judge the edit
    try:
        # Load images
        orig_img_pil = Image.open(original_img)
        edit_img_pil = Image.open(edited_img)
        
        gpt_scores = gpt_judge.judge_single_edit(
            original_image=orig_img_pil,
            generated_image=edit_img_pil,
            instruction=user_prompt
        )
        
        # Add to image_metrics
        if 'image_metrics' not in sample:
            sample['image_metrics'] = {}
        
        sample['image_metrics']['gpt_judge_instruction_following'] = gpt_scores.get('instruction_following', 0.0)
        sample['image_metrics']['gpt_judge_visual_quality'] = gpt_scores.get('visual_quality', 0.0)
        sample['image_metrics']['gpt_judge_transformation_strength'] = gpt_scores.get('transformation_strength', 0.0)
        sample['image_metrics']['gpt_judge_coherence'] = gpt_scores.get('coherence', 0.0)
        sample['image_metrics']['gpt_judge_semantic_accuracy'] = gpt_scores.get('semantic_accuracy', 0.0)
        sample['image_metrics']['gpt_judge_technical_execution'] = gpt_scores.get('technical_execution', 0.0)
        sample['image_metrics']['gpt_judge_overall'] = gpt_scores.get('overall_image_score', 0.0)
        sample['image_metrics']['gpt_judge_reasoning'] = gpt_scores.get('reasoning', '')
        
        print("✅")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# Save updated results
print("")
print("💾 Saving updated results...")
with open(detailed_file, 'w') as f:
    json.dump(detailed_results, f, indent=2)
print(f"✅ Saved to {detailed_file}")

print("")
print("════════════════════════════════════════════════════════════════════════════════")
print("  ✅ COMPLETE! GPT-4o now has image judge scores")
print("════════════════════════════════════════════════════════════════════════════════")
PYTHON_SCRIPT

echo ""
echo "✅ Done! Run consolidation to see GPT-4o image judge scores in tables."
