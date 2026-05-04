#!/usr/bin/env python3
"""
Aggregate GPT-4o image judge scores from detailed_results_all.json into evaluation_summary_all.json
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def aggregate_image_scores(dataset_dir):
    """Aggregate GPT-4o image judge scores from detailed results"""
    dataset_path = Path(dataset_dir)
    detailed_file = dataset_path / "detailed_results_all.json"
    summary_file = dataset_path / "evaluation_summary_all.json"
    
    if not detailed_file.exists():
        print(f"❌ Error: {detailed_file} not found")
        return False
    
    # Load detailed results
    with open(detailed_file, 'r') as f:
        detailed_results = json.load(f)
    
    # Aggregate image judge scores
    image_scores = {
        'instruction_following': [],
        'visual_quality': [],
        'transformation_strength': [],
        'coherence': [],
        'semantic_accuracy': [],
        'technical_execution': [],
        'overall_image_score': []
    }
    
    for sample in detailed_results:
        if 'image_metrics' in sample and 'gpt_judge_overall' in sample['image_metrics']:
            im = sample['image_metrics']
            image_scores['instruction_following'].append(im['gpt_judge_instruction_following'])
            image_scores['visual_quality'].append(im['gpt_judge_visual_quality'])
            image_scores['transformation_strength'].append(im['gpt_judge_transformation_strength'])
            image_scores['coherence'].append(im['gpt_judge_coherence'])
            image_scores['semantic_accuracy'].append(im['gpt_judge_semantic_accuracy'])
            image_scores['technical_execution'].append(im['gpt_judge_technical_execution'])
            image_scores['overall_image_score'].append(im['gpt_judge_overall'])
    
    if not image_scores['overall_image_score']:
        print(f"❌ No GPT-4o image judge scores found in {detailed_file}")
        return False
    
    # Calculate averages
    gpt_image_scores = {
        key: sum(values) / len(values)
        for key, values in image_scores.items()
    }
    
    # Load or create summary
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Add image judge scores
    summary['gpt_image_scores'] = gpt_image_scores
    
    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Aggregated {len(image_scores['overall_image_score'])} samples")
    print(f"   Average overall score: {gpt_image_scores['overall_image_score']:.2f}")
    print(f"   Saved to: {summary_file}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_gpt4o_image_scores.py <dataset_dir> [<dataset_dir2> ...]")
        print("")
        print("Example:")
        print("  python aggregate_gpt4o_image_scores.py \\")
        print("    evaluation_results/text_parallel_cot_8b_trajectory/gpt4o \\")
        print("    evaluation_results/text_parallel_complex_cot_8b_trajectory/gpt4o")
        sys.exit(1)
    
    print("════════════════════════════════════════════════════════════════════════════════")
    print("  Aggregating GPT-4o Image Judge Scores")
    print("════════════════════════════════════════════════════════════════════════════════")
    print("")
    
    success_count = 0
    for dataset_dir in sys.argv[1:]:
        print(f"📂 {dataset_dir}")
        if aggregate_image_scores(dataset_dir):
            success_count += 1
        print("")
    
    print("════════════════════════════════════════════════════════════════════════════════")
    print(f"  ✅ Complete: {success_count}/{len(sys.argv)-1} datasets processed")
    print("════════════════════════════════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()

