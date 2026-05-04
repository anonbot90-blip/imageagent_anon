#!/usr/bin/env python3
"""
Aggregate GPT-4o scores from individual sample reward_scores.json files.
Creates gpt4o_action_scores_all.json and gpt4o_image_scores_all.json for each model.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

def load_reward_scores(sample_dir: Path) -> Dict:
    """Load reward scores from a sample directory."""
    scores_file = sample_dir / "reward_scores.json"
    if not scores_file.exists():
        return None
    
    with open(scores_file, 'r') as f:
        return json.load(f)

def aggregate_action_scores(model_dir: Path) -> Dict:
    """Aggregate action-related GPT scores from all samples."""
    samples_dir = model_dir / "samples"
    
    if not samples_dir.exists():
        print(f"⚠️  No samples directory found in {model_dir}")
        return None
    
    # Metrics to aggregate (action plan quality and reasoning)
    action_metrics = [
        'action_plan_quality',
        'plan_reasoning',
        'reasoning_quality'
    ]
    
    all_scores = {metric: [] for metric in action_metrics}
    sample_count = 0
    
    # Iterate through all sample directories
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        reward_data = load_reward_scores(sample_dir)
        if not reward_data or 'scores' not in reward_data:
            continue
        
        scores = reward_data['scores']
        for metric in action_metrics:
            if metric in scores and 'score' in scores[metric]:
                all_scores[metric].append(scores[metric]['score'])
        
        sample_count += 1
    
    if sample_count == 0:
        print(f"⚠️  No valid samples found in {model_dir}")
        return None
    
    # Compute aggregated statistics
    aggregated = {}
    for metric, values in all_scores.items():
        if values:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    return {
        'aggregated_scores': aggregated,
        'sample_count': sample_count,
        'model_dir': str(model_dir)
    }

def aggregate_image_scores(model_dir: Path) -> Dict:
    """Aggregate image-related GPT scores from all samples."""
    samples_dir = model_dir / "samples"
    
    if not samples_dir.exists():
        print(f"⚠️  No samples directory found in {model_dir}")
        return None
    
    # Metrics to aggregate (image quality assessment)
    image_metrics = [
        'final_image_quality',
        'adherence_to_plan',
        'adherence_to_prompt',
        'overall_quality'
    ]
    
    all_scores = {metric: [] for metric in image_metrics}
    sample_count = 0
    
    # Iterate through all sample directories
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        reward_data = load_reward_scores(sample_dir)
        if not reward_data or 'scores' not in reward_data:
            continue
        
        scores = reward_data['scores']
        for metric in image_metrics:
            if metric in scores and 'score' in scores[metric]:
                all_scores[metric].append(scores[metric]['score'])
        
        sample_count += 1
    
    if sample_count == 0:
        print(f"⚠️  No valid samples found in {model_dir}")
        return None
    
    # Compute aggregated statistics
    aggregated = {}
    for metric, values in all_scores.items():
        if values:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    return {
        'aggregated_scores': aggregated,
        'sample_count': sample_count,
        'model_dir': str(model_dir)
    }

def main():
    parser = argparse.ArgumentParser(description="Aggregate GPT-4o scores from samples")
    parser.add_argument("--model-dir", required=True, help="Model directory containing samples/")
    parser.add_argument("--output-action", help="Output file for action scores (default: model_dir/gpt4o_action_scores_all.json)")
    parser.add_argument("--output-image", help="Output file for image scores (default: model_dir/gpt4o_image_scores_all.json)")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Set default output paths
    action_output = Path(args.output_action) if args.output_action else model_dir / "gpt4o_action_scores_all.json"
    image_output = Path(args.output_image) if args.output_image else model_dir / "gpt4o_image_scores_all.json"
    
    print(f"📊 Aggregating GPT-4o scores from: {model_dir}")
    
    # Aggregate action scores
    action_scores = aggregate_action_scores(model_dir)
    if action_scores:
        with open(action_output, 'w') as f:
            json.dump(action_scores, f, indent=2)
        print(f"✅ Action scores saved to: {action_output}")
        print(f"   Samples processed: {action_scores['sample_count']}")
    
    # Aggregate image scores
    image_scores = aggregate_image_scores(model_dir)
    if image_scores:
        with open(image_output, 'w') as f:
            json.dump(image_scores, f, indent=2)
        print(f"✅ Image scores saved to: {image_output}")
        print(f"   Samples processed: {image_scores['sample_count']}")

if __name__ == "__main__":
    main()
