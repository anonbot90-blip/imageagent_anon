#!/usr/bin/env python3
"""
Generate Standardized Weighted (SW) Training Data for Trajectory-Based Training

This script generates training data with standardized trajectory-level weights:
1. Loads trajectory average rewards
2. Computes standardized weights (z-scores) for each trajectory
3. Assigns the same standardized weight to all samples from each trajectory
4. Includes ALL samples (no filtering by threshold)

Weight calculation:
    weight_T = (trajectory_avg_reward - global_mean) / (global_std + epsilon)

This allows:
- Above-average trajectories get positive weights (emphasized in training)
- Below-average trajectories get negative weights (de-emphasized/unlearned)
- Smooth, continuous weighting (no discrete buckets like RW)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from trajectory_utils import (
    parse_trajectory_id,
    compute_standardized_trajectory_weights
)


def generate_sw_training_data(
    results_dir: Path,
    output_dir: Path,
    exclude_file: Path,
    trajectory_rewards_file: Path,
    threshold: float = 0.0,
    reward_metric: str = "overall_quality"
):
    """
    Generate SW training data with standardized trajectory-level weights.
    
    Args:
        results_dir: Directory containing imageagent results
        output_dir: Output directory for training data
        exclude_file: File containing test sample IDs to exclude
        trajectory_rewards_file: JSON file with trajectory average rewards
        threshold: Minimum reward threshold (default 0.0 = include all)
        reward_metric: Reward metric name (for verification)
    """
    print(f"Generating SW training data...")
    print(f"  Results dir: {results_dir}")
    print(f"  Threshold: {threshold} (includes ALL samples)")
    print(f"  Reward metric: {reward_metric}")
    print()
    
    # Load trajectory average rewards
    print("Loading trajectory rewards...")
    with open(trajectory_rewards_file, 'r') as f:
        trajectory_rewards = json.load(f)
    print(f"  ✓ Loaded {len(trajectory_rewards)} trajectory rewards")
    print()
    
    # Compute standardized weights
    print("Computing standardized trajectory weights...")
    standardized_weights = compute_standardized_trajectory_weights(trajectory_rewards)
    print(f"  ✓ Computed {len(standardized_weights)} standardized weights")
    print()
    
    # Load exclusion list (test samples)
    exclude_samples: Set[str] = set()
    if exclude_file.exists():
        with open(exclude_file, 'r') as f:
            exclude_samples = {line.strip() for line in f if line.strip()}
        print(f"✓ Loaded exclusion list: {len(exclude_samples)} test samples")
    else:
        print("⚠️  Warning: No exclusion file found, including all samples")
    print()
    
    # Process all samples
    training_samples = []
    skipped_count = 0
    excluded_count = 0
    missing_weight_count = 0
    
    print("Processing samples...")
    results_dir = Path(results_dir)
    
    # Find all sample folders
    sample_folders = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("image_")
    ])
    
    for folder in sample_folders:
        # Check exclusion
        if folder.name in exclude_samples:
            excluded_count += 1
            continue
        
        # Get trajectory ID
        trajectory_id = parse_trajectory_id(folder.name)
        
        # Get standardized weight for this trajectory
        if trajectory_id not in standardized_weights:
            print(f"⚠️  Warning: No standardized weight for trajectory {trajectory_id} (sample: {folder.name})")
            missing_weight_count += 1
            continue
        
        standardized_weight = standardized_weights[trajectory_id]
        trajectory_avg_reward = trajectory_rewards[trajectory_id]
        
        # Check paths
        original_path = folder / "original.png"
        action_plan_path = folder / "action_plan.json"
        prompt_path = folder / "prompt.json"
        
        if not original_path.exists() or not action_plan_path.exists() or not prompt_path.exists():
            skipped_count += 1
            continue
        
        # Load data
        try:
            # Load prompt
            with open(prompt_path, 'r') as f:
                prompt_data = json.load(f)
            
            # Extract user prompt
            if 'edit_info' in prompt_data and 'text' in prompt_data['edit_info']:
                user_prompt = prompt_data['edit_info']['text']
            elif 'edit' in prompt_data and 'text' in prompt_data['edit']:
                user_prompt = prompt_data['edit']['text']
            else:
                skipped_count += 1
                continue
            
            # Load action plan
            with open(action_plan_path, 'r') as f:
                action_plan = json.load(f)
            
            # Load reward score for verification
            reward_path = folder / "reward_scores.json"
            if reward_path.exists():
                with open(reward_path, 'r') as f:
                    reward_data = json.load(f)
                    if "scores" in reward_data and reward_metric in reward_data["scores"]:
                        individual_score = reward_data["scores"][reward_metric]["score"]
                    else:
                        individual_score = None
            else:
                individual_score = None
            
            # Create training sample with standardized weight
            analysis_path = folder / "analysis.json"
            
            # Use absolute paths to avoid symlink/mount point issues
            sample = {
                "id": folder.name,
                "image_path": str(original_path.resolve()),
                "analysis_path": str(analysis_path.resolve()),
                "user_prompt": user_prompt,
                "target_action_plan": action_plan,
                "trajectory_id": trajectory_id,
                "trajectory_avg_reward": trajectory_avg_reward,
                "standardized_weight": standardized_weight,  # Key field for SW training
                "individual_score": individual_score,  # Optional: for debugging
                "metadata": {
                    "source_dir": str(folder.resolve()),
                    "folder_name": folder.name
                }
            }
            
            # Apply threshold check (though default is 0.0)
            if trajectory_avg_reward < threshold:
                skipped_count += 1
                continue
            
            training_samples.append(sample)
            
            if len(training_samples) % 100 == 0:
                print(f"  ✓ Processed {len(training_samples)} samples...")
        
        except Exception as e:
            print(f"⚠️  Error processing {folder.name}: {str(e)}")
            skipped_count += 1
            continue
    
    # Save training data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "planner_training_data.json"
    
    # Save in the correct format (dict with 'samples' key, matching other data generation scripts)
    output_data = {
        "version": "1.0",
        "description": "Trajectory-based SW (Standardized Weighted) training data",
        "total_samples": len(training_samples),
        "threshold": threshold,
        "weighting": "z-score normalized (standardized)",
        "samples": training_samples
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print("="*70)
    print("SW Training Data Generation Complete")
    print("="*70)
    print()
    print(f"Total samples processed: {len(sample_folders)}")
    print(f"  ✓ Training samples: {len(training_samples)}")
    print(f"  - Excluded (test set): {excluded_count}")
    print(f"  - Skipped (missing data): {skipped_count}")
    print(f"  - Missing weights: {missing_weight_count}")
    print()
    
    # Weight distribution stats
    if training_samples:
        weights = [s["standardized_weight"] for s in training_samples]
        traj_avgs = [s["trajectory_avg_reward"] for s in training_samples]
        
        print(f"Standardized weight distribution:")
        print(f"  Min weight: {min(weights):.4f}")
        print(f"  Max weight: {max(weights):.4f}")
        print(f"  Positive weights: {sum(1 for w in weights if w > 0)} samples")
        print(f"  Negative weights: {sum(1 for w in weights if w < 0)} samples")
        print()
        print(f"Trajectory average reward distribution:")
        print(f"  Min: {min(traj_avgs):.4f}")
        print(f"  Max: {max(traj_avgs):.4f}")
        print()
    
    print(f"Output file: {output_file}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SW training data with standardized weights")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing imageagent results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for training data"
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        required=True,
        help="File containing test sample IDs to exclude"
    )
    parser.add_argument(
        "--trajectory-rewards-file",
        type=str,
        required=True,
        help="JSON file with trajectory average rewards"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum trajectory average reward threshold (default: 0.0)"
    )
    parser.add_argument(
        "--reward-metric",
        type=str,
        default="overall_quality",
        help="Reward metric name (default: overall_quality)"
    )
    
    args = parser.parse_args()
    
    generate_sw_training_data(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        exclude_file=Path(args.exclude_file),
        trajectory_rewards_file=Path(args.trajectory_rewards_file),
        threshold=args.threshold,
        reward_metric=args.reward_metric
    )


