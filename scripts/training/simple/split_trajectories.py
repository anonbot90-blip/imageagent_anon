#!/usr/bin/env python3
"""
Split Trajectories into Train/Test Sets

Creates trajectory-based train/test split with no leakage.
All samples from a test trajectory go to test set.
All samples from a train trajectory go to train set.

Usage:
    python scripts/training/split_trajectories.py \
        --results-dir imageagent_results_16000_cot \
        --test-threshold 2.5 \
        --test-count 200 \
        --reward-metric overall_quality \
        --output-dir training_data/cot_8b_trajectory \
        --prefix cot_8b
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to import trajectory_utils
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from trajectory_utils import (
    compute_trajectory_rewards,
    split_trajectories_train_test,
    save_trajectory_split,
    print_trajectory_statistics
)


def main():
    parser = argparse.ArgumentParser(description="Split trajectories into train/test sets")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result folders"
    )
    parser.add_argument(
        "--test-threshold",
        type=float,
        required=True,
        help="Test set threshold (trajectories with avg < threshold)"
    )
    parser.add_argument(
        "--test-count",
        type=int,
        required=True,
        help="Target number of test samples"
    )
    parser.add_argument(
        "--reward-metric",
        type=str,
        default="overall_quality",
        help="Reward metric to use (default: overall_quality)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for split files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="cot_8b",
        help="Filename prefix (default: cot_8b)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("Trajectory-Based Train/Test Split")
    print(f"{'='*70}\n")
    print(f"Results directory: {results_dir}")
    print(f"Test threshold: < {args.test_threshold}")
    print(f"Target test samples: {args.test_count}")
    print(f"Reward metric: {args.reward_metric}")
    print(f"Output directory: {output_dir}")
    print(f"Prefix: {args.prefix}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Compute trajectory rewards
    print("Step 1/3: Computing trajectory rewards...")
    trajectory_rewards, trajectory_sample_scores = compute_trajectory_rewards(
        results_dir,
        reward_metric=args.reward_metric
    )
    
    print(f"✓ Computed rewards for {len(trajectory_rewards)} trajectories")
    print()
    
    # Print statistics
    print_trajectory_statistics(trajectory_rewards, trajectory_sample_scores)
    
    # Split trajectories
    print(f"\n{'='*70}")
    print("Step 2/3: Splitting trajectories...")
    print(f"{'='*70}\n")
    
    train_trajs, test_trajs, train_samples, test_samples = split_trajectories_train_test(
        trajectory_rewards,
        trajectory_sample_scores,
        test_threshold=args.test_threshold,
        test_target_samples=args.test_count,
        seed=args.seed
    )
    
    print(f"✓ Train: {len(train_trajs)} trajectories, {len(train_samples)} samples")
    print(f"✓ Test: {len(test_trajs)} trajectories, {len(test_samples)} samples")
    
    # Verify no overlap
    train_set = set(train_trajs)
    test_set = set(test_trajs)
    overlap = train_set & test_set
    
    if overlap:
        print(f"❌ ERROR: {len(overlap)} trajectories in both train and test!")
        for traj_id in list(overlap)[:10]:
            print(f"  - {traj_id}")
        sys.exit(1)
    else:
        print(f"✓ No overlap: Train and test sets are completely separate")
    
    # Save split
    print(f"\n{'='*70}")
    print("Step 3/3: Saving split files...")
    print(f"{'='*70}\n")
    
    save_trajectory_split(
        output_dir,
        train_trajs,
        test_trajs,
        train_samples,
        test_samples,
        trajectory_rewards,
        trajectory_sample_scores,
        prefix=args.prefix
    )
    
    print(f"\n{'='*70}")
    print("✓ Trajectory Split Complete!")
    print(f"{'='*70}\n")
    
    # Print summary
    test_rewards = [trajectory_rewards[t] for t in test_trajs]
    train_rewards = [trajectory_rewards[t] for t in train_trajs]
    
    print("Summary:")
    print(f"  Train trajectories: {len(train_trajs)}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Train reward range: {min(train_rewards):.2f} - {max(train_rewards):.2f}")
    print(f"  Train reward avg: {sum(train_rewards)/len(train_rewards):.2f}")
    print()
    print(f"  Test trajectories: {len(test_trajs)}")
    print(f"  Test samples: {len(test_samples)}")
    print(f"  Test reward range: {min(test_rewards):.2f} - {max(test_rewards):.2f}")
    print(f"  Test reward avg: {sum(test_rewards)/len(test_rewards):.2f}")
    print()
    
    print(f"Output files ({output_dir}):")
    print(f"  - train_trajectories_{args.prefix}.txt")
    print(f"  - test_trajectories_{args.prefix}.txt")
    print(f"  - train_samples_{args.prefix}.txt")
    print(f"  - test_samples_{args.prefix}.txt")
    print(f"  - trajectory_rewards_{args.prefix}.json")
    print(f"  - trajectory_samples_{args.prefix}.json")
    print()


if __name__ == "__main__":
    main()

