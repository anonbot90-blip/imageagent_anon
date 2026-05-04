#!/usr/bin/env python3
"""
Split Complex Theme Dataset into Train/Test Sets (Trajectory-Based)

This script groups samples by trajectory (image_hash + style) and creates
a trajectory-level train/test split to prevent data leakage.

Usage:
    python split_trajectories.py --results-dir imageagent_results_normal_cot \
                                  --output-base training_data/complex_cot_8b_trajectory \
                                  --test-threshold 2.5 \
                                  --test-target-samples 200 \
                                  --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import trajectory_utils
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from trajectory_utils import (
    compute_trajectory_rewards,
    split_trajectories_train_test,
    save_trajectory_split,
    print_trajectory_statistics
)


def main():
    parser = argparse.ArgumentParser(
        description="Split complex theme dataset by trajectories"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result folders (e.g., imageagent_results_normal_cot)"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        required=True,
        help="Base directory for output (e.g., training_data/complex_cot_8b_trajectory)"
    )
    parser.add_argument(
        "--test-threshold",
        type=float,
        default=2.5,
        help="Trajectories with avg reward < threshold go to test set (default: 2.5)"
    )
    parser.add_argument(
        "--test-target-samples",
        type=int,
        default=200,
        help="Target number of samples in test set (default: 200)"
    )
    parser.add_argument(
        "--reward-metric",
        type=str,
        default="overall_quality",
        help="Reward metric to use from reward_scores.json (default: overall_quality)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix (default: auto-detect from output_base, e.g., complex_cot_8b)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    results_dir = Path(args.results_dir)
    output_base = Path(args.output_base)
    
    # Auto-detect prefix from output_base if not provided
    if args.prefix is None:
        # Extract last component (e.g., "complex_cot_8b_trajectory" -> "complex_cot_8b")
        prefix = output_base.name.replace("_trajectory", "")
    else:
        prefix = args.prefix
    
    print("=" * 80)
    print("Complex Theme Dataset - Trajectory-Based Split")
    print("=" * 80)
    print()
    print(f"📂 Results directory: {results_dir}")
    print(f"📂 Output directory: {output_base}")
    print(f"🎯 Test threshold: {args.test_threshold} (trajectories with avg < threshold → test)")
    print(f"🎯 Test target samples: {args.test_target_samples}")
    print(f"📊 Reward metric: {args.reward_metric}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📝 Filename prefix: {prefix}")
    print()
    
    # Validate results directory
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Count total samples
    sample_folders = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("image_")
    ]
    print(f"✓ Found {len(sample_folders)} samples in results directory")
    print()
    
    # Step 1: Compute trajectory rewards
    print("=" * 80)
    print("Step 1: Computing trajectory-level rewards")
    print("=" * 80)
    print()
    
    trajectory_rewards, trajectory_sample_scores = compute_trajectory_rewards(
        results_dir,
        reward_metric=args.reward_metric
    )
    
    print(f"✓ Computed rewards for {len(trajectory_rewards)} trajectories")
    print()
    
    # Print statistics
    print_trajectory_statistics(trajectory_rewards, trajectory_sample_scores)
    
    # Step 2: Split trajectories
    print("=" * 80)
    print("Step 2: Splitting trajectories into train/test")
    print("=" * 80)
    print()
    
    train_trajectories, test_trajectories, train_samples, test_samples = split_trajectories_train_test(
        trajectory_rewards,
        trajectory_sample_scores,
        test_threshold=args.test_threshold,
        test_target_samples=args.test_target_samples,
        seed=args.seed
    )
    
    print(f"✓ Train trajectories: {len(train_trajectories)}")
    print(f"✓ Test trajectories: {len(test_trajectories)}")
    print(f"✓ Train samples: {len(train_samples)}")
    print(f"✓ Test samples: {len(test_samples)}")
    print()
    
    # Compute test set statistics
    test_rewards = [
        trajectory_rewards[traj_id]
        for traj_id in test_trajectories
    ]
    if test_rewards:
        avg_test_reward = sum(test_rewards) / len(test_rewards)
        print(f"📊 Test set avg reward: {avg_test_reward:.3f}")
        print(f"   (All trajectories < {args.test_threshold} threshold)")
    print()
    
    # Step 3: Save split
    print("=" * 80)
    print("Step 3: Saving trajectory split")
    print("=" * 80)
    print()
    
    save_trajectory_split(
        output_base,
        train_trajectories,
        test_trajectories,
        train_samples,
        test_samples,
        trajectory_rewards,
        trajectory_sample_scores,
        prefix=prefix
    )
    
    print()
    print("=" * 80)
    print("✅ Trajectory split complete!")
    print("=" * 80)
    print()
    print(f"📂 Output files:")
    print(f"   - {output_base}/train_trajectories_{prefix}.txt")
    print(f"   - {output_base}/test_trajectories_{prefix}.txt")
    print(f"   - {output_base}/train_samples_{prefix}.txt")
    print(f"   - {output_base}/test_samples_{prefix}.txt")
    print(f"   - {output_base}/trajectory_rewards_{prefix}.json")
    print(f"   - {output_base}/trajectory_samples_{prefix}.json")
    print()
    print(f"🎯 Next steps:")
    print(f"   1. Generate training data using trajectory-based scripts")
    print(f"   2. Train models with trajectory-level strategies")
    print(f"   3. Evaluate on low-quality test trajectories")
    print()


if __name__ == "__main__":
    main()

