#!/usr/bin/env python3
"""
Create a test subset of the complex theme dataset for fast pipeline testing.

This script performs stratified sampling to ensure balanced representation across:
- 10 interior design styles
- 6 quality ranges (to support all training methods: Standard, RL, RW, DPO)

Target: ~150 samples across ~35-40 trajectories
"""

import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

def get_reward(reward_file: Path) -> float:
    """Extract reward score from reward_scores.json"""
    try:
        with open(reward_file) as f:
            data = json.load(f)
            # Try nested structure first
            if "scores" in data and "overall_quality" in data["scores"]:
                return data["scores"]["overall_quality"].get("score", 0)
            # Fall back to flat structure
            return data.get("overall_quality", 0)
    except Exception as e:
        print(f"Warning: Failed to read {reward_file}: {e}")
        return 0.0

def parse_trajectory_id(folder_name: str) -> Tuple[str, str]:
    """
    Parse trajectory ID from folder name.
    Format: image_{hash}_{cycle}_{style}
    Returns: (image_hash, style)
    """
    parts = folder_name.split('_')
    if len(parts) >= 4:
        image_hash = parts[1]
        style = '_'.join(parts[3:])
        return image_hash, style
    return None, None

def categorize_reward(reward: float) -> str:
    """
    Categorize reward into bins for stratified sampling.
    Note: Rewards are typically integers (1-5), so we use integer thresholds.
    """
    if reward >= 5.0:
        return "excellent"  # 5
    elif reward >= 4.0:
        return "high"       # 4
    elif reward >= 3.0:
        return "medium"     # 3
    elif reward >= 2.0:
        return "low"        # 2
    else:
        return "very_low"   # 1

def analyze_full_dataset(results_dir: Path) -> Dict:
    """
    Analyze the full complex dataset.
    
    Returns:
        trajectories: {
            trajectory_id: {
                'style': str,
                'image_hash': str,
                'samples': [(folder_name, reward), ...],
                'avg_reward': float,
                'sample_count': int,
                'quality_bin': str
            }
        }
    """
    print("\n🔍 Step 1/4: Analyzing full dataset...")
    print(f"  Source: {results_dir}")
    
    trajectories = defaultdict(lambda: {
        'samples': [],
        'style': None,
        'image_hash': None
    })
    
    # Scan all folders
    folders = sorted([f for f in results_dir.iterdir() if f.is_dir()])
    print(f"  Found {len(folders)} folders to process")
    
    for folder in tqdm(folders, desc="  Scanning", unit="folder"):
        reward_file = folder / "reward_scores.json"
        if not reward_file.exists():
            continue
        
        # Parse trajectory ID
        image_hash, style = parse_trajectory_id(folder.name)
        if not image_hash or not style:
            continue
        
        # Get reward
        reward = get_reward(reward_file)
        
        # Add to trajectory
        traj_id = f"{image_hash}_{style}"
        trajectories[traj_id]['samples'].append((folder.name, reward))
        trajectories[traj_id]['style'] = style
        trajectories[traj_id]['image_hash'] = image_hash
    
    # Compute trajectory-level statistics
    print("  Computing trajectory statistics...")
    for traj_id, traj_data in trajectories.items():
        rewards = [r for _, r in traj_data['samples']]
        traj_data['avg_reward'] = np.mean(rewards)
        traj_data['sample_count'] = len(rewards)
        traj_data['quality_bin'] = categorize_reward(traj_data['avg_reward'])
    
    print(f"✓ Found {len(trajectories)} unique trajectories")
    print(f"  Total samples: {sum(len(t['samples']) for t in trajectories.values())}")
    
    # Print statistics
    print("\n  Reward Distribution:")
    quality_bins = defaultdict(int)
    for traj_data in trajectories.values():
        quality_bins[traj_data['quality_bin']] += 1
    
    bin_order = ["excellent", "high", "good", "medium", "low", "very_low"]
    for bin_name in bin_order:
        count = quality_bins[bin_name]
        pct = (count / len(trajectories)) * 100 if trajectories else 0
        print(f"    {bin_name:12s}: {count:5d} ({pct:5.1f}%)")
    
    return dict(trajectories)

def stratified_selection(
    trajectories: Dict,
    target_trajectories: int = 40,
    seed: int = 42
) -> List[str]:
    """
    Perform stratified sampling to select trajectories.
    
    Strategy:
    - 10 styles × 4 trajectories = 40 total
    - Per style: distribute across quality bins (1 high, 1 good, 1 medium, 1 low)
    
    Returns:
        List of selected trajectory IDs
    """
    print(f"\n🎯 Step 2/4: Stratified selection (target: {target_trajectories} trajectories)...")
    
    random.seed(seed)
    
    # Group trajectories by style and quality bin
    style_quality_groups = defaultdict(lambda: defaultdict(list))
    styles_set = set()
    
    for traj_id, traj_data in trajectories.items():
        style = traj_data['style']
        quality_bin = traj_data['quality_bin']
        style_quality_groups[style][quality_bin].append(traj_id)
        styles_set.add(style)
    
    styles = sorted(styles_set)
    num_styles = len(styles)
    trajectories_per_style = 5  # Increased to 5 to cover all 5 quality bins
    
    print(f"  Found {num_styles} unique styles")
    print(f"  Target: {trajectories_per_style} trajectories per style (total: {num_styles * trajectories_per_style})")
    
    # Define target distribution per style
    # Based on integer reward bins: excellent(5), high(4), medium(3), low(2), very_low(1)
    # For training methods: DPO chosen (>=4), DPO rejected (2-3), RL (>=3), Test (<2)
    # With 5 trajectories per style, select 1 from each bin for balanced coverage
    target_distribution = {
        "excellent": 1,  # 5 - DPO chosen (high confidence)
        "high": 1,       # 4 - DPO chosen
        "medium": 1,     # 3 - RL/RW, DPO rejected
        "low": 1,        # 2 - DPO rejected
        "very_low": 1    # 1 - Test set
    }
    
    selected_trajectories = []
    
    for style in sorted(styles):
        style_bins = style_quality_groups[style]
        print(f"\n  Style: {style}")
        
        # Calculate how many to select from each bin for this style
        for quality_bin, count in target_distribution.items():
            available = style_bins.get(quality_bin, [])
            
            if count > 0 and available:
                # Select min(target, available)
                select_count = min(count, len(available))
                selected = random.sample(available, select_count)
                selected_trajectories.extend(selected)
                print(f"    {quality_bin:12s}: selected {select_count}/{len(available)} available")
            elif count > 0:
                print(f"    {quality_bin:12s}: wanted {count}, but none available")
    
    print(f"\n✓ Selected {len(selected_trajectories)} trajectories")
    
    # Print selected distribution
    quality_counts = defaultdict(int)
    for traj_id in selected_trajectories:
        quality_counts[trajectories[traj_id]['quality_bin']] += 1
    
    print("\n  Selected distribution:")
    for bin_name in ["excellent", "high", "good", "medium", "low", "very_low"]:
        count = quality_counts[bin_name]
        pct = (count / len(selected_trajectories)) * 100 if selected_trajectories else 0
        print(f"    {bin_name:12s}: {count:3d} ({pct:5.1f}%)")
    
    return selected_trajectories

def copy_selected_data(
    trajectories: Dict,
    selected_traj_ids: List[str],
    source_dir: Path,
    dest_dir: Path
) -> Tuple[int, int]:
    """
    Copy selected trajectory data to test directory.
    
    Returns:
        (num_trajectories_copied, num_samples_copied)
    """
    print(f"\n📁 Step 3/4: Copying selected data...")
    print(f"  Source: {source_dir}")
    print(f"  Destination: {dest_dir}")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    
    for traj_id in tqdm(selected_traj_ids, desc="  Copying", unit="traj"):
        traj_data = trajectories[traj_id]
        
        # Copy all samples from this trajectory
        for folder_name, reward in traj_data['samples']:
            src_folder = source_dir / folder_name
            dst_folder = dest_dir / folder_name
            
            if src_folder.exists() and not dst_folder.exists():
                shutil.copytree(src_folder, dst_folder)
                total_samples += 1
    
    print(f"✓ Copied {len(selected_traj_ids)} trajectories ({total_samples} samples)")
    
    return len(selected_traj_ids), total_samples

def validate_test_dataset(
    trajectories: Dict,
    selected_traj_ids: List[str],
    test_dir: Path
) -> bool:
    """
    Validate that the test dataset meets all requirements.
    
    Returns:
        True if valid, False otherwise
    """
    print("\n✅ Step 4/4: Validating test dataset...")
    
    # Count actual samples
    actual_samples = len([f for f in test_dir.iterdir() if f.is_dir()])
    print(f"  Total samples: {actual_samples}")
    print(f"  Total trajectories: {len(selected_traj_ids)}")
    
    if len(selected_traj_ids) > 0:
        avg_samples = actual_samples / len(selected_traj_ids)
        print(f"  Avg samples per trajectory: {avg_samples:.1f}")
    
    # Check training method compatibility
    print("\n  Training Method Compatibility:")
    
    # Compute stats
    quality_bins = defaultdict(int)
    avg_rewards = []
    
    for traj_id in selected_traj_ids:
        traj_data = trajectories[traj_id]
        quality_bins[traj_data['quality_bin']] += 1
        avg_rewards.append(traj_data['avg_reward'])
    
    # DPO - using integer thresholds
    dpo_chosen = quality_bins["excellent"] + quality_bins["high"]  # scores 4-5
    dpo_rejected = quality_bins["medium"] + quality_bins["low"]  # scores 2-3
    print(f"    DPO Chosen (scores 4-5):   {dpo_chosen} trajectories")
    print(f"    DPO Rejected (scores 2-3): {dpo_rejected} trajectories")
    
    dpo_ok = dpo_chosen >= 10 and dpo_rejected >= 10
    print(f"    DPO pairs: {'✅ OK' if dpo_ok else '❌ INSUFFICIENT'}")
    
    # RL
    rl_eligible = sum(1 for r in avg_rewards if r >= 3.0)
    rl_pct = (rl_eligible / len(avg_rewards)) * 100 if avg_rewards else 0
    print(f"    RL Training (>= 3.0):  {rl_eligible} trajectories ({rl_pct:.1f}%)")
    
    rl_ok = rl_eligible >= 20
    print(f"    RL data: {'✅ OK' if rl_ok else '❌ INSUFFICIENT'}")
    
    # RW
    reward_range = max(avg_rewards) - min(avg_rewards) if avg_rewards else 0
    print(f"    RW Reward range: {min(avg_rewards):.2f} - {max(avg_rewards):.2f} (span: {reward_range:.2f})")
    
    rw_ok = reward_range >= 2.0
    print(f"    RW variance: {'✅ OK' if rw_ok else '❌ INSUFFICIENT'}")
    
    # Test set - score of 1
    test_eligible = quality_bins["very_low"]
    print(f"    Test Set (score 1):        {test_eligible} trajectories")
    
    test_ok = test_eligible >= 3
    print(f"    Test set: {'✅ OK' if test_ok else '❌ INSUFFICIENT'}")
    
    # Overall
    all_ok = dpo_ok and rl_ok and rw_ok and test_ok
    
    print(f"\n  Overall validation: {'✅ PASSED' if all_ok else '❌ FAILED'}")
    
    return all_ok

def main():
    parser = argparse.ArgumentParser(
        description="Create test subset of complex theme dataset with stratified sampling"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="imageagent_results_normal_cot",
        help="Source directory containing full dataset"
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="imageagent_results_normal_cot_test",
        help="Destination directory for test dataset"
    )
    parser.add_argument(
        "--target-trajectories",
        type=int,
        default=50,
        help="Target number of trajectories to select (default: 50 = 10 styles × 5 bins)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform selection without copying data"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    # Validate source directory
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        return 1
    
    print("=" * 70)
    print("  CREATE COMPLEX TEST DATASET")
    print("=" * 70)
    print(f"  Source: {source_dir}")
    print(f"  Destination: {dest_dir}")
    print(f"  Target trajectories: {args.target_trajectories}")
    print(f"  Seed: {args.seed}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)
    
    # Step 1: Analyze full dataset
    trajectories = analyze_full_dataset(source_dir)
    
    if not trajectories:
        print("❌ Error: No trajectories found in source directory")
        return 1
    
    # Step 2: Stratified selection
    selected_traj_ids = stratified_selection(
        trajectories,
        target_trajectories=args.target_trajectories,
        seed=args.seed
    )
    
    if not selected_traj_ids:
        print("❌ Error: No trajectories selected")
        return 1
    
    # Step 3: Copy data (unless dry run)
    if not args.dry_run:
        num_traj, num_samples = copy_selected_data(
            trajectories,
            selected_traj_ids,
            source_dir,
            dest_dir
        )
        
        # Step 4: Validate
        validation_ok = validate_test_dataset(
            trajectories,
            selected_traj_ids,
            dest_dir
        )
        
        print("\n" + "=" * 70)
        if validation_ok:
            print("🎉 SUCCESS! Test dataset created and validated.")
        else:
            print("⚠️  WARNING: Test dataset created but validation failed.")
        print("=" * 70)
        print(f"\nTest dataset location: {dest_dir}")
        print(f"Total trajectories: {num_traj}")
        print(f"Total samples: {num_samples}")
        print("\nNext steps:")
        print(f"  1. Run trajectory split:")
        print(f"     python scripts/training/complex_theme/split_trajectories.py \\")
        print(f"       --results-dir {dest_dir} \\")
        print(f"       --trajectory-prefix complex_cot_8b_test")
        print(f"  2. Test data generation:")
        print(f"     RESULTS_DIR={dest_dir} bash scripts/training/complex_theme/generate_*_trajectory.sh")
        print()
        
        return 0 if validation_ok else 1
    else:
        print("\n" + "=" * 70)
        print("✓ DRY RUN COMPLETE - No data copied")
        print("=" * 70)
        print(f"\nWould select {len(selected_traj_ids)} trajectories")
        print("Run without --dry-run to actually copy data")
        print()
        return 0

if __name__ == "__main__":
    exit(main())

