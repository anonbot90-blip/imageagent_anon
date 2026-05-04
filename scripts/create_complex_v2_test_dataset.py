#!/usr/bin/env python3
"""
Create a test subset of the complex V2 dataset for fast pipeline testing.

This script performs stratified sampling to ensure balanced representation across:
- Multiple themes/subjects (discovered from data)
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

def parse_trajectory_id(folder_name: str) -> Tuple[str, str, str]:
    """
    Parse trajectory ID from V2 folder name.
    Format: image_{hash}_v2_l{level}_{id}_{theme}_{variation}_{type}
    Example: image_00094d2f_v2_l3_0182_dragon_art_movement_multi
    
    Returns: (image_hash, theme_variation, full_theme)
    """
    parts = folder_name.split('_')
    
    # Parse: image_{hash}_v2_l{level}_{id}_{...theme...}_{type}
    if len(parts) < 6 or parts[2] != 'v2':
        return None, None, None
    
    try:
        image_hash = parts[1]  # e.g., "00094d2f"
        # Skip parts[2] = "v2", parts[3] = "l3", parts[4] = "0182"
        # Everything from index 5 to second-to-last is the theme + variation
        # Last part is the type (dual/triple/multi/complex)
        
        theme_parts = parts[5:-1]  # All middle parts are theme + variation
        theme_variation = '_'.join(theme_parts)  # e.g., "dragon_art_movement"
        
        # For grouping, we might want just the main theme (first word)
        main_theme = parts[5] if len(parts) > 5 else "unknown"
        
        return image_hash, theme_variation, main_theme
    except Exception as e:
        print(f"Warning: Failed to parse {folder_name}: {e}")
        return None, None, None

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
    Analyze the full complex V2 dataset.
    
    Returns:
        trajectories: {
            trajectory_id: {
                'theme_variation': str,
                'main_theme': str,
                'image_hash': str,
                'samples': [(folder_name, reward), ...],
                'avg_reward': float,
                'sample_count': int,
                'quality_bin': str
            }
        }
    """
    print("\n🔍 Step 1/4: Analyzing full V2 dataset...")
    print(f"  Source: {results_dir}")
    
    trajectories = defaultdict(lambda: {
        'samples': [],
        'theme_variation': None,
        'main_theme': None,
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
        image_hash, theme_variation, main_theme = parse_trajectory_id(folder.name)
        if not image_hash or not theme_variation:
            continue
        
        # Get reward
        reward = get_reward(reward_file)
        
        # Add to trajectory (group by hash + theme_variation)
        traj_id = f"{image_hash}_{theme_variation}"
        trajectories[traj_id]['samples'].append((folder.name, reward))
        trajectories[traj_id]['theme_variation'] = theme_variation
        trajectories[traj_id]['main_theme'] = main_theme
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
    
    bin_order = ["excellent", "high", "medium", "low", "very_low"]
    for bin_name in bin_order:
        count = quality_bins[bin_name]
        pct = (count / len(trajectories)) * 100 if trajectories else 0
        print(f"    {bin_name:12s}: {count:5d} ({pct:5.1f}%)")
    
    # Print theme distribution
    print("\n  Theme Distribution (top 20):")
    theme_counts = defaultdict(int)
    for traj_data in trajectories.values():
        theme_counts[traj_data['main_theme']] += 1
    
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"    {theme:20s}: {count:4d}")
    
    return dict(trajectories)

def stratified_selection(
    trajectories: Dict,
    target_trajectories: int = 50,
    seed: int = 42
) -> List[str]:
    """
    Perform stratified sampling to select trajectories.
    
    Strategy:
    - Select trajectories across quality bins to ensure training method compatibility
    - Balance across main themes when possible
    
    Returns:
        List of selected trajectory IDs
    """
    print(f"\n🎯 Step 2/4: Stratified selection (target: {target_trajectories} trajectories)...")
    
    random.seed(seed)
    
    # Group trajectories by main_theme and quality bin
    theme_quality_groups = defaultdict(lambda: defaultdict(list))
    themes_set = set()
    
    for traj_id, traj_data in trajectories.items():
        main_theme = traj_data['main_theme']
        quality_bin = traj_data['quality_bin']
        theme_quality_groups[main_theme][quality_bin].append(traj_id)
        themes_set.add(main_theme)
    
    themes = sorted(themes_set)
    num_themes = len(themes)
    
    print(f"  Found {num_themes} unique main themes")
    
    # Define target distribution for quality bins
    # Based on training method requirements:
    # - DPO chosen (>=4): excellent + high
    # - DPO rejected (2-3): medium + low
    # - RL training (>=3): excellent + high + medium
    # - Test set (<2): very_low
    target_distribution = {
        "excellent": 0.15,  # 15%
        "high": 0.25,       # 25% - DPO chosen
        "medium": 0.25,     # 25% - RL/RW, DPO rejected
        "low": 0.20,        # 20% - DPO rejected
        "very_low": 0.15    # 15% - Test set
    }
    
    # Calculate how many trajectories to select from each quality bin
    bin_targets = {}
    for quality_bin, proportion in target_distribution.items():
        bin_targets[quality_bin] = int(target_trajectories * proportion)
    
    # Adjust to ensure sum equals target
    total_allocated = sum(bin_targets.values())
    if total_allocated < target_trajectories:
        # Add extras to "high" and "medium" (most useful for training)
        bin_targets["high"] += (target_trajectories - total_allocated) // 2
        bin_targets["medium"] += (target_trajectories - total_allocated + 1) // 2
    
    print(f"\n  Target distribution:")
    for quality_bin in ["excellent", "high", "medium", "low", "very_low"]:
        count = bin_targets.get(quality_bin, 0)
        print(f"    {quality_bin:12s}: {count:3d} trajectories")
    
    selected_trajectories = []
    
    # Select from each quality bin, balancing across themes
    for quality_bin, target_count in bin_targets.items():
        # Get all trajectories in this quality bin across all themes
        candidates = []
        for theme in themes:
            candidates.extend(theme_quality_groups[theme].get(quality_bin, []))
        
        # Randomly select target_count trajectories
        if len(candidates) >= target_count:
            selected = random.sample(candidates, target_count)
        else:
            selected = candidates  # Take all if not enough
            print(f"    Warning: Only {len(candidates)} trajectories available for {quality_bin} (wanted {target_count})")
        
        selected_trajectories.extend(selected)
        print(f"    {quality_bin:12s}: selected {len(selected)}/{len(candidates)} available")
    
    print(f"\n✓ Selected {len(selected_trajectories)} trajectories")
    
    # Print selected distribution
    quality_counts = defaultdict(int)
    theme_counts = defaultdict(int)
    for traj_id in selected_trajectories:
        quality_counts[trajectories[traj_id]['quality_bin']] += 1
        theme_counts[trajectories[traj_id]['main_theme']] += 1
    
    print("\n  Selected quality distribution:")
    for bin_name in ["excellent", "high", "medium", "low", "very_low"]:
        count = quality_counts[bin_name]
        pct = (count / len(selected_trajectories)) * 100 if selected_trajectories else 0
        print(f"    {bin_name:12s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n  Selected theme distribution (top 15):")
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"    {theme:20s}: {count:3d}")
    
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
    if avg_rewards:
        reward_range = max(avg_rewards) - min(avg_rewards)
        print(f"    RW Reward range: {min(avg_rewards):.2f} - {max(avg_rewards):.2f} (span: {reward_range:.2f})")
        rw_ok = reward_range >= 2.0
    else:
        print(f"    RW Reward range: N/A")
        rw_ok = False
    
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
        description="Create test subset of complex V2 dataset with stratified sampling"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="imageagent_results_complex_v2_10k_cot",
        help="Source directory containing full V2 dataset"
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="imageagent_results_complex_v2_10k_cot_test",
        help="Destination directory for test dataset"
    )
    parser.add_argument(
        "--target-trajectories",
        type=int,
        default=50,
        help="Target number of trajectories to select (default: 50)"
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
    print("  CREATE COMPLEX V2 TEST DATASET")
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
        print(f"  1. Run trajectory split (if needed)")
        print(f"  2. Test data generation pipeline")
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

