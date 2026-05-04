#!/usr/bin/env python3
"""
Create a test subset of the simple 10K CoT dataset for fast pipeline testing.

This script performs stratified sampling to ensure balanced representation across:
- Diverse themes (60+ scene/style themes)
- 5 quality ranges (to support all training methods: Standard, RL, RW, DPO, SW)

Target: ~200 samples across ~50 trajectories

Critical for training methods:
- DPO Chosen: >= 4.0 (need ~35% of trajectories)
- DPO Rejected: 2.5-3.5 (need ~40% of trajectories)
- RL Training: >= 3.0 (need ~75% of trajectories)
- RW Weighting: Full range (all quality levels)
- Test Set: < 2.5 (need ~10% of trajectories)
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
    """Extract average reward score from reward_scores.json"""
    try:
        with open(reward_file) as f:
            data = json.load(f)
            # Calculate average across all reward dimensions
            if "scores" in data:
                scores = data["scores"]
                reward_values = []
                for key, value in scores.items():
                    if isinstance(value, dict) and "score" in value:
                        reward_values.append(value["score"])
                return np.mean(reward_values) if reward_values else 0.0
            return 0.0
    except Exception as e:
        print(f"Warning: Failed to read {reward_file}: {e}")
        return 0.0

def parse_trajectory_id(folder_name: str) -> Tuple[str, str]:
    """
    Parse trajectory ID from folder name.
    Format: image_{hash}_{id}_{theme}
    Returns: (image_hash, theme)
    """
    parts = folder_name.split('_')
    if len(parts) >= 4:
        image_hash = parts[1]  # e.g., '0001f629'
        theme = '_'.join(parts[3:])  # e.g., 'western_house'
        return image_hash, theme
    return None, None

def categorize_reward(reward: float) -> str:
    """
    Categorize reward into bins for stratified sampling.
    These bins align with training method requirements.
    """
    if reward >= 4.5:
        return "excellent"  # >= 4.5 - DPO chosen (high confidence)
    elif reward >= 4.0:
        return "high"       # 4.0-4.5 - DPO chosen
    elif reward >= 3.5:
        return "good"       # 3.5-4.0 - RL training, RW high weight
    elif reward >= 3.0:
        return "medium"     # 3.0-3.5 - DPO rejected, RL training
    elif reward >= 2.5:
        return "low"        # 2.5-3.0 - DPO rejected
    else:
        return "very_low"   # < 2.5 - Test set exclusion

def analyze_full_dataset(results_dir: Path) -> Dict:
    """
    Analyze the full simple 10K dataset.
    
    Returns:
        trajectories: {
            trajectory_id: {
                'theme': str,
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
        'theme': None,
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
        image_hash, theme = parse_trajectory_id(folder.name)
        if not image_hash or not theme:
            continue
        
        # Get reward
        reward = get_reward(reward_file)
        
        # Add to trajectory
        traj_id = f"{image_hash}_{theme}"
        trajectories[traj_id]['samples'].append((folder.name, reward))
        trajectories[traj_id]['theme'] = theme
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
    print("\n  Reward Distribution (Trajectory-level):")
    quality_bins = defaultdict(int)
    for traj_data in trajectories.values():
        quality_bins[traj_data['quality_bin']] += 1
    
    bin_order = ["excellent", "high", "good", "medium", "low", "very_low"]
    for bin_name in bin_order:
        count = quality_bins[bin_name]
        pct = (count / len(trajectories)) * 100 if trajectories else 0
        print(f"    {bin_name:12s}: {count:5d} ({pct:5.1f}%)")
    
    # Print theme diversity
    themes = set(t['theme'] for t in trajectories.values())
    print(f"\n  Theme diversity: {len(themes)} unique themes")
    
    return dict(trajectories)

def stratified_selection(
    trajectories: Dict,
    target_trajectories: int = 50,
    seed: int = 42
) -> List[str]:
    """
    Perform stratified sampling to select trajectories.
    
    Strategy:
    - Quality-first stratification (not theme-based like complex)
    - Target distribution to support all training methods:
      * Excellent (>=4.5): 15% = 7-8 trajectories
      * High (4.0-4.5): 20% = 10 trajectories
      * Good (3.5-4.0): 15% = 7-8 trajectories  
      * Medium (3.0-3.5): 25% = 12-13 trajectories
      * Low (2.5-3.0): 15% = 7-8 trajectories
      * Very Low (<2.5): 10% = 5 trajectories
    
    Returns:
        List of selected trajectory IDs
    """
    print(f"\n🎯 Step 2/4: Stratified selection (target: {target_trajectories} trajectories)...")
    
    random.seed(seed)
    
    # Group trajectories by quality bin
    quality_groups = defaultdict(list)
    
    for traj_id, traj_data in trajectories.items():
        quality_bin = traj_data['quality_bin']
        quality_groups[quality_bin].append(traj_id)
    
    # Define target distribution (scale to target_trajectories)
    distribution_percentages = {
        "excellent": 0.15,   # 15% - DPO chosen (high confidence)
        "high": 0.20,        # 20% - DPO chosen
        "good": 0.15,        # 15% - RL/RW
        "medium": 0.25,      # 25% - DPO rejected, RL/RW
        "low": 0.15,         # 15% - DPO rejected
        "very_low": 0.10     # 10% - Test set
    }
    
    target_distribution = {
        "excellent": int(target_trajectories * distribution_percentages["excellent"]),
        "high": int(target_trajectories * distribution_percentages["high"]),
        "good": int(target_trajectories * distribution_percentages["good"]),
        "medium": int(target_trajectories * distribution_percentages["medium"]),
        "low": int(target_trajectories * distribution_percentages["low"]),
        "very_low": int(target_trajectories * distribution_percentages["very_low"])
    }
    
    selected_trajectories = []
    
    print("\n  Selection by quality bin:")
    for quality_bin, target_count in target_distribution.items():
        available = quality_groups.get(quality_bin, [])
        
        if target_count > 0 and available:
            # Select min(target, available)
            select_count = min(target_count, len(available))
            selected = random.sample(available, select_count)
            selected_trajectories.extend(selected)
            
            print(f"    {quality_bin:12s}: {select_count:3d} / {target_count:3d} (available: {len(available)})")
        else:
            print(f"    {quality_bin:12s}:   0 / {target_count:3d} (available: {len(available)})")
    
    print(f"\n✓ Selected {len(selected_trajectories)} trajectories")
    
    # Verify training method compatibility
    print("\n  Training Method Compatibility Check:")
    selected_trajs_data = {tid: trajectories[tid] for tid in selected_trajectories}
    
    dpo_chosen = sum(1 for t in selected_trajs_data.values() if t['avg_reward'] >= 4.0)
    dpo_rejected = sum(1 for t in selected_trajs_data.values() if 2.5 <= t['avg_reward'] < 3.5)
    rl_eligible = sum(1 for t in selected_trajs_data.values() if t['avg_reward'] >= 3.0)
    test_set = sum(1 for t in selected_trajs_data.values() if t['avg_reward'] < 2.5)
    
    print(f"    DPO Chosen (>=4.0):       {dpo_chosen:3d} ({dpo_chosen/len(selected_trajectories)*100:.1f}%) - Need ~35%")
    print(f"    DPO Rejected (2.5-3.5):   {dpo_rejected:3d} ({dpo_rejected/len(selected_trajectories)*100:.1f}%) - Need ~40%")
    print(f"    RL Eligible (>=3.0):      {rl_eligible:3d} ({rl_eligible/len(selected_trajectories)*100:.1f}%) - Need ~75%")
    print(f"    Test Set (<2.5):          {test_set:3d} ({test_set/len(selected_trajectories)*100:.1f}%) - Need ~10%")
    
    # Check if requirements are met
    all_good = True
    if dpo_chosen < 15:
        print(f"    ⚠️  WARNING: DPO chosen count too low ({dpo_chosen} < 15)")
        all_good = False
    if dpo_rejected < 17:
        print(f"    ⚠️  WARNING: DPO rejected count too low ({dpo_rejected} < 17)")
        all_good = False
    if rl_eligible < 35:
        print(f"    ⚠️  WARNING: RL eligible count too low ({rl_eligible} < 35)")
        all_good = False
    if test_set < 4:
        print(f"    ⚠️  WARNING: Test set count too low ({test_set} < 4)")
        all_good = False
    
    if all_good:
        print("    ✅ All training method requirements met!")
    
    return selected_trajectories

def copy_selected_data(
    selected_trajectory_ids: List[str],
    trajectories: Dict,
    source_dir: Path,
    output_dir: Path
) -> None:
    """
    Copy selected trajectory samples to output directory.
    """
    print(f"\n📁 Step 3/4: Copying selected data...")
    print(f"  Destination: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    for traj_id in tqdm(selected_trajectory_ids, desc="  Copying trajectories"):
        traj_data = trajectories[traj_id]
        for folder_name, reward in traj_data['samples']:
            src = source_dir / folder_name
            dst = output_dir / folder_name
            
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                total_samples += 1
    
    print(f"✓ Copied {total_samples} samples from {len(selected_trajectory_ids)} trajectories")

def generate_report(
    selected_trajectory_ids: List[str],
    trajectories: Dict,
    output_dir: Path
) -> None:
    """
    Generate a report summarizing the test dataset.
    """
    print(f"\n📊 Step 4/4: Generating report...")
    
    selected_trajs = {tid: trajectories[tid] for tid in selected_trajectory_ids}
    
    report = {
        "metadata": {
            "total_trajectories": len(selected_trajectory_ids),
            "total_samples": sum(len(t['samples']) for t in selected_trajs.values()),
            "seed": 42
        },
        "quality_distribution": {},
        "training_method_compatibility": {},
        "theme_diversity": {},
        "trajectories": []
    }
    
    # Quality distribution
    quality_bins = defaultdict(int)
    for traj_data in selected_trajs.values():
        quality_bins[traj_data['quality_bin']] += 1
    
    for bin_name, count in sorted(quality_bins.items()):
        pct = (count / len(selected_trajectory_ids)) * 100
        report["quality_distribution"][bin_name] = {
            "count": count,
            "percentage": round(pct, 1)
        }
    
    # Training method compatibility
    dpo_chosen = sum(1 for t in selected_trajs.values() if t['avg_reward'] >= 4.0)
    dpo_rejected = sum(1 for t in selected_trajs.values() if 2.5 <= t['avg_reward'] < 3.5)
    rl_eligible = sum(1 for t in selected_trajs.values() if t['avg_reward'] >= 3.0)
    test_set = sum(1 for t in selected_trajs.values() if t['avg_reward'] < 2.5)
    
    report["training_method_compatibility"] = {
        "dpo_chosen": {"count": dpo_chosen, "threshold": ">=4.0"},
        "dpo_rejected": {"count": dpo_rejected, "threshold": "2.5-3.5"},
        "rl_eligible": {"count": rl_eligible, "threshold": ">=3.0"},
        "test_set": {"count": test_set, "threshold": "<2.5"}
    }
    
    # Theme diversity
    themes = defaultdict(int)
    for traj_data in selected_trajs.values():
        themes[traj_data['theme']] += 1
    
    report["theme_diversity"] = {
        "unique_themes": len(themes),
        "theme_counts": dict(sorted(themes.items(), key=lambda x: x[1], reverse=True)[:20])
    }
    
    # Trajectory details
    for traj_id in sorted(selected_trajectory_ids):
        traj_data = trajectories[traj_id]
        report["trajectories"].append({
            "trajectory_id": traj_id,
            "image_hash": traj_data['image_hash'],
            "theme": traj_data['theme'],
            "sample_count": traj_data['sample_count'],
            "avg_reward": round(traj_data['avg_reward'], 2),
            "quality_bin": traj_data['quality_bin']
        })
    
    # Save report
    report_file = output_dir / "test_dataset_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("  📋 TEST DATASET SUMMARY")
    print("="*70)
    print(f"  Trajectories: {len(selected_trajectory_ids)}")
    print(f"  Samples: {report['metadata']['total_samples']}")
    print(f"  Unique themes: {report['theme_diversity']['unique_themes']}")
    print(f"\n  Quality Distribution:")
    for bin_name, data in sorted(report["quality_distribution"].items()):
        print(f"    {bin_name:12s}: {data['count']:3d} ({data['percentage']:5.1f}%)")
    print(f"\n  Training Method Compatibility:")
    for method, data in report["training_method_compatibility"].items():
        print(f"    {method:15s}: {data['count']:3d}")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Create simple 10K CoT test dataset")
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path('imageagent_results_10000_cot'),
        help='Source dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('imageagent_results_10000_cot_test'),
        help='Output test dataset directory'
    )
    parser.add_argument(
        '--target-trajectories',
        type=int,
        default=50,
        help='Target number of trajectories to select'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  🎯 CREATE SIMPLE 10K COT TEST DATASET")
    print("="*70)
    print(f"  Source: {args.source_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Target: {args.target_trajectories} trajectories (~200 samples)")
    print(f"  Seed: {args.seed}")
    print("="*70)
    
    # Step 1: Analyze full dataset
    trajectories = analyze_full_dataset(args.source_dir)
    
    # Step 2: Stratified selection
    selected_trajectory_ids = stratified_selection(
        trajectories,
        target_trajectories=args.target_trajectories,
        seed=args.seed
    )
    
    # Step 3: Copy selected data
    copy_selected_data(
        selected_trajectory_ids,
        trajectories,
        args.source_dir,
        args.output_dir
    )
    
    # Step 4: Generate report
    generate_report(
        selected_trajectory_ids,
        trajectories,
        args.output_dir
    )
    
    print("\n✅ Test dataset creation complete!")
    print(f"   Output: {args.output_dir}")
    print(f"   Report: {args.output_dir}/test_dataset_report.json")

if __name__ == "__main__":
    main()

