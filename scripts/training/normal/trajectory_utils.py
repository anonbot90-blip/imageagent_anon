"""
Trajectory Utilities for Trajectory-Based Training

Provides functions for:
- Parsing trajectory IDs from sample folders
- Grouping samples by trajectory
- Computing trajectory-level rewards (average of all samples in trajectory)
- Splitting trajectories into train/test sets

A trajectory represents all samples generated from the same (original_image, prompt/style) pair.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_trajectory_id(sample_folder: str) -> str:
    """
    Extract trajectory ID from sample folder name.
    
    Format: image_<hash>_<number>_<style>
    Trajectory ID: <hash>_<style> (identifies unique original_image + target_style)
    
    The middle number is ignored as it's just a run identifier.
    A trajectory represents all attempts to transform the same original image
    to the same target style.
    
    Args:
        sample_folder: Folder name like "image_6b92e052_3217_renaissance_1500s"
    
    Returns:
        Trajectory ID like "6b92e052_renaissance_1500s"
    
    Example:
        >>> parse_trajectory_id("image_6b92e052_3217_renaissance_1500s")
        "6b92e052_renaissance_1500s"
        >>> parse_trajectory_id("image_6b92e052_9999_renaissance_1500s")
        "6b92e052_renaissance_1500s"  # Same trajectory!
    """
    # Remove "image_" prefix
    if sample_folder.startswith("image_"):
        sample_folder = sample_folder[6:]  # Remove "image_"
    
    # Split by underscore
    parts = sample_folder.split('_')
    
    if len(parts) < 3:
        # Fallback: return as-is
        return sample_folder
    
    # Extract hash (first part) and style (everything after the number)
    hash_part = parts[0]
    style_parts = parts[2:]  # Skip the number in middle
    style = '_'.join(style_parts)
    
    return f"{hash_part}_{style}"


def group_samples_by_trajectory(results_dir: Path) -> Dict[str, List[str]]:
    """
    Group all sample folders by their trajectory ID.
    
    Args:
        results_dir: Directory containing all result folders
    
    Returns:
        Dict mapping trajectory_id -> [sample_folder1, sample_folder2, ...]
    
    Example:
        {
            "6b92e052_renaissance_1500s": [
                "image_6b92e052_3217_renaissance_1500s",
                "image_6b92e052_9999_renaissance_1500s"  # Multiple attempts at same transformation
            ],
            "2ef5c011_autumn_falling": [
                "image_2ef5c011_3803_autumn_falling"
            ],
            ...
        }
    """
    results_dir = Path(results_dir)
    trajectory_groups = defaultdict(list)
    
    # Find all sample folders
    sample_folders = [
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("image_")
    ]
    
    # Group by trajectory
    for sample_folder in sample_folders:
        traj_id = parse_trajectory_id(sample_folder)
        trajectory_groups[traj_id].append(sample_folder)
    
    return dict(trajectory_groups)


def compute_trajectory_rewards(
    results_dir: Path,
    reward_metric: str = "overall_quality"
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
    """
    Compute average reward for each trajectory.
    
    For each trajectory:
    1. Load rewards from all samples in that trajectory
    2. Compute average reward across all samples
    3. Return both trajectory averages and individual sample scores
    
    Args:
        results_dir: Directory containing all result folders
        reward_metric: Reward key to extract from reward_scores.json
    
    Returns:
        Tuple of:
        - trajectory_rewards: Dict[trajectory_id -> avg_reward]
        - trajectory_sample_scores: Dict[trajectory_id -> [(sample_id, score), ...]]
    
    Example:
        trajectory_rewards = {
            "6b92e052_3217_renaissance_1500s": 4.0,  # avg of [5.0, 4.0, 3.0, 4.0]
            "2ef5c011_3803_autumn_falling": 5.0,     # avg of [5.0, 5.0]
            ...
        }
    """
    results_dir = Path(results_dir)
    
    # Group samples by trajectory
    trajectory_groups = group_samples_by_trajectory(results_dir)
    
    trajectory_rewards = {}
    trajectory_sample_scores = {}
    
    for traj_id, sample_folders in trajectory_groups.items():
        sample_scores = []
        
        # Load reward for each sample in this trajectory
        for sample_folder in sample_folders:
            reward_file = results_dir / sample_folder / "reward_scores.json"
            
            if not reward_file.exists():
                print(f"⚠️  Warning: No reward file for {sample_folder}, skipping")
                continue
            
            try:
                with open(reward_file, 'r') as f:
                    reward_data = json.load(f)
                    
                    # Handle both old and new reward file formats
                    # New format: {"scores": {"overall_quality": {"score": 5}}}
                    # Old format: {"overall_quality": 5}
                    score = None
                    
                    if "scores" in reward_data and reward_metric in reward_data["scores"]:
                        # New format
                        score = reward_data["scores"][reward_metric].get("score", None)
                    elif reward_metric in reward_data:
                        # Old format
                        score = reward_data.get(reward_metric, None)
                    
                    if score is not None:
                        sample_scores.append((sample_folder, float(score)))
                    else:
                        print(f"⚠️  Warning: No {reward_metric} found in {sample_folder}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load reward for {sample_folder}: {e}")
                continue
        
        # Compute trajectory average
        if sample_scores:
            avg_reward = sum(score for _, score in sample_scores) / len(sample_scores)
            trajectory_rewards[traj_id] = avg_reward
            trajectory_sample_scores[traj_id] = sample_scores
        else:
            print(f"⚠️  Warning: No valid samples for trajectory {traj_id}")
    
    return trajectory_rewards, trajectory_sample_scores


def filter_trajectories_by_reward(
    trajectory_rewards: Dict[str, float],
    min_threshold: float = None,
    max_threshold: float = None
) -> List[str]:
    """
    Filter trajectories by reward range.
    
    Args:
        trajectory_rewards: Dict[trajectory_id -> avg_reward]
        min_threshold: Minimum reward (inclusive), None means no minimum
        max_threshold: Maximum reward (inclusive), None means no maximum
    
    Returns:
        List of trajectory IDs that meet the criteria
    
    Example:
        >>> filter_trajectories_by_reward(rewards, min_threshold=3.0)
        ["6b92e052_3217_renaissance_1500s", "2ef5c011_3803_autumn_falling", ...]
    """
    filtered = []
    
    for traj_id, avg_reward in trajectory_rewards.items():
        if min_threshold is not None and avg_reward < min_threshold:
            continue
        if max_threshold is not None and avg_reward > max_threshold:
            continue
        filtered.append(traj_id)
    
    return filtered


def split_trajectories_train_test(
    trajectory_rewards: Dict[str, float],
    trajectory_sample_scores: Dict[str, List[Tuple[str, float]]],
    test_threshold: float,
    test_target_samples: int,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split trajectories into train/test sets based on quality.
    
    Test set: Low-quality trajectories (avg < test_threshold)
    Train set: All other trajectories
    
    Args:
        trajectory_rewards: Dict[trajectory_id -> avg_reward]
        trajectory_sample_scores: Dict[trajectory_id -> [(sample_id, score), ...]]
        test_threshold: Threshold for test set (avg < threshold)
        test_target_samples: Target number of samples in test set (~200)
        seed: Random seed
    
    Returns:
        Tuple of (train_trajectory_ids, test_trajectory_ids, train_sample_ids, test_sample_ids)
    
    Example:
        train_trajs = ["6b92e052_3217_renaissance_1500s", ...]
        test_trajs = ["0a5e6d90_120_urban_city", ...]
        train_samples = ["image_6b92e052_3217_renaissance_1500s", ...]
        test_samples = ["image_0a5e6d90_120_urban_city", ...]
    """
    random.seed(seed)
    
    # Separate trajectories by quality
    test_candidates = [
        traj_id for traj_id, avg_reward in trajectory_rewards.items()
        if avg_reward < test_threshold
    ]
    
    # Shuffle test candidates
    random.shuffle(test_candidates)
    
    # Select test trajectories to reach target sample count
    test_trajectories = []
    test_sample_count = 0
    
    for traj_id in test_candidates:
        samples = trajectory_sample_scores.get(traj_id, [])
        test_trajectories.append(traj_id)
        test_sample_count += len(samples)
        
        # Stop when we reach target
        if test_sample_count >= test_target_samples:
            break
    
    # All other trajectories go to train
    test_set = set(test_trajectories)
    train_trajectories = [
        traj_id for traj_id in trajectory_rewards.keys()
        if traj_id not in test_set
    ]
    
    # Get all sample IDs for train/test
    train_samples = []
    for traj_id in train_trajectories:
        samples = trajectory_sample_scores.get(traj_id, [])
        train_samples.extend([sample_id for sample_id, _ in samples])
    
    test_samples = []
    for traj_id in test_trajectories:
        samples = trajectory_sample_scores.get(traj_id, [])
        test_samples.extend([sample_id for sample_id, _ in samples])
    
    return train_trajectories, test_trajectories, train_samples, test_samples


def save_trajectory_split(
    output_dir: Path,
    train_trajectories: List[str],
    test_trajectories: List[str],
    train_samples: List[str],
    test_samples: List[str],
    trajectory_rewards: Dict[str, float],
    trajectory_sample_scores: Dict[str, List[Tuple[str, float]]],
    prefix: str = "cot_8b"
):
    """
    Save trajectory split to disk.
    
    Creates:
    - train_trajectories_{prefix}.txt: List of train trajectory IDs
    - test_trajectories_{prefix}.txt: List of test trajectory IDs
    - train_samples_{prefix}.txt: List of train sample IDs
    - test_samples_{prefix}.txt: List of test sample IDs
    - trajectory_rewards_{prefix}.json: Trajectory ID -> avg reward
    - trajectory_samples_{prefix}.json: Trajectory ID -> [sample IDs]
    
    Args:
        output_dir: Directory to save files
        train_trajectories: List of train trajectory IDs
        test_trajectories: List of test trajectory IDs
        train_samples: List of train sample IDs
        test_samples: List of test sample IDs
        trajectory_rewards: Dict[trajectory_id -> avg_reward]
        trajectory_sample_scores: Dict[trajectory_id -> [(sample_id, score), ...]]
        prefix: Filename prefix (e.g., "cot_8b")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectory lists
    with open(output_dir / f"train_trajectories_{prefix}.txt", 'w') as f:
        f.write('\n'.join(train_trajectories))
    
    with open(output_dir / f"test_trajectories_{prefix}.txt", 'w') as f:
        f.write('\n'.join(test_trajectories))
    
    # Save sample lists
    with open(output_dir / f"train_samples_{prefix}.txt", 'w') as f:
        f.write('\n'.join(train_samples))
    
    with open(output_dir / f"test_samples_{prefix}.txt", 'w') as f:
        f.write('\n'.join(test_samples))
    
    # Save trajectory rewards
    with open(output_dir / f"trajectory_rewards_{prefix}.json", 'w') as f:
        json.dump(trajectory_rewards, f, indent=2)
    
    # Save trajectory -> samples mapping
    trajectory_samples_map = {
        traj_id: [sample_id for sample_id, _ in samples]
        for traj_id, samples in trajectory_sample_scores.items()
    }
    
    with open(output_dir / f"trajectory_samples_{prefix}.json", 'w') as f:
        json.dump(trajectory_samples_map, f, indent=2)
    
    print(f"✓ Saved trajectory split to {output_dir}")
    print(f"  - Train: {len(train_trajectories)} trajectories, {len(train_samples)} samples")
    print(f"  - Test: {len(test_trajectories)} trajectories, {len(test_samples)} samples")


def compute_standardized_trajectory_weights(
    trajectory_rewards: Dict[str, float],
    epsilon: float = 1e-18
) -> Dict[str, float]:
    """
    Standardize trajectory rewards using z-score normalization.
    
    Computes standardized weights for trajectories based on their average rewards.
    Each trajectory gets a weight = (reward - mean) / (std + epsilon)
    
    This allows trajectories to be weighted by their relative quality:
    - Above-average trajectories get positive weights
    - Below-average trajectories get negative weights
    - Weights follow a standard normal distribution
    
    Args:
        trajectory_rewards: Dict[trajectory_id -> avg_reward]
        epsilon: Small constant for numerical stability (default: 1e-18)
    
    Returns:
        Dict[trajectory_id -> standardized_weight]
    
    Example:
        >>> trajectory_rewards = {
        ...     "traj_a": 4.6,  # High quality
        ...     "traj_b": 3.5,  # Average
        ...     "traj_c": 2.2   # Low quality
        ... }
        >>> weights = compute_standardized_trajectory_weights(trajectory_rewards)
        >>> # traj_a gets positive weight (above mean)
        >>> # traj_b gets ~0 weight (near mean)
        >>> # traj_c gets negative weight (below mean)
    """
    import numpy as np
    
    if not trajectory_rewards:
        print("⚠️  Warning: Empty trajectory_rewards dict")
        return {}
    
    # Extract all trajectory average rewards
    reward_values = list(trajectory_rewards.values())
    
    # Compute global statistics
    mean_reward = np.mean(reward_values)
    std_reward = np.std(reward_values)
    
    print(f"\n{'='*70}")
    print("Standardized Weight Statistics")
    print(f"{'='*70}\n")
    print(f"Trajectory reward distribution:")
    print(f"  Mean: {mean_reward:.4f}")
    print(f"  Std:  {std_reward:.4f}")
    print(f"  Min:  {min(reward_values):.4f}")
    print(f"  Max:  {max(reward_values):.4f}")
    
    # Standardize each trajectory's reward
    standardized_weights = {}
    
    for traj_id, avg_reward in trajectory_rewards.items():
        # z-score: (x - mean) / std
        standardized_weight = (avg_reward - mean_reward) / (std_reward + epsilon)
        standardized_weights[traj_id] = standardized_weight
    
    # Print weight distribution statistics
    weight_values = list(standardized_weights.values())
    print(f"\nStandardized weight distribution:")
    print(f"  Mean: {np.mean(weight_values):.4f} (should be ~0)")
    print(f"  Std:  {np.std(weight_values):.4f} (should be ~1)")
    print(f"  Min:  {min(weight_values):.4f}")
    print(f"  Max:  {max(weight_values):.4f}")
    
    # Count by z-score ranges
    positive_count = sum(1 for w in weight_values if w > 0)
    negative_count = sum(1 for w in weight_values if w < 0)
    strong_positive = sum(1 for w in weight_values if w > 1.0)
    strong_negative = sum(1 for w in weight_values if w < -1.0)
    
    total = len(weight_values)
    print(f"\nWeight distribution by z-score:")
    print(f"  Strong positive (>+1.0): {strong_positive} ({strong_positive/total*100:.1f}%)")
    print(f"  Positive (0 to +1.0):    {positive_count - strong_positive} ({(positive_count - strong_positive)/total*100:.1f}%)")
    print(f"  Negative (-1.0 to 0):    {negative_count - strong_negative} ({(negative_count - strong_negative)/total*100:.1f}%)")
    print(f"  Strong negative (<-1.0): {strong_negative} ({strong_negative/total*100:.1f}%)")
    print()
    
    return standardized_weights


def print_trajectory_statistics(
    trajectory_rewards: Dict[str, float],
    trajectory_sample_scores: Dict[str, List[Tuple[str, float]]]
):
    """
    Print statistics about trajectory distribution.
    
    Args:
        trajectory_rewards: Dict[trajectory_id -> avg_reward]
        trajectory_sample_scores: Dict[trajectory_id -> [(sample_id, score), ...]]
    """
    total_trajectories = len(trajectory_rewards)
    total_samples = sum(len(samples) for samples in trajectory_sample_scores.values())
    
    # Samples per trajectory distribution
    samples_per_traj = [len(samples) for samples in trajectory_sample_scores.values()]
    
    # Trajectory reward distribution
    reward_bins = {
        "1.0-2.0": 0,
        "2.0-3.0": 0,
        "3.0-4.0": 0,
        "4.0-5.0": 0
    }
    
    for avg_reward in trajectory_rewards.values():
        if avg_reward < 2.0:
            reward_bins["1.0-2.0"] += 1
        elif avg_reward < 3.0:
            reward_bins["2.0-3.0"] += 1
        elif avg_reward < 4.0:
            reward_bins["3.0-4.0"] += 1
        else:
            reward_bins["4.0-5.0"] += 1
    
    print(f"\n{'='*70}")
    print("Trajectory Statistics")
    print(f"{'='*70}\n")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Total samples: {total_samples}")
    
    if total_trajectories > 0:
        print(f"Avg samples per trajectory: {total_samples / total_trajectories:.2f}")
    else:
        print("⚠️  No trajectories found!")
        return
    print(f"\nSamples per trajectory (all trajectories have 1 sample in this dataset):")
    print(f"  1 sample: {samples_per_traj.count(1)} trajectories")
    print(f"\nTrajectory reward distribution (averages):")
    for bin_range, count in reward_bins.items():
        pct = (count / total_trajectories * 100) if total_trajectories > 0 else 0
        # Calculate total samples in this bin
        total_in_bin = sum(
            len(trajectory_sample_scores[traj_id])
            for traj_id, avg_reward in trajectory_rewards.items()
            if (bin_range == "1.0-2.0" and avg_reward < 2.0) or
               (bin_range == "2.0-3.0" and 2.0 <= avg_reward < 3.0) or
               (bin_range == "3.0-4.0" and 3.0 <= avg_reward < 4.0) or
               (bin_range == "4.0-5.0" and avg_reward >= 4.0)
        )
        print(f"  {bin_range}: {count} trajectories ({pct:.1f}%), {total_in_bin} samples")
    print()

