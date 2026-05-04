#!/usr/bin/env python3
"""
Create train/test split with quality-based sampling strategies
Integrated into full pipeline for training data generation
"""

import json
import sys
import argparse
from pathlib import Path
import random
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm


def load_all_samples_with_scores(results_dir: Path):
    """Load all samples and their reward scores"""
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    samples_by_score = defaultdict(list)
    
    for subdir in tqdm(subdirs, desc="Loading samples"):
        reward_file = subdir / "reward_scores.json"
        if reward_file.exists():
            try:
                with open(reward_file) as f:
                    data = json.load(f)
                    if 'scores' in data and 'overall_quality' in data['scores']:
                        score = data['scores']['overall_quality'].get('score')
                        if score is not None:
                            samples_by_score[score].append(subdir.name)
            except Exception as e:
                pass
    
    return samples_by_score


def create_eval_set(samples_by_score: dict, num_eval: int, threshold: float, seed: int):
    """Create evaluation set from low quality samples"""
    random.seed(seed)
    
    # Get all samples below threshold
    low_quality_samples = []
    for score, sample_list in samples_by_score.items():
        if score < threshold:
            low_quality_samples.extend(sample_list)
    
    print(f"\n✓ Found {len(low_quality_samples)} samples with score < {threshold}")
    
    # Sample for eval
    if len(low_quality_samples) < num_eval:
        print(f"⚠️  Warning: Only {len(low_quality_samples)} samples available, requested {num_eval}")
        eval_samples = low_quality_samples
    else:
        eval_samples = random.sample(low_quality_samples, num_eval)
    
    return eval_samples


def create_equal_dist_train_set(samples_by_score: dict, num_train: int, 
                                 eval_samples: list, seed: int):
    """Create training set with equal distribution across quality levels"""
    random.seed(seed + 1)  # Different seed from eval
    
    # Determine how many samples per bucket
    available_scores = sorted(samples_by_score.keys())
    num_buckets = len(available_scores)
    samples_per_bucket = num_train // num_buckets
    
    print(f"\n✓ Equal distribution: {samples_per_bucket} samples per score")
    print(f"  Score buckets: {available_scores}")
    
    train_samples = []
    distribution = {}
    
    eval_set = set(eval_samples)
    
    for score in available_scores:
        # Get samples and exclude eval set
        available = [s for s in samples_by_score[score] if s not in eval_set]
        
        if len(available) < samples_per_bucket:
            print(f"⚠️  Warning: Score {score} has only {len(available)} samples, need {samples_per_bucket}")
            sampled = available
        else:
            sampled = random.sample(available, samples_per_bucket)
        
        train_samples.extend(sampled)
        distribution[score] = len(sampled)
    
    return train_samples, distribution


def load_sample_data(results_dir: Path, sample_id: str):
    """Load full sample data from results directory"""
    sample_dir = results_dir / sample_id
    
    # Load action plan
    action_plan_file = sample_dir / "action_plan.json"
    if not action_plan_file.exists():
        return None
    
    with open(action_plan_file) as f:
        action_plan = json.load(f)
    
    # Load user_prompt from prompt.json (contains edit_info.text)
    prompt_file = sample_dir / "prompt.json"
    user_prompt = ""
    if prompt_file.exists():
        with open(prompt_file) as f:
            prompt_data = json.load(f)
            user_prompt = prompt_data.get("edit_info", {}).get("text", "")
    
    # Load reward scores
    reward_file = sample_dir / "reward_scores.json"
    reward_scores = None
    if reward_file.exists():
        with open(reward_file) as f:
            reward_data = json.load(f)
            reward_scores = reward_data.get('scores', {})
    
    # Construct sample
    sample = {
        "id": sample_id,
        "image_path": f"imageagent_results_16000_cot/{sample_id}/original.png",
        "analysis_path": f"imageagent_results_16000_cot/{sample_id}/analysis.json",
        "user_prompt": user_prompt,  # Now properly loaded from prompt.json
        "target_action_plan": action_plan,
        "metadata": {
            "source_dir": f"imageagent_results_16000_cot/{sample_id}",
            "folder_name": sample_id
        }
    }
    
    if reward_scores:
        sample["reward_score"] = reward_scores.get("overall_quality", {}).get("score")
        sample["reward_scores"] = reward_scores
    
    return sample


def save_training_data(results_dir: Path, sample_ids: list, output_file: Path):
    """Save training data JSON file"""
    samples = []
    
    for sample_id in tqdm(sample_ids, desc="Loading sample data"):
        sample = load_sample_data(results_dir, sample_id)
        if sample:
            samples.append(sample)
    
    output = {
        "version": "3.0",
        "description": "Planner training data with ground truth action plans (equal distribution)",
        "total_samples": len(samples),
        "rl_filtered": False,
        "samples": samples
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {output_file}")


def save_exclusion_list(sample_ids: list, output_file: Path):
    """Save exclusion list (one sample ID per line)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for sample_id in sorted(sample_ids):
            f.write(f"{sample_id}\n")
    
    print(f"✓ Saved exclusion list to {output_file}")


def save_statistics(stats: dict, output_file: Path):
    """Save split statistics"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to {output_file}")


def print_statistics(samples_by_score: dict, eval_samples: list, train_samples: list, 
                     train_distribution: dict, eval_threshold: float):
    """Print comprehensive statistics"""
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT STATISTICS")
    print("=" * 80)
    
    # Overall
    total_samples = sum(len(v) for v in samples_by_score.values())
    print(f"\nTotal available samples: {total_samples}")
    print(f"Evaluation samples: {len(eval_samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Remaining: {total_samples - len(eval_samples) - len(train_samples)}")
    
    # Eval distribution
    eval_dist = Counter()
    for sample in eval_samples:
        for score, sample_list in samples_by_score.items():
            if sample in sample_list:
                eval_dist[score] += 1
                break
    
    print(f"\n📊 EVALUATION SET (threshold < {eval_threshold}):")
    for score in sorted(eval_dist.keys()):
        count = eval_dist[score]
        pct = (count / len(eval_samples)) * 100
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")
    
    eval_scores = [score for score, sample_list in samples_by_score.items() 
                   for sample in eval_samples if sample in sample_list]
    if eval_scores:
        print(f"  Mean: {np.mean(eval_scores):.2f}, Median: {np.median(eval_scores):.2f}")
    
    # Train distribution
    print(f"\n📊 TRAINING SET (equal distribution):")
    for score in sorted(train_distribution.keys()):
        count = train_distribution[score]
        pct = (count / len(train_samples)) * 100
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")
    
    train_scores = [score for score, sample_list in samples_by_score.items() 
                    for sample in train_samples if sample in sample_list]
    if train_scores:
        print(f"  Mean: {np.mean(train_scores):.2f}, Median: {np.median(train_scores):.2f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test split with quality-based sampling"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing imageagent results"
    )
    
    parser.add_argument(
        "--output-base",
        type=str,
        required=True,
        help="Base output directory for training data"
    )
    
    parser.add_argument(
        "--num-train",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000)"
    )
    
    parser.add_argument(
        "--num-eval",
        type=int,
        default=300,
        help="Number of evaluation samples (default: 300)"
    )
    
    parser.add_argument(
        "--equal-dist",
        action="store_true",
        help="Use equal distribution across quality levels for training"
    )
    
    parser.add_argument(
        "--low-quality-eval",
        action="store_true",
        help="Use only low quality samples for evaluation"
    )
    
    parser.add_argument(
        "--low-quality-eval-threshold",
        type=float,
        default=2.0,
        help="Threshold for low quality eval (samples < threshold, default: 2.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_base = Path(args.output_base)
    
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("TRAIN/TEST SPLIT CREATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Results directory: {results_dir}")
    print(f"  Output base: {output_base}")
    print(f"  Training samples: {args.num_train}")
    print(f"  Evaluation samples: {args.num_eval}")
    print(f"  Equal distribution: {args.equal_dist}")
    print(f"  Low quality eval: {args.low_quality_eval}")
    if args.low_quality_eval:
        print(f"  Eval threshold: < {args.low_quality_eval_threshold}")
    print(f"  Random seed: {args.seed}")
    
    # Load all samples with scores
    print("\nStep 1: Loading samples...")
    samples_by_score = load_all_samples_with_scores(results_dir)
    
    total_samples = sum(len(v) for v in samples_by_score.values())
    print(f"✓ Loaded {total_samples} samples")
    print(f"  Score distribution:")
    for score in sorted(samples_by_score.keys()):
        count = len(samples_by_score[score])
        pct = (count / total_samples) * 100
        print(f"    Score {score}: {count:5d} ({pct:5.1f}%)")
    
    # Create evaluation set
    print("\nStep 2: Creating evaluation set...")
    eval_samples = create_eval_set(
        samples_by_score, 
        args.num_eval, 
        args.low_quality_eval_threshold,
        args.seed
    )
    print(f"✓ Created evaluation set with {len(eval_samples)} samples")
    
    # Create training set
    print("\nStep 3: Creating training set with equal distribution...")
    train_samples, train_distribution = create_equal_dist_train_set(
        samples_by_score,
        args.num_train,
        eval_samples,
        args.seed
    )
    print(f"✓ Created training set with {len(train_samples)} samples")
    
    # Print statistics
    print_statistics(
        samples_by_score, 
        eval_samples, 
        train_samples, 
        train_distribution,
        args.low_quality_eval_threshold
    )
    
    # Save files
    print("\nStep 4: Saving files...")
    
    # Save exclusion list
    exclusion_list_path = output_base / "test_samples_cot_8b.txt"
    save_exclusion_list(eval_samples, exclusion_list_path)
    
    # Save training data
    train_output_file = output_base / "standard" / "planner_training_data.json"
    save_training_data(results_dir, train_samples, train_output_file)
    
    # Save statistics
    stats = {
        "total_available": total_samples,
        "num_train": len(train_samples),
        "num_eval": len(eval_samples),
        "equal_dist": args.equal_dist,
        "low_quality_eval": args.low_quality_eval,
        "eval_threshold": args.low_quality_eval_threshold,
        "seed": args.seed,
        "train_distribution": train_distribution,
        "eval_distribution": dict(Counter([
            score for score, sample_list in samples_by_score.items()
            for sample in eval_samples if sample in sample_list
        ]))
    }
    
    stats_file = output_base / "standard" / "split_statistics.json"
    save_statistics(stats, stats_file)
    
    print("\n✅ Split creation complete!")
    print(f"\nOutputs:")
    print(f"  1. Exclusion list: {exclusion_list_path}")
    print(f"  2. Training data: {train_output_file}")
    print(f"  3. Statistics: {stats_file}")


if __name__ == "__main__":
    main()

