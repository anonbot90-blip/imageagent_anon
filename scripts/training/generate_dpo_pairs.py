#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) Training Data

Creates preference pairs (chosen, rejected) for the same prompt/image:
- Chosen: High-quality plans (reward >= 4.5)
- Rejected: Mediocre plans (reward 2.5-3.5)

For DPO training, we need pairs where both plans are for the SAME input
but have different quality levels.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Set random seed for reproducibility
random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_reward_scores(results_dir: Path, reward_metric: str, excluded_ids: set = None) -> Dict[str, float]:
    """Load reward scores for all samples, excluding test samples."""
    scores = {}
    excluded_count = 0
    
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        
        # Skip if in exclusion list (test samples)
        if excluded_ids and folder.name in excluded_ids:
            excluded_count += 1
            continue
        
        reward_file = folder / "reward_scores.json"
        if not reward_file.exists():
            continue
        
        try:
            with open(reward_file, 'r') as f:
                data = json.load(f)
            
            if reward_metric in data["scores"]:
                score = data["scores"][reward_metric]["score"]
                scores[folder.name] = score
        except Exception as e:
            print(f"⚠️  Warning: Failed to load {reward_file}: {e}")
    
    if excluded_count > 0:
        print(f"🚫 Excluded {excluded_count} test samples from scoring")
    
    return scores


def create_preference_pairs(
    results_dir: Path,
    scores: Dict[str, float],
    chosen_threshold: float,
    rejected_min: float,
    rejected_max: float
) -> List[Tuple[str, str, float, float]]:
    """
    Create preference pairs (chosen_id, rejected_id, chosen_score, rejected_score).
    
    Strategy:
    1. Get all high-quality samples (chosen >= threshold)
    2. Get all low-quality samples (rejected in range)
    3. Pair each chosen with a rejected sample (with shuffling for diversity)
    """
    pairs = []
    
    # Get all chosen and rejected candidates
    chosen_candidates = [(sid, score) for sid, score in scores.items() if score >= chosen_threshold]
    rejected_candidates = [(sid, score) for sid, score in scores.items() 
                          if rejected_min <= score <= rejected_max]
    
    if not chosen_candidates:
        print(f"⚠️  No samples found with score >= {chosen_threshold}")
        return pairs
    
    if not rejected_candidates:
        print(f"⚠️  No samples found with score in range [{rejected_min}, {rejected_max}]")
        return pairs
    
    # Shuffle rejected candidates for diversity
    random.shuffle(rejected_candidates)
    
    # Create pairs: each chosen gets paired with a rejected
    # If more chosen than rejected, cycle through rejected list
    for i, (chosen_id, chosen_score) in enumerate(chosen_candidates):
        rejected_id, rejected_score = rejected_candidates[i % len(rejected_candidates)]
        
        # Ensure significant quality gap (at least 1.0 point difference)
        if chosen_score - rejected_score >= 1.0:
            pairs.append((chosen_id, rejected_id, chosen_score, rejected_score))
    
    return pairs


def create_dpo_dataset(
    results_dir: Path,
    pairs: List[Tuple[str, str, float, float]],
    output_path: Path
) -> None:
    """Create DPO training dataset from preference pairs."""
    
    dpo_samples = []
    skipped = 0
    
    for chosen_id, rejected_id, chosen_score, rejected_score in pairs:
        chosen_folder = results_dir / chosen_id
        rejected_folder = results_dir / rejected_id
        
        # Load chosen sample
        try:
            with open(chosen_folder / "action_plan.json", 'r') as f:
                chosen_plan = json.load(f)
            
            with open(rejected_folder / "action_plan.json", 'r') as f:
                rejected_plan = json.load(f)
            
            # Get image path and prompt (should be same for both)
            chosen_original = chosen_folder / "original.png"
            rejected_original = rejected_folder / "original.png"
            
            if not chosen_original.exists() or not rejected_original.exists():
                skipped += 1
                continue
            
            # Create DPO sample
            sample = {
                "id": f"{chosen_id}_vs_{rejected_id}",
                "image_path": str(chosen_original.relative_to(PROJECT_ROOT)),
                "user_prompt": chosen_plan["overall_instruction"],
                "chosen_plan": chosen_plan,
                "rejected_plan": rejected_plan,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_margin": chosen_score - rejected_score,
                "metadata": {
                    "chosen_id": chosen_id,
                    "rejected_id": rejected_id,
                    "chosen_folder": str(chosen_folder.relative_to(PROJECT_ROOT)),
                    "rejected_folder": str(rejected_folder.relative_to(PROJECT_ROOT))
                }
            }
            
            dpo_samples.append(sample)
            
        except Exception as e:
            print(f"⚠️  Skipping pair ({chosen_id}, {rejected_id}): {e}")
            skipped += 1
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "total_samples": len(dpo_samples),
        "samples": dpo_samples,
        "metadata": {
            "format": "dpo_preference_pairs",
            "description": "DPO training data with (chosen, rejected) preference pairs",
            "skipped_samples": skipped
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Saved {len(dpo_samples)} DPO pairs to {output_path}")
    print(f"  Skipped: {skipped} pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs from imageagent results"
    )
    
    parser.add_argument(
        "results_dir",
        help="Directory containing imageagent results"
    )
    
    parser.add_argument(
        "--output-dir-text",
        default="./training_data/dpo_text",
        help="Output directory for text-only DPO data"
    )
    
    parser.add_argument(
        "--output-dir-vision",
        default="./training_data/dpo_vision",
        help="Output directory for vision DPO data"
    )
    
    parser.add_argument(
        "--chosen-threshold",
        type=float,
        default=4.5,
        help="Minimum score for chosen samples (default: 4.5)"
    )
    
    parser.add_argument(
        "--rejected-min",
        type=float,
        default=2.5,
        help="Minimum score for rejected samples (default: 2.5)"
    )
    
    parser.add_argument(
        "--rejected-max",
        type=float,
        default=3.5,
        help="Maximum score for rejected samples (default: 3.5)"
    )
    
    parser.add_argument(
        "--reward-metric",
        type=str,
        default="overall_quality",
        help="Reward metric to use (default: overall_quality)"
    )
    
    parser.add_argument(
        "--exclude-file",
        type=str,
        default=None,
        help="Path to file containing sample IDs to exclude (e.g., test_samples.txt)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return 1
    
    # Load excluded sample IDs if provided
    excluded_ids = set()
    if args.exclude_file:
        exclude_path = Path(args.exclude_file)
        if exclude_path.exists():
            with open(exclude_path, 'r') as f:
                excluded_ids = set(line.strip() for line in f if line.strip())
    
    print("=" * 70)
    print("DPO Preference Pairs Generation")
    print("=" * 70)
    print()
    print(f"Results directory: {results_dir}")
    print(f"Chosen threshold: >= {args.chosen_threshold}")
    print(f"Rejected range: {args.rejected_min} - {args.rejected_max}")
    print(f"Reward metric: {args.reward_metric}")
    if excluded_ids:
        print(f"Excluding: {len(excluded_ids)} test samples")
    print()
    
    # Load reward scores
    print("Loading reward scores...")
    scores = load_reward_scores(results_dir, args.reward_metric, excluded_ids)
    print(f"✓ Loaded scores for {len(scores)} samples")
    
    # Show distribution
    chosen_count = sum(1 for s in scores.values() if s >= args.chosen_threshold)
    rejected_count = sum(1 for s in scores.values() if args.rejected_min <= s <= args.rejected_max)
    print(f"  Chosen candidates (>= {args.chosen_threshold}): {chosen_count}")
    print(f"  Rejected candidates ({args.rejected_min}-{args.rejected_max}): {rejected_count}")
    
    # Create preference pairs
    print("\nCreating preference pairs...")
    pairs = create_preference_pairs(
        results_dir,
        scores,
        args.chosen_threshold,
        args.rejected_min,
        args.rejected_max
    )
    print(f"✓ Created {len(pairs)} preference pairs")
    
    if len(pairs) == 0:
        print("\n❌ Error: No valid preference pairs found!")
        print("   Try adjusting thresholds or check reward scores distribution")
        return 1
    
    # Show statistics
    print("\nPair Statistics:")
    chosen_scores = [cs for _, _, cs, _ in pairs]
    rejected_scores = [rs for _, _, _, rs in pairs]
    margins = [cs - rs for _, _, cs, rs in pairs]
    
    print(f"  Chosen scores:   {min(chosen_scores):.2f} - {max(chosen_scores):.2f} (avg: {sum(chosen_scores)/len(chosen_scores):.2f})")
    print(f"  Rejected scores: {min(rejected_scores):.2f} - {max(rejected_scores):.2f} (avg: {sum(rejected_scores)/len(rejected_scores):.2f})")
    print(f"  Score margins:   {min(margins):.2f} - {max(margins):.2f} (avg: {sum(margins)/len(margins):.2f})")
    
    # Create datasets (same pairs for both text and vision)
    print("\nGenerating datasets...")
    
    # Text-only dataset
    output_text = Path(args.output_dir_text) / "planner_training_data_dpo.json"
    create_dpo_dataset(results_dir, pairs, output_text)
    
    # Vision dataset (same data, will use cached embeddings)
    output_vision = Path(args.output_dir_vision) / "planner_training_data_dpo.json"
    create_dpo_dataset(results_dir, pairs, output_vision)
    
    print("\n" + "=" * 70)
    print("✅ DPO Preference Pairs Generation Complete")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

