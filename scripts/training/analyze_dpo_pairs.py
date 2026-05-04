#!/usr/bin/env python3
"""
Analyze DPO Preference Pairs

Validates and visualizes the quality of DPO training data.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter


def analyze_dpo_pairs(data_path: Path):
    """Analyze DPO preference pairs dataset."""
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    samples = data["samples"]
    total = data["total_samples"]
    
    print("=" * 70)
    print(f"DPO Pairs Analysis: {data_path.name}")
    print("=" * 70)
    print()
    print(f"Total pairs: {total}")
    print()
    
    # Extract scores
    chosen_scores = [s["chosen_score"] for s in samples]
    rejected_scores = [s["rejected_score"] for s in samples]
    margins = [s["score_margin"] for s in samples]
    
    # Statistics
    print("Score Statistics:")
    print(f"  Chosen:   min={min(chosen_scores):.2f}, max={max(chosen_scores):.2f}, "
          f"mean={np.mean(chosen_scores):.2f}, std={np.std(chosen_scores):.2f}")
    print(f"  Rejected: min={min(rejected_scores):.2f}, max={max(rejected_scores):.2f}, "
          f"mean={np.mean(rejected_scores):.2f}, std={np.std(rejected_scores):.2f}")
    print(f"  Margins:  min={min(margins):.2f}, max={max(margins):.2f}, "
          f"mean={np.mean(margins):.2f}, std={np.std(margins):.2f}")
    print()
    
    # Quality checks
    print("Quality Checks:")
    
    # Check 1: All margins should be positive
    negative_margins = sum(1 for m in margins if m <= 0)
    if negative_margins > 0:
        print(f"  ⚠️  WARNING: {negative_margins} pairs have non-positive margins!")
    else:
        print(f"  ✓ All pairs have positive margins")
    
    # Check 2: Sufficient margin (>= 1.0)
    good_margins = sum(1 for m in margins if m >= 1.0)
    print(f"  ✓ {good_margins}/{total} pairs have margin >= 1.0 ({100*good_margins/total:.1f}%)")
    
    # Check 3: Score distribution
    chosen_high_quality = sum(1 for s in chosen_scores if s >= 4.5)
    rejected_mediocre = sum(1 for s in rejected_scores if 2.5 <= s <= 3.5)
    print(f"  ✓ {chosen_high_quality}/{total} chosen are high-quality (>= 4.5)")
    print(f"  ✓ {rejected_mediocre}/{total} rejected are mediocre (2.5-3.5)")
    print()
    
    # Visualizations
    output_dir = data_path.parent
    
    # 1. Score distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(chosen_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_xlabel('Chosen Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Chosen Scores Distribution')
    axes[0].axvline(np.mean(chosen_scores), color='red', linestyle='--', label=f'Mean: {np.mean(chosen_scores):.2f}')
    axes[0].legend()
    
    axes[1].hist(rejected_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Rejected Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Rejected Scores Distribution')
    axes[1].axvline(np.mean(rejected_scores), color='red', linestyle='--', label=f'Mean: {np.mean(rejected_scores):.2f}')
    axes[1].legend()
    
    axes[2].hist(margins, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[2].set_xlabel('Score Margin (Chosen - Rejected)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Score Margins Distribution')
    axes[2].axvline(np.mean(margins), color='red', linestyle='--', label=f'Mean: {np.mean(margins):.2f}')
    axes[2].legend()
    
    plt.tight_layout()
    output_path = output_dir / "dpo_score_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved score distributions to: {output_path}")
    
    # 2. Scatter plot: Chosen vs Rejected
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(rejected_scores, chosen_scores, alpha=0.5, c=margins, 
                        cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (y=x)
    min_score = min(min(rejected_scores), min(chosen_scores))
    max_score = max(max(rejected_scores), max(chosen_scores))
    ax.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.5, label='y=x (no preference)')
    
    ax.set_xlabel('Rejected Score')
    ax.set_ylabel('Chosen Score')
    ax.set_title('Chosen vs Rejected Scores\n(Points above diagonal = valid preferences)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Score Margin')
    
    plt.tight_layout()
    output_path = output_dir / "dpo_chosen_vs_rejected.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved scatter plot to: {output_path}")
    
    # 3. Sample some pairs for inspection
    print()
    print("Sample Pairs (showing 3 examples):")
    print("-" * 70)
    
    for i, sample in enumerate(samples[:3]):
        print(f"\nPair {i+1}:")
        print(f"  Prompt: {sample['user_prompt'][:80]}...")
        print(f"  Chosen score:   {sample['chosen_score']:.2f}")
        print(f"  Rejected score: {sample['rejected_score']:.2f}")
        print(f"  Margin:         {sample['score_margin']:.2f}")
        print(f"  Chosen actions:   {len(sample['chosen_plan']['actions'])} actions")
        print(f"  Rejected actions: {len(sample['rejected_plan']['actions'])} actions")
    
    print()
    print("=" * 70)
    print("✅ Analysis Complete")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze DPO preference pairs")
    parser.add_argument("data_path", help="Path to DPO training data JSON file")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Error: File not found: {data_path}")
        return 1
    
    analyze_dpo_pairs(data_path)
    return 0


if __name__ == "__main__":
    exit(main())

