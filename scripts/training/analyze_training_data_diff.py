#!/usr/bin/env python3
"""
Analyze differences between Standard and RL training datasets.
Provides detailed comparison of overlap, unique samples, and quality distributions.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import sys

# Try to import matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - skipping visualizations")


class TrainingDataComparator:
    def __init__(self, standard_path: str, rl_path: str, results_dir: str, output_dir: str):
        self.standard_path = Path(standard_path)
        self.rl_path = Path(rl_path)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        print("📂 Loading datasets...")
        with open(self.standard_path) as f:
            self.standard_data = json.load(f)
        with open(self.rl_path) as f:
            self.rl_data = json.load(f)
        
        # Extract samples
        self.standard_samples = {s['id']: s for s in self.standard_data['samples']}
        self.rl_samples = {s['id']: s for s in self.rl_data['samples']}
        
        print(f"   Standard: {len(self.standard_samples)} samples")
        print(f"   RL: {len(self.rl_samples)} samples")
        print()
    
    def find_overlap_and_differences(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """Find overlap and unique samples."""
        standard_ids = set(self.standard_samples.keys())
        rl_ids = set(self.rl_samples.keys())
        
        overlap = standard_ids & rl_ids
        standard_only = standard_ids - rl_ids
        rl_only = rl_ids - standard_ids
        
        return overlap, standard_only, rl_only
    
    def load_reward_scores(self, sample_id: str) -> Dict:
        """Load reward scores for a sample."""
        reward_file = self.results_dir / sample_id / "reward_scores.json"
        if reward_file.exists():
            with open(reward_file) as f:
                data = json.load(f)
                # Extract scores from nested structure
                if 'scores' in data:
                    scores = {}
                    for key, value in data['scores'].items():
                        if isinstance(value, dict) and 'score' in value:
                            scores[key] = value['score']
                        else:
                            scores[key] = value
                    return scores
                return data
        return {}
    
    def analyze_quality_distribution(self, sample_ids: Set[str], label: str) -> Dict:
        """Analyze quality distribution for a set of samples."""
        scores = []
        missing_count = 0
        
        for sample_id in sample_ids:
            reward_data = self.load_reward_scores(sample_id)
            if reward_data and 'overall_quality' in reward_data:
                scores.append(reward_data['overall_quality'])
            else:
                missing_count += 1
        
        if not scores:
            return {
                'label': label,
                'count': len(sample_ids),
                'missing_scores': missing_count,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }
        
        scores.sort()
        n = len(scores)
        
        return {
            'label': label,
            'count': len(sample_ids),
            'missing_scores': missing_count,
            'mean': sum(scores) / n,
            'median': scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2,
            'min': min(scores),
            'max': max(scores),
            'std': (sum((x - sum(scores) / n) ** 2 for x in scores) / n) ** 0.5,
            'scores': scores,
            'distribution': self._get_distribution(scores)
        }
    
    def _get_distribution(self, scores: List[float]) -> Dict:
        """Get score distribution in bins."""
        bins = {
            '1.0-2.0': 0,
            '2.0-3.0': 0,
            '3.0-4.0': 0,
            '4.0-5.0': 0
        }
        
        for score in scores:
            if score < 2.0:
                bins['1.0-2.0'] += 1
            elif score < 3.0:
                bins['2.0-3.0'] += 1
            elif score < 4.0:
                bins['3.0-4.0'] += 1
            else:
                bins['4.0-5.0'] += 1
        
        return bins
    
    def analyze_action_complexity(self, sample_ids: Set[str], dataset: Dict) -> Dict:
        """Analyze action plan complexity."""
        action_counts = []
        
        for sample_id in sample_ids:
            if sample_id in dataset:
                sample = dataset[sample_id]
                num_actions = len(sample['target_action_plan']['actions'])
                action_counts.append(num_actions)
        
        if not action_counts:
            return {'mean': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': sum(action_counts) / len(action_counts),
            'min': min(action_counts),
            'max': max(action_counts),
            'distribution': dict(sorted(
                {k: action_counts.count(k) for k in set(action_counts)}.items()
            ))
        }
    
    def analyze_prompt_characteristics(self, sample_ids: Set[str], dataset: Dict) -> Dict:
        """Analyze prompt characteristics."""
        prompt_lengths = []
        
        for sample_id in sample_ids:
            if sample_id in dataset:
                sample = dataset[sample_id]
                prompt_length = len(sample['user_prompt'])
                prompt_lengths.append(prompt_length)
        
        if not prompt_lengths:
            return {'mean': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': sum(prompt_lengths) / len(prompt_lengths),
            'min': min(prompt_lengths),
            'max': max(prompt_lengths)
        }
    
    def create_visualizations(self, quality_stats: Dict):
        """Create quality distribution visualizations."""
        if not HAS_MATPLOTLIB:
            return
        
        print("📊 Creating visualizations...")
        
        # Quality distribution comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (key, stats) in enumerate([
            ('overlap', quality_stats['overlap']),
            ('standard_only', quality_stats['standard_only']),
            ('rl_only', quality_stats['rl_only'])
        ]):
            if 'scores' in stats and stats['scores']:
                axes[idx].hist(stats['scores'], bins=20, edgecolor='black', alpha=0.7)
                axes[idx].axvline(4.0, color='red', linestyle='--', label='RL Threshold (4.0)')
                axes[idx].set_xlabel('Overall Quality Score')
                axes[idx].set_ylabel('Count')
                axes[idx].set_title(f"{stats['label']}\n(n={len(stats['scores'])}, mean={stats['mean']:.2f})")
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, 'No data available', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f"{stats['label']}\n(no scores)")
        
        plt.tight_layout()
        output_path = self.output_dir / "quality_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: {output_path}")
    
    def generate_report(self, overlap: Set[str], standard_only: Set[str], rl_only: Set[str],
                       quality_stats: Dict, complexity_stats: Dict, prompt_stats: Dict):
        """Generate comprehensive markdown report."""
        print("📝 Generating comparison report...")
        
        report_path = self.output_dir / "COMPARISON_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Training Data Comparison Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"- **Standard Dataset**: {len(self.standard_samples)} samples\n")
            f.write(f"- **RL Dataset**: {len(self.rl_samples)} samples\n")
            f.write(f"- **Overlap**: {len(overlap)} samples ({len(overlap) / len(self.standard_samples) * 100:.1f}%)\n")
            f.write(f"- **Standard Only**: {len(standard_only)} samples\n")
            f.write(f"- **RL Only**: {len(rl_only)} samples\n\n")
            
            # Quality Distribution
            f.write("## Quality Distribution\n\n")
            f.write("| Metric | Overlap | Standard Only | RL Only |\n")
            f.write("|--------|---------|---------------|----------|\n")
            
            for metric in ['count', 'mean', 'median', 'min', 'max', 'std']:
                f.write(f"| {metric.capitalize()} |")
                for key in ['overlap', 'standard_only', 'rl_only']:
                    val = quality_stats[key][metric]
                    if val is None:
                        f.write(" N/A |")
                    elif metric == 'count':
                        f.write(f" {val} |")
                    else:
                        f.write(f" {val:.2f} |")
                f.write("\n")
            
            # Distribution bins
            f.write("\n### Score Distribution by Range\n\n")
            f.write("| Range | Overlap | Standard Only | RL Only |\n")
            f.write("|-------|---------|---------------|----------|\n")
            
            for bin_range in ['1.0-2.0', '2.0-3.0', '3.0-4.0', '4.0-5.0']:
                f.write(f"| {bin_range} |")
                for key in ['overlap', 'standard_only', 'rl_only']:
                    if 'distribution' in quality_stats[key] and quality_stats[key]['distribution']:
                        count = quality_stats[key]['distribution'].get(bin_range, 0)
                        pct = count / quality_stats[key]['count'] * 100 if quality_stats[key]['count'] > 0 else 0
                        f.write(f" {count} ({pct:.1f}%) |")
                    else:
                        f.write(" N/A |")
                f.write("\n")
            
            # Action Complexity
            f.write("\n## Action Plan Complexity\n\n")
            f.write("| Metric | Overlap | Standard Only | RL Only |\n")
            f.write("|--------|---------|---------------|----------|\n")
            
            for metric in ['mean', 'min', 'max']:
                f.write(f"| {metric.capitalize()} |")
                for key in ['overlap', 'standard_only', 'rl_only']:
                    val = complexity_stats[key][metric]
                    f.write(f" {val:.2f} |")
                f.write("\n")
            
            # Prompt Characteristics
            f.write("\n## Prompt Characteristics (Length)\n\n")
            f.write("| Metric | Overlap | Standard Only | RL Only |\n")
            f.write("|--------|---------|---------------|----------|\n")
            
            for metric in ['mean', 'min', 'max']:
                f.write(f"| {metric.capitalize()} |")
                for key in ['overlap', 'standard_only', 'rl_only']:
                    val = prompt_stats[key][metric]
                    f.write(f" {val:.1f} |")
                f.write("\n")
            
            # Key Findings
            f.write("\n## Key Findings\n\n")
            
            # Compare quality scores
            std_only_mean = quality_stats['standard_only']['mean']
            rl_only_mean = quality_stats['rl_only']['mean']
            overlap_mean = quality_stats['overlap']['mean']
            
            if std_only_mean is not None and rl_only_mean is not None:
                f.write(f"### Quality Analysis\n\n")
                f.write(f"1. **Standard-Only samples** have average quality: **{std_only_mean:.2f}**\n")
                
                # Count how many are below threshold
                if 'distribution' in quality_stats['standard_only'] and quality_stats['standard_only']['distribution']:
                    std_only_below_threshold = quality_stats['standard_only']['distribution'].get('1.0-2.0', 0) + \
                                              quality_stats['standard_only']['distribution'].get('2.0-3.0', 0) + \
                                              quality_stats['standard_only']['distribution'].get('3.0-4.0', 0)
                    std_only_pct_below = std_only_below_threshold / len(standard_only) * 100 if len(standard_only) > 0 else 0
                    
                    f.write(f"   - {std_only_below_threshold} samples ({std_only_pct_below:.1f}%) are below RL threshold (4.0)\n")
                    f.write(f"   - These are low-quality samples that Standard model trains on but RL model excludes\n\n")
                else:
                    f.write(f"   - Distribution data not available\n\n")
                
                f.write(f"2. **RL-Only samples** have average quality: **{rl_only_mean:.2f}**\n")
                
                # Count how many are above threshold
                if 'distribution' in quality_stats['rl_only'] and quality_stats['rl_only']['distribution']:
                    rl_only_above_threshold = quality_stats['rl_only']['distribution'].get('4.0-5.0', 0)
                    rl_only_pct_above = rl_only_above_threshold / len(rl_only) * 100 if len(rl_only) > 0 else 0
                    
                    f.write(f"   - {rl_only_above_threshold} samples ({rl_only_pct_above:.1f}%) are above RL threshold (4.0)\n")
                    f.write(f"   - These are high-quality samples that RL model trains on but Standard model misses\n\n")
                else:
                    f.write(f"   - Distribution data not available\n\n")
                
                f.write(f"3. **Overlap samples** have average quality: **{overlap_mean:.2f}**\n")
                f.write(f"   - These samples appear in both datasets\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if std_only_mean is not None and std_only_mean < 4.0:
                f.write(f"1. **Standard dataset includes {len(standard_only)} low-quality samples**\n")
                f.write(f"   - Average quality: {std_only_mean:.2f}\n")
                f.write(f"   - Consider: Use ALL available samples (~2800) instead of limiting to 1555\n")
                f.write(f"   - This would give Standard model more training data\n\n")
            
            if rl_only_mean is not None and rl_only_mean >= 4.0:
                f.write(f"2. **RL dataset correctly filters high-quality samples**\n")
                f.write(f"   - Average quality: {rl_only_mean:.2f}\n")
                f.write(f"   - Threshold of 4.0 appears appropriate\n")
                f.write(f"   - Keep current RL filtering strategy\n\n")
            
            f.write(f"3. **Only {len(overlap)} samples ({len(overlap) / len(self.standard_samples) * 100:.1f}%) overlap**\n")
            f.write(f"   - Standard and RL models are training on largely different data\n")
            f.write(f"   - This is expected: Standard takes first N, RL filters by quality\n\n")
        
        print(f"   ✓ Saved: {report_path}")
    
    def save_sample_lists(self, overlap: Set[str], standard_only: Set[str], rl_only: Set[str]):
        """Save lists of sample IDs to JSON files."""
        print("💾 Saving sample lists...")
        
        # Overlap
        overlap_path = self.output_dir / "overlap_samples.json"
        with open(overlap_path, 'w') as f:
            json.dump({
                'count': len(overlap),
                'sample_ids': sorted(list(overlap))
            }, f, indent=2)
        print(f"   ✓ Saved: {overlap_path}")
        
        # Standard only
        standard_only_path = self.output_dir / "standard_only_samples.json"
        with open(standard_only_path, 'w') as f:
            json.dump({
                'count': len(standard_only),
                'sample_ids': sorted(list(standard_only))
            }, f, indent=2)
        print(f"   ✓ Saved: {standard_only_path}")
        
        # RL only
        rl_only_path = self.output_dir / "rl_only_samples.json"
        with open(rl_only_path, 'w') as f:
            json.dump({
                'count': len(rl_only),
                'sample_ids': sorted(list(rl_only))
            }, f, indent=2)
        print(f"   ✓ Saved: {rl_only_path}")
    
    def run_analysis(self):
        """Run complete analysis."""
        print("🔍 Analyzing training data differences...\n")
        
        # Find overlap and differences
        overlap, standard_only, rl_only = self.find_overlap_and_differences()
        
        print(f"📊 Dataset Breakdown:")
        print(f"   Overlap:       {len(overlap)} samples")
        print(f"   Standard Only: {len(standard_only)} samples")
        print(f"   RL Only:       {len(rl_only)} samples")
        print()
        
        # Analyze quality distribution
        print("📈 Analyzing quality distribution...")
        quality_stats = {
            'overlap': self.analyze_quality_distribution(overlap, "Overlap"),
            'standard_only': self.analyze_quality_distribution(standard_only, "Standard Only"),
            'rl_only': self.analyze_quality_distribution(rl_only, "RL Only")
        }
        
        # Save quality stats (without full scores list for cleaner JSON)
        quality_path = self.output_dir / "quality_distribution.json"
        with open(quality_path, 'w') as f:
            # Remove 'scores' list for cleaner JSON
            clean_stats = {}
            for key, stats in quality_stats.items():
                clean_stats[key] = {k: v for k, v in stats.items() if k != 'scores'}
            json.dump(clean_stats, f, indent=2)
        print(f"   ✓ Saved: {quality_path}")
        print()
        
        # Analyze action complexity
        print("🎯 Analyzing action plan complexity...")
        complexity_stats = {
            'overlap': self.analyze_action_complexity(overlap, self.standard_samples),
            'standard_only': self.analyze_action_complexity(standard_only, self.standard_samples),
            'rl_only': self.analyze_action_complexity(rl_only, self.rl_samples)
        }
        
        complexity_path = self.output_dir / "action_complexity_stats.json"
        with open(complexity_path, 'w') as f:
            json.dump(complexity_stats, f, indent=2)
        print(f"   ✓ Saved: {complexity_path}")
        print()
        
        # Analyze prompt characteristics
        print("📝 Analyzing prompt characteristics...")
        prompt_stats = {
            'overlap': self.analyze_prompt_characteristics(overlap, self.standard_samples),
            'standard_only': self.analyze_prompt_characteristics(standard_only, self.standard_samples),
            'rl_only': self.analyze_prompt_characteristics(rl_only, self.rl_samples)
        }
        print()
        
        # Create visualizations
        if HAS_MATPLOTLIB:
            self.create_visualizations(quality_stats)
            print()
        
        # Save sample lists
        self.save_sample_lists(overlap, standard_only, rl_only)
        print()
        
        # Generate report
        self.generate_report(overlap, standard_only, rl_only, 
                           quality_stats, complexity_stats, prompt_stats)
        print()


def main():
    if len(sys.argv) != 5:
        print("Usage: analyze_training_data_diff.py <standard_json> <rl_json> <results_dir> <output_dir>")
        sys.exit(1)
    
    standard_path = sys.argv[1]
    rl_path = sys.argv[2]
    results_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    comparator = TrainingDataComparator(standard_path, rl_path, results_dir, output_dir)
    comparator.run_analysis()
    
    print("✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

