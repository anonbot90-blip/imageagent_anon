#!/usr/bin/env python3
"""
Compare Teacher-Based Metrics vs GPT-4o Action Quality Scores

This utility script loads both evaluation files and creates a side-by-side
comparison to identify interesting cases:
- Low teacher F1 but high GPT score (student improved over teacher)
- High teacher F1 but low GPT score (both are poor quality)
- High teacher F1 and high GPT score (aligned with good quality)

Usage:
    python scripts/evaluation/compare_teacher_vs_gpt.py \\
        --teacher evaluation_results/text_parallel_eval_40000/standard_text/evaluation_summary_val.json \\
        --gpt evaluation_results/text_parallel_eval_40000/standard_text/gpt4o_action_scores_val.json \\
        --output comparison_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def load_json(path: Path) -> Dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def create_comparison_report(teacher_data: Dict, gpt_data: Dict, output_path: Path):
    """Create markdown comparison report"""
    
    lines = []
    lines.append("# Teacher Metrics vs GPT-4o Quality Comparison")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"**Checkpoint**: `{teacher_data.get('checkpoint', 'N/A')}`")
    lines.append(f"**Samples Evaluated**: {teacher_data.get('num_samples', 0)}")
    lines.append(f"**GPT Samples Evaluated**: {gpt_data.get('num_evaluated', 0)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Extract key metrics
    teacher_metrics = teacher_data.get('aggregated_planner_metrics', {})
    gpt_metrics = gpt_data.get('aggregated_scores', {})
    
    lines.append("## Key Metrics Comparison")
    lines.append("")
    lines.append("### Teacher-Based Metrics (How well it mimics teacher)")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("|--------|------|-----|-----|")
    
    for key in ["planner_action_f1_mean", "planner_action_precision_mean", "planner_action_recall_mean"]:
        if key in teacher_metrics:
            mean_val = teacher_metrics[key]
            min_key = key.replace("_mean", "_min")
            max_key = key.replace("_mean", "_max")
            min_val = teacher_metrics.get(min_key, 0)
            max_val = teacher_metrics.get(max_key, 0)
            metric_name = key.replace("planner_", "").replace("_mean", "")
            lines.append(f"| {metric_name} | {mean_val:.3f} | {min_val:.3f} | {max_val:.3f} |")
    
    lines.append("")
    lines.append("### GPT-4o Quality Metrics (Independent quality assessment)")
    lines.append("")
    lines.append("| Dimension | Mean | Min | Max |")
    lines.append("|-----------|------|-----|-----|")
    
    for dim in ["relevance", "completeness", "efficiency", "correctness", "overall_score"]:
        mean_key = f"{dim}_mean"
        if mean_key in gpt_metrics:
            mean_val = gpt_metrics[mean_key]
            min_val = gpt_metrics.get(f"{dim}_min", 0)
            max_val = gpt_metrics.get(f"{dim}_max", 0)
            lines.append(f"| {dim} | {mean_val:.2f}/10 | {min_val:.2f}/10 | {max_val:.2f}/10 |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    
    # Calculate correlation insight
    action_f1 = teacher_metrics.get("planner_action_f1_mean", 0)
    gpt_overall = gpt_metrics.get("overall_score_mean", 0) / 10  # Normalize to 0-1
    
    lines.append(f"**Teacher F1 Score**: {action_f1:.3f} (How well model copies teacher actions)")
    lines.append(f"**GPT Overall Score**: {gpt_overall:.3f} (Independent quality rating, normalized)")
    lines.append("")
    
    if action_f1 > 0.7 and gpt_overall > 0.7:
        lines.append("✅ **Strong Performance**: Model aligns well with teacher AND produces high-quality plans.")
    elif action_f1 > 0.7 and gpt_overall < 0.6:
        lines.append("⚠️ **Overfitting to Teacher**: Model mimics teacher well, but quality is low. Teacher may be suboptimal.")
    elif action_f1 < 0.6 and gpt_overall > 0.7:
        lines.append("🎯 **Independent Improvement**: Model diverges from teacher but produces high-quality plans. Student may have learned better strategies!")
    elif action_f1 < 0.6 and gpt_overall < 0.6:
        lines.append("❌ **Poor Performance**: Model neither follows teacher well nor produces high-quality plans.")
    else:
        lines.append("📊 **Mixed Performance**: Results show moderate alignment with varying quality.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    
    if action_f1 < 0.6 and gpt_overall > 0.7:
        lines.append("- 🔍 **Review samples** where teacher F1 is low but GPT score is high")
        lines.append("- 💡 Consider using this model's outputs to improve teacher training data")
        lines.append("- 📈 This indicates the student model may have surpassed the teacher")
    elif action_f1 > 0.7 and gpt_overall < 0.6:
        lines.append("- 🔍 **Review teacher training data quality**")
        lines.append("- 💡 Teacher model may need retraining with better data")
        lines.append("- 📊 High F1 but low quality suggests systematic teacher bias")
    else:
        lines.append("- ✅ Model performance is reasonable")
        lines.append("- 📊 Continue monitoring both metrics in future evaluations")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("### Teacher-Based Metrics")
    lines.append("- **Action F1**: Harmonic mean of precision and recall for action selection")
    lines.append("- **Precision**: What fraction of predicted actions match teacher")
    lines.append("- **Recall**: What fraction of teacher actions were predicted")
    lines.append("")
    lines.append("### GPT-4o Quality Dimensions (0-10 scale)")
    lines.append("- **Relevance**: Does the plan address what the user asked for?")
    lines.append("- **Completeness**: Are all necessary edits covered?")
    lines.append("- **Efficiency**: Is the plan well-structured and not overly complex?")
    lines.append("- **Correctness**: Are action parameters reasonable and appropriate?")
    lines.append("- **Overall Score**: Average of the 4 dimensions")
    lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare teacher-based metrics vs GPT-4o quality scores"
    )
    
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Path to teacher-based evaluation summary JSON"
    )
    
    parser.add_argument(
        "--gpt",
        type=str,
        required=True,
        help="Path to GPT-4o action scores JSON"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for comparison report (markdown)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading teacher metrics from: {args.teacher}")
    teacher_data = load_json(Path(args.teacher))
    
    print(f"📂 Loading GPT scores from: {args.gpt}")
    gpt_data = load_json(Path(args.gpt))
    
    # Create report
    print(f"\n📊 Generating comparison report...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_comparison_report(teacher_data, gpt_data, output_path)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

