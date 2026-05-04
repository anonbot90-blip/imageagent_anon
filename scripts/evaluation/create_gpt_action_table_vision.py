#!/usr/bin/env python3
"""
Create GPT-4o Action Judge comparison table for VISION models.
Reads gpt4o_action_scores_val.json from each model and creates a visual comparison table.
Format: Metrics in rows, Models in columns (matching planner_metrics_table.png format)
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_gpt_scores(model_dir: Path) -> dict:
    """Load GPT-4o action scores from a model directory."""
    # First try the standard location
    scores_file = model_dir / "gpt4o_action_scores_all.json"
    if scores_file.exists():
        with open(scores_file, 'r') as f:
            return json.load(f)
    
    # For GPT-4o's own scores, check evaluation_summary_all.json
    summary_file = model_dir / "evaluation_summary_all.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            if 'gpt_action_scores' in summary:
                # Wrap in same structure as other models
                return {'aggregated_scores': summary['gpt_action_scores']}
    
    return None

def create_gpt_action_table(model_dirs: dict, output_path: Path):
    """Create a comparison table of GPT-4o action judge scores."""
    
    # Load scores from all models
    all_scores = {}
    for model_name, model_dir in model_dirs.items():
        scores = load_gpt_scores(Path(model_dir))
        if scores and 'aggregated_scores' in scores:
            all_scores[model_name] = scores['aggregated_scores']
    
    if not all_scores:
        print("⚠️  No GPT-4o action scores found in any model directory")
        return
    
    # Metrics to display (in rows)
    # Action Dimensions (5)
    metrics = [
        ('Relevance', 'relevance', True),
        ('Theme/Style Focus', 'theme_style_focus', True),
        ('Completeness', 'completeness', True),
        ('Efficiency', 'efficiency', True),
        ('Correctness', 'correctness', True),
        (None, None, None),  # Separator
        ('Reasoning Conciseness', 'reasoning_conciseness', True),
        ('Reasoning Completeness', 'reasoning_completeness', True),
        ('Reasoning Specificity', 'reasoning_specificity', True),
        (None, None, None),  # Separator
        ('OVERALL ACTION QUALITY', 'overall_action_quality', True),
        ('OVERALL REASONING QUALITY', 'overall_reasoning_quality', True),
        ('OVERALL SCORE', 'overall_score', True),
    ]
    
    # Model order
    # Model order (updated for 8 models, but Edit-Only will show N/A)
    model_order = ["Baseline", "Edit-Only", "Standard Vision", "RL Vision", "RW Vision", "DPO Vision", "SW Vision", "GPT-4o"]
    
    # Prepare table data: Metrics in rows, Models in columns
    table_data = []
    
    # Header row: Metric | B | E | S | R | RW | D | SW | GPT-4o | Winner
    header = ['Metric', 'B', 'E', 'S', 'R', 'RW', 'D', 'SW', 'GPT-4o', 'Winner']
    table_data.append(header)
    n_cols = len(header)  # Define n_cols here for separator rows
    
    # Track cells to bold (row, col)
    bold_cells = []
    
    # Data rows - one row per metric
    for metric_label, metric_key, higher_is_better in metrics:
        row = [metric_label if metric_label is not None else '']
        
        # Handle separator rows
        if metric_key is None:
            # Add separator dashes
            row.extend(['─' * 4] * (n_cols - 1))
            table_data.append(row)
            continue
        
        # Collect values for all models
        values = []
        for i, model_name in enumerate(model_order):
            # Edit-Only (index 1) has no action plan, so always N/A
            if i == 1:  # Edit-Only
                values.append(None)
            elif model_name in all_scores:
                # Try with _mean suffix first (for other models), then without (for GPT-4o)
                mean_key = f'{metric_key}_mean'
                if mean_key in all_scores[model_name]:
                    values.append(all_scores[model_name][mean_key])
                elif metric_key in all_scores[model_name]:
                    values.append(all_scores[model_name][metric_key])
                else:
                    values.append(None)
            else:
                values.append(None)
        
        # Format values
        for val in values:
            if val is not None:
                row.append(f'{val:.2f}')
            else:
                row.append('N/A')
        
        # Determine winner (highest value) - exclude Edit-Only (index 1)
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None and i != 1]  # Skip Edit-Only
        if valid_values:
            if higher_is_better:
                winner_idx = max(valid_values, key=lambda x: x[1])[0]
            else:
                winner_idx = min(valid_values, key=lambda x: x[1])[0]
            
            winner_abbrevs = ['B', 'E', 'S', 'R', 'RW', 'D', 'SW', 'GPT-4o']
            winner_abbrev = winner_abbrevs[winner_idx]
            row.append(winner_abbrev)
            
            # Bold the winner's cell (row_idx, col_idx)
            # row_idx = len(table_data), col_idx = winner_idx + 1 (skip metric name column)
            bold_cells.append((len(table_data), winner_idx + 1))
        else:
            row.append('-')
        
        table_data.append(row)
    
    # Create figure matching the style of other tables
    n_rows = len(table_data)
    n_cols = len(table_data[0])
    
    col_widths = [0.20, 0.065, 0.065, 0.065, 0.065, 0.07, 0.065, 0.065, 0.09, 0.13]  # Metric, B, E, S, R, RW, D, SW, GPT-4o, Winner
    
    fig_width = sum(col_widths) * 8.5
    fig_height = max(3.5, n_rows * 0.45 + 1.5)  # Extra space for caption
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=col_widths)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    
    # Color header row (matching blue from other tables)
    for i in range(n_cols):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Bold winner cells
    for row, col in bold_cells:
        cell = table[(row, col)]
        cell.set_text_props(weight='bold', color='#2E86AB', fontsize=11)
    
    # Color winner column (last column)
    for i in range(1, n_rows):
        cell = table[(i, n_cols - 1)]
        cell.set_facecolor('#E8F4F8')
        cell.set_text_props(weight='bold', color='#2E86AB')
    
    # Alternate row colors
    for i in range(1, n_rows):
        for j in range(n_cols):
            if j != n_cols - 1:  # Skip winner column
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('white')
    
    # Add title
    plt.title('GPT-4o Action Plan Quality Assessment (Vision Models)', 
              fontsize=14, weight='bold', pad=20)
    
    # Add caption matching the style of other tables
    caption = (
        "Legend: B=Baseline | E=Edit-Only (N/A-no plan) | S=Standard | R=RL | RW=RW | D=DPO | SW=SW | GPT-4o=GPT-4o | "
        "GPT-4o evaluates: 5 Action Dimensions (Relevance, Theme/Style, Completeness, Efficiency, Correctness) + "
        "3 Reasoning Dimensions (Conciseness, Completeness, Specificity) | All scores 0-10 scale | "
        "Bold = Best score"
    )
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, 
            style='italic', color='#555555', wrap=True)
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✅ GPT-4o Action Judge table saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create GPT-4o Action Judge comparison table (Vision)")
    parser.add_argument("--baseline-dir", required=True, help="Baseline model directory")
    parser.add_argument("--edit-only-dir", required=False, help="Edit-Only model directory")
    parser.add_argument("--standard-vision-dir", required=True, help="Standard vision model directory")
    parser.add_argument("--rl-vision-dir", required=True, help="RL vision model directory")
    parser.add_argument("--rw-vision-dir", required=True, help="RW vision model directory")
    parser.add_argument("--dpo-vision-dir", required=True, help="DPO vision model directory")
    parser.add_argument("--sw-vision-dir", required=False, help="SW vision model directory")
    parser.add_argument("--gpt4o-dir", required=False, help="GPT-4o model directory")
    parser.add_argument("--output", required=True, help="Output PNG file path")
    
    args = parser.parse_args()
    
    model_dirs = {
        "Baseline": args.baseline_dir,
        "Standard Vision": args.standard_vision_dir,
        "RL Vision": args.rl_vision_dir,
        "RW Vision": args.rw_vision_dir,
        "DPO Vision": args.dpo_vision_dir
    }
    
    # Add Edit-Only, SW, and GPT-4o if provided
    if args.edit_only_dir:
        model_dirs["Edit-Only"] = args.edit_only_dir
    if args.sw_vision_dir:
        model_dirs["SW Vision"] = args.sw_vision_dir
    if args.gpt4o_dir:
        model_dirs["GPT-4o"] = args.gpt4o_dir
    
    create_gpt_action_table(model_dirs, Path(args.output))

if __name__ == "__main__":
    main()
