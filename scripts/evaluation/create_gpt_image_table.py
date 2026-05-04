#!/usr/bin/env python3
"""
Create GPT-4o Image Quality comparison table from evaluation results.
Reads detailed_results_all.json from each model and creates a visual comparison table.
Format: Metrics in rows, Models in columns (matching other tables)
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_image_scores(model_dir: Path) -> dict:
    """Load GPT-4o image scores from a model directory."""
    # Try both all and test detailed results files (Edit-Only uses test)
    detailed_file = model_dir / "detailed_results_all.json"
    if not detailed_file.exists():
        detailed_file = model_dir / "detailed_results_test.json"
    
    if not detailed_file.exists():
        return None
    
    with open(detailed_file, 'r') as f:
        detailed_results = json.load(f)
    
    # Extract GPT image judge scores from all samples
    all_scores = []
    for sample in detailed_results:
        # Check if this is Edit-Only format (has reward_scores instead of image_metrics)
        if 'reward_scores' in sample and 'scores' in sample['reward_scores']:
            # Edit-Only format: extract from reward_scores
            scores = sample['reward_scores']['scores']
            gpt_scores = {
                'instruction_following': scores.get('adherence_to_prompt', {}).get('score'),
                'visual_quality': scores.get('final_image_quality', {}).get('score'),
                'transformation_strength': None,  # Not evaluated for Edit-Only
                'coherence': None,  # Not evaluated for Edit-Only
                'semantic_accuracy': None,  # Not evaluated for Edit-Only
                'technical_execution': None,  # Not evaluated for Edit-Only
                'overall_image_score': scores.get('overall_quality', {}).get('score')
            }
            # Only add if we have actual scores (not None/0)
            if any(v and v > 0 for v in [gpt_scores['instruction_following'], 
                                          gpt_scores['visual_quality'], 
                                          gpt_scores['overall_image_score']]):
                all_scores.append(gpt_scores)
        elif 'image_metrics' in sample:
            # Standard format: extract from image_metrics
            metrics = sample['image_metrics']
            # Extract GPT judge scores (all have gpt_judge_ prefix)
            gpt_scores = {
                'instruction_following': metrics.get('gpt_judge_instruction_following'),
                'visual_quality': metrics.get('gpt_judge_visual_quality'),
                'transformation_strength': metrics.get('gpt_judge_transformation_strength'),
                'coherence': metrics.get('gpt_judge_coherence'),
                'semantic_accuracy': metrics.get('gpt_judge_semantic_accuracy'),
                'technical_execution': metrics.get('gpt_judge_technical_execution'),
                'overall_image_score': metrics.get('gpt_judge_overall')
            }
            # Only add if we have actual scores (not None/0)
            if any(v and v > 0 for v in gpt_scores.values()):
                all_scores.append(gpt_scores)
    
    if not all_scores:
        return None
    
    # Aggregate scores
    aggregated = {}
    for key in all_scores[0].keys():
        values = [s[key] for s in all_scores if s.get(key) and s[key] > 0]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    aggregated['num_evaluated'] = len(all_scores)
    return aggregated

def create_gpt_image_table(model_dirs: dict, output_path: Path):
    """Create a comparison table of GPT-4o image quality scores."""
    
    # Load scores from all models
    all_scores = {}
    for model_name, model_dir in model_dirs.items():
        scores = load_image_scores(Path(model_dir))
        if scores:
            all_scores[model_name] = scores
    
    if not all_scores:
        print("⚠️  No GPT-4o image scores found in any model directory")
        return
    
    # Metrics to display (in rows)
    metrics = [
        ('Instruction Following', 'instruction_following', True),
        ('Visual Quality', 'visual_quality', True),
        ('Transformation Strength', 'transformation_strength', True),
        ('Coherence', 'coherence', True),
        ('Semantic Accuracy', 'semantic_accuracy', True),
        ('Technical Execution', 'technical_execution', True),
        ('─' * 25, None, None),  # Separator
        ('OVERALL IMAGE SCORE', 'overall_image_score', True),
    ]
    
    # Model order (updated for 8 models)
    model_order = ["Baseline", "Edit-Only", "Standard", "RL", "RW", "DPO", "SW", "GPT-4o"]
    
    # Prepare table data: Metrics in rows, Models in columns
    table_data = []
    
    # Header row: Metric | B | E | S | R | RW | D | SW | GPT-4o | Winner
    header = ['Metric', 'B', 'E', 'S', 'R', 'RW', 'D', 'SW', 'GPT-4o', 'Winner']
    table_data.append(header)
    
    # Track cells to bold (row, col)
    bold_cells = []
    
    # Get number of columns from header
    n_cols = len(header)
    
    # Plan-related metrics (Edit-Only doesn't have these, show N/A)
    plan_related_metrics = ['transformation_strength', 'coherence', 'semantic_accuracy', 'technical_execution']
    
    # Data rows - one row per metric
    for metric_label, metric_key, higher_is_better in metrics:
        row = [metric_label]
        
        # Handle separator rows
        if metric_key is None:
            # Create separator row spanning all columns
            row.extend(['─' * 4] * (n_cols - 1))
            table_data.append(row)
            continue
        
        # Check if this is a plan-related metric
        is_plan_metric = metric_key in plan_related_metrics
        
        # Collect values for all models
        values = []
        for i, model_name in enumerate(model_order):
            # Edit-Only (index 1) doesn't have plan-related metrics
            if is_plan_metric and i == 1:  # Edit-Only
                values.append(None)
            elif model_name in all_scores:
                mean_key = f'{metric_key}_mean'
                if mean_key in all_scores[model_name]:
                    values.append(all_scores[model_name][mean_key])
                else:
                    values.append(None)
            else:
                values.append(None)
        
        # Format values
        for i, val in enumerate(values):
            if val is not None:
                row.append(f'{val:.2f}')
            else:
                # Show N/A for Edit-Only on plan-related metrics
                if is_plan_metric and i == 1:  # Edit-Only
                    row.append('N/A')
                else:
                    row.append('-')
        
        # Determine winner (highest value) - exclude Edit-Only for plan-related metrics
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        # For plan metrics, exclude Edit-Only
        if is_plan_metric:
            valid_values = [(i, v) for i, v in valid_values if i != 1]
        
        if valid_values:
            if higher_is_better:
                winner_idx = max(valid_values, key=lambda x: x[1])[0]
            else:
                winner_idx = min(valid_values, key=lambda x: x[1])[0]
            
            winner_abbrev = ['B', 'E', 'S', 'R', 'RW', 'D', 'SW', 'GPT-4o'][winner_idx]
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
    
    col_widths = [0.15, 0.065, 0.065, 0.065, 0.065, 0.07, 0.065, 0.065, 0.09, 0.13]  # Metric, B, E, S, R, RW, D, SW, GPT-4o, Winner
    
    fig_width = sum(col_widths) * 9
    fig_height = max(4.5, n_rows * 0.45 + 1.5)  # Extra space for title and caption
    
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
    
    # Alternate row colors and handle separator rows
    for i in range(1, n_rows):
        # Check if this is a separator row
        is_separator = '─' in str(table_data[i][0])
        
        for j in range(n_cols):
            if j != n_cols - 1:  # Skip winner column
                if is_separator:
                    # Special styling for separator row
                    table[(i, j)].set_facecolor('#D0D0D0')
                    table[(i, j)].set_text_props(weight='bold', fontsize=9)
                elif i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('white')
    
    # Add title
    plt.title('GPT-4o Image Quality Assessment (Text Models)', 
              fontsize=14, weight='bold', pad=20)
    
    # Add caption matching the style of other tables
    caption = (
        "Legend: B=Baseline | S=Standard | R=RL | RW=RW | D=DPO | SW=SW | GPT-4o=GPT-4o | "
        "GPT-4o evaluates: 6 image quality dimensions (0-10 scale) | "
        "Bold = Best score"
    )
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, 
            style='italic', color='#555555', wrap=True)
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✅ GPT-4o Image Quality table saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create GPT-4o Image Quality comparison table")
    parser.add_argument("--baseline-dir", required=True, help="Baseline model directory")
    parser.add_argument("--edit-only-dir", required=True, help="Edit-Only model directory")
    parser.add_argument("--standard-text-dir", required=True, help="Standard text model directory")
    parser.add_argument("--rl-text-dir", required=True, help="RL text model directory")
    parser.add_argument("--rw-text-dir", required=True, help="RW text model directory")
    parser.add_argument("--dpo-text-dir", required=True, help="DPO text model directory")
    parser.add_argument("--sw-text-dir", required=True, help="SW text model directory")
    parser.add_argument("--gpt4o-dir", required=True, help="GPT-4o model directory")
    parser.add_argument("--output", required=True, help="Output PNG file path")
    
    args = parser.parse_args()
    
    model_dirs = {
        "Baseline": args.baseline_dir,
        "Edit-Only": args.edit_only_dir,
        "Standard": args.standard_text_dir,
        "RL": args.rl_text_dir,
        "RW": args.rw_text_dir,
        "DPO": args.dpo_text_dir,
        "SW": args.sw_text_dir,
        "GPT-4o": args.gpt4o_dir
    }
    
    create_gpt_image_table(model_dirs, Path(args.output))

if __name__ == "__main__":
    main()

