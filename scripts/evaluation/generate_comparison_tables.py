#!/usr/bin/env python3
"""
Clean Table Generation Script for ImageAgent Evaluation Results
Reads consolidated_summary.json and generates comparison tables
Professional white/blue style matching original format
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import font_manager
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - image generation disabled")


class ComparisonTableGenerator:
    """Generate comparison tables from evaluation results"""
    
    # Professional color scheme (matching existing tables)
    HEADER_COLOR = '#2E86AB'  # Blue header
    HEADER_TEXT_COLOR = 'white'
    ROW_COLORS = ['white', '#F5F5F5']  # White and light gray alternating
    WINNER_COLOR = '#E8F4F8'  # Light blue for winner column
    GPT4O_COLOR = '#D0D0D0'  # Grey for GPT-4o column
    
    def __init__(self, results_path: str):
        """Load consolidated evaluation results"""
        self.results_path = Path(results_path)
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
        
        # Extract model names and labels
        self.labels = self.data.pop("labels", [])
        self.model_keys = [k for k in self.data.keys() if k != "labels"]
        
        # Create short names for tables
        self.short_names = {
            "baseline": "B",
            "edit_only": "E",
            "standard_text": "S",
            "rl_text": "R",
            "rw_text": "RW",
            "dpo_text": "D",
            "sw_text": "SW",
            "gpt4o": "GPT-4o"
        }
    
    def get_metric(self, model: str, metric_type: str, metric_name: str) -> Optional[float]:
        """Extract metric value from results"""
        # Edit-Only doesn't have planner metrics
        if model == "edit_only" and metric_type == "planner":
            return "N/A"
        
        try:
            if model not in self.data:
                return None
            
            if f"{metric_type}_metrics" not in self.data[model]:
                return None
            
            metric_data = self.data[model][f"{metric_type}_metrics"].get(metric_name, {})
            
            if isinstance(metric_data, dict):
                return metric_data.get("mean")
            elif isinstance(metric_data, (int, float)):
                return float(metric_data)
            
            return None
        except (KeyError, TypeError, AttributeError):
            return None
    
    def get_gpt_judge_metric(self, model: str, judge_type: str, metric_name: str = "overall_score") -> Optional[float]:
        """Extract GPT judge metric"""
        # Edit-Only doesn't have GPT action judge
        if model == "edit_only" and judge_type == "action":
            return "N/A"
        
        # Edit-Only doesn't have plan-related image metrics
        if model == "edit_only" and judge_type == "image":
            plan_related_metrics = ["transformation_strength", "coherence", "semantic_accuracy", "technical_execution"]
            if metric_name in plan_related_metrics:
                return "N/A"
        
        try:
            if model not in self.data:
                return None
            
            judge_data = self.data[model].get(f"gpt_{judge_type}_judge", {})
            
            if not judge_data or not isinstance(judge_data, dict):
                return None
            
            # For image judge, use overall_image_score if overall_score is requested
            if judge_type == "image" and metric_name == "overall_score":
                metric_name = "overall_image_score"
            
            metric_data = judge_data.get(metric_name, {})
            
            if isinstance(metric_data, dict):
                return metric_data.get("mean")
            elif isinstance(metric_data, (int, float)):
                return float(metric_data)
            
            return None
        except (KeyError, TypeError, AttributeError):
            return None
    
    def format_value(self, value, decimals: int = 3) -> str:
        """Format value for display"""
        if value == "N/A":
            return "N/A"
        if value is None:
            return "-"
        return f"{value:.{decimals}f}"
    
    def find_winner(self, values: List, higher_is_better: bool = True) -> int:
        """Find index of best value (excluding N/A)"""
        valid = [(i, v) for i, v in enumerate(values) if v is not None and v != "N/A"]
        if not valid:
            return -1
        
        if higher_is_better:
            return max(valid, key=lambda x: x[1])[0]
        else:
            return min(valid, key=lambda x: x[1])[0]
    
    def _style_table(self, table, num_rows, num_cols, best_value_positions=None, gpt4o_col_idx=None):
        """Apply professional white/blue styling to table (matching existing style)"""
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        
        # Style header row
        for col in range(num_cols):
            cell = table[(0, col)]
            cell.set_facecolor(self.HEADER_COLOR)
            cell.set_text_props(weight='bold', color=self.HEADER_TEXT_COLOR, fontsize=11)
        
        # Style data rows
        for row in range(1, num_rows + 1):
            # Alternating row colors
            row_color = self.ROW_COLORS[(row - 1) % 2]
            
            for col in range(num_cols):
                cell = table[(row, col)]
                
                # Winner column (last column)
                if col == num_cols - 1:
                    cell.set_facecolor(self.WINNER_COLOR)
                    cell.set_text_props(weight='bold', color=self.HEADER_COLOR)
                # GPT-4o column (grey)
                elif gpt4o_col_idx is not None and col == gpt4o_col_idx:
                    cell.set_facecolor(self.GPT4O_COLOR)
                # Metric name column (first column)
                elif col == 0:
                    cell.set_facecolor(row_color)
                    cell.set_text_props(weight='bold', fontsize=10)
                else:
                    cell.set_facecolor(row_color)
                
                # Bold best values
                if best_value_positions and (row - 1, col - 1) in best_value_positions:
                    cell.set_text_props(weight='bold', color=self.HEADER_COLOR, fontsize=11)
    
    def generate_planner_metrics_table(self, output_path: Optional[Path] = None) -> str:
        """Generate planner metrics comparison table"""
        if not HAS_MATPLOTLIB or output_path is None:
            return ""
        
        # Define metrics
        metrics = [
            ("Action F1 Score", "action_f1", True),
            ("Action IoU", "action_iou", True),
            ("Action Precision", "action_precision", True),
            ("Action Recall", "action_recall", True),
            ("Priority Correlation", "priority_correlation", True),
            ("Predicted Actions", "num_predicted_actions", False),
            ("Valid JSON", "valid_json", True),
        ]
        
        # Extract data
        table_data = []
        best_value_positions = set()
        
        for row_idx, (metric_name, metric_key, higher_is_better) in enumerate(metrics):
            row = [metric_name]
            values = []
            for model in self.model_keys:
                value = self.get_metric(model, "planner", metric_key)
                values.append(value)
                row.append(self.format_value(value))
            
            # Find winner and best value position
            winner_idx = self.find_winner(values, higher_is_better)
            if winner_idx >= 0:
                winner = self.short_names.get(self.model_keys[winner_idx], "-")
                best_value_positions.add((row_idx, winner_idx))
            else:
                winner = "-"
            row.append(winner)
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis('tight')
        ax.axis('off')
        
        # Column headers
        col_labels = ["Metric"] + [self.short_names.get(k, k) for k in self.model_keys] + ["Winner"]
        
        # Column widths (give more space to metric names)
        num_models = len(self.model_keys)
        metric_width = 0.25  # Increased from default
        model_width = (1.0 - metric_width - 0.12) / num_models  # Distribute remaining space
        col_widths = [metric_width] + [model_width] * num_models + [0.12]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=col_widths,
                        bbox=[0, 0, 1, 1])
        
        # Find GPT-4o column index
        gpt4o_col_idx = None
        try:
            gpt4o_col_idx = self.model_keys.index("gpt4o") + 1  # +1 for metric name column
        except ValueError:
            pass
        
        # Apply styling
        self._style_table(table, len(table_data), len(col_labels), best_value_positions, gpt4o_col_idx)
        
        plt.title("Planner Metrics Comparison", fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_text = " | ".join([f"{self.short_names.get(k, k)} = {self.labels[i] if i < len(self.labels) else k}" 
                                  for i, k in enumerate(self.model_keys)])
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic', color='#666666')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def generate_image_metrics_table(self, output_path: Optional[Path] = None) -> str:
        """Generate image quality metrics comparison table"""
        if not HAS_MATPLOTLIB or output_path is None:
            return ""
        
        # Define metrics
        metrics = [
            ("LPIPS ↓", "lpips", False),
            ("PSNR ↑", "psnr", True),
            ("SSIM ↑", "ssim", True),
            ("CLIP Score ↑", "clip_score", True),
            ("GPT-4o ↑", None, True),  # Special handling
        ]
        
        # Extract data
        table_data = []
        best_value_positions = set()
        
        for row_idx, (metric_name, metric_key, higher_is_better) in enumerate(metrics):
            row = [metric_name]
            values = []
            for model in self.model_keys:
                if metric_key is None:  # GPT judge
                    value = self.get_gpt_judge_metric(model, "image")
                else:
                    value = self.get_metric(model, "image", metric_key)
                values.append(value)
                row.append(self.format_value(value))
            
            # Find winner and best value position
            winner_idx = self.find_winner(values, higher_is_better)
            if winner_idx >= 0:
                winner = self.short_names.get(self.model_keys[winner_idx], "-")
                best_value_positions.add((row_idx, winner_idx))
            else:
                winner = "-"
            row.append(winner)
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Column headers
        col_labels = ["Metric"] + [self.short_names.get(k, k) for k in self.model_keys] + ["Winner"]
        
        # Column widths (give more space to metric names)
        num_models = len(self.model_keys)
        metric_width = 0.22  # Increased from default
        model_width = (1.0 - metric_width - 0.12) / num_models  # Distribute remaining space
        col_widths = [metric_width] + [model_width] * num_models + [0.12]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=col_widths,
                        bbox=[0, 0, 1, 1])
        
        # Find GPT-4o column index
        gpt4o_col_idx = None
        try:
            gpt4o_col_idx = self.model_keys.index("gpt4o") + 1  # +1 for metric name column
        except ValueError:
            pass
        
        # Apply styling
        self._style_table(table, len(table_data), len(col_labels), best_value_positions, gpt4o_col_idx)
        
        plt.title("Image Quality Metrics Comparison", fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_text = " | ".join([f"{self.short_names.get(k, k)} = {self.labels[i] if i < len(self.labels) else k}" 
                                  for i, k in enumerate(self.model_keys)])
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic', color='#666666')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def generate_gpt_action_judge_table(self, output_path: Optional[Path] = None) -> str:
        """Generate GPT-4o action judge comparison table"""
        if not HAS_MATPLOTLIB or output_path is None:
            return ""
        
        # Define metrics
        metrics = [
            ("Relevance", "relevance"),
            ("Theme/Style Focus", "theme_style_focus"),
            ("Completeness", "completeness"),
            ("Efficiency", "efficiency"),
            ("Correctness", "correctness"),
            ("Reasoning Quality", "overall_reasoning_quality"),
            ("Overall Score", "overall_score"),
        ]
        
        # Extract data
        table_data = []
        best_value_positions = set()
        
        for row_idx, (metric_name, metric_key) in enumerate(metrics):
            row = [metric_name]
            values = []
            for model in self.model_keys:
                value = self.get_gpt_judge_metric(model, "action", metric_key)
                values.append(value)
                row.append(self.format_value(value, decimals=2))
            
            # Find winner (higher is better for all GPT metrics)
            winner_idx = self.find_winner(values, True)
            if winner_idx >= 0:
                winner = self.short_names.get(self.model_keys[winner_idx], "-")
                best_value_positions.add((row_idx, winner_idx))
            else:
                winner = "-"
            row.append(winner)
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis('tight')
        ax.axis('off')
        
        # Column headers
        col_labels = ["Metric"] + [self.short_names.get(k, k) for k in self.model_keys] + ["Winner"]
        
        # Column widths (give more space to metric names)
        num_models = len(self.model_keys)
        metric_width = 0.28  # Increased from default (GPT metrics have longer names)
        model_width = (1.0 - metric_width - 0.12) / num_models  # Distribute remaining space
        col_widths = [metric_width] + [model_width] * num_models + [0.12]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=col_widths,
                        bbox=[0, 0, 1, 1])
        
        # Find GPT-4o column index
        gpt4o_col_idx = None
        try:
            gpt4o_col_idx = self.model_keys.index("gpt4o") + 1  # +1 for metric name column
        except ValueError:
            pass
        
        # Apply styling
        self._style_table(table, len(table_data), len(col_labels), best_value_positions, gpt4o_col_idx)
        
        plt.title("GPT-4o Action Plan Quality Assessment (Text Models)", fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_text = " | ".join([f"{self.short_names.get(k, k)} = {self.labels[i] if i < len(self.labels) else k}" 
                                  for i, k in enumerate(self.model_keys)])
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic', color='#666666')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def generate_gpt_image_judge_table(self, output_path: Optional[Path] = None) -> str:
        """Generate GPT-4o image judge comparison table"""
        if not HAS_MATPLOTLIB or output_path is None:
            return ""
        
        # Define metrics
        metrics = [
            ("Instruction Following", "instruction_following"),
            ("Visual Quality", "visual_quality"),
            ("Transformation Strength", "transformation_strength"),
            ("Coherence", "coherence"),
            ("Semantic Accuracy", "semantic_accuracy"),
            ("Technical Execution", "technical_execution"),
            ("Overall Image Score", "overall_image_score"),
        ]
        
        # Extract data
        table_data = []
        best_value_positions = set()
        
        for row_idx, (metric_name, metric_key) in enumerate(metrics):
            row = [metric_name]
            values = []
            for model in self.model_keys:
                value = self.get_gpt_judge_metric(model, "image", metric_key)
                values.append(value)
                row.append(self.format_value(value, decimals=2))
            
            # Find winner (higher is better for all GPT metrics)
            winner_idx = self.find_winner(values, True)
            if winner_idx >= 0:
                winner = self.short_names.get(self.model_keys[winner_idx], "-")
                best_value_positions.add((row_idx, winner_idx))
            else:
                winner = "-"
            row.append(winner)
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis('tight')
        ax.axis('off')
        
        # Column headers
        col_labels = ["Metric"] + [self.short_names.get(k, k) for k in self.model_keys] + ["Winner"]
        
        # Column widths (give more space to metric names)
        num_models = len(self.model_keys)
        metric_width = 0.28  # Increased from default (GPT metrics have longer names)
        model_width = (1.0 - metric_width - 0.12) / num_models  # Distribute remaining space
        col_widths = [metric_width] + [model_width] * num_models + [0.12]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=col_widths,
                        bbox=[0, 0, 1, 1])
        
        # Find GPT-4o column index
        gpt4o_col_idx = None
        try:
            gpt4o_col_idx = self.model_keys.index("gpt4o") + 1  # +1 for metric name column
        except ValueError:
            pass
        
        # Apply styling
        self._style_table(table, len(table_data), len(col_labels), best_value_positions, gpt4o_col_idx)
        
        plt.title("GPT-4o Image Quality Assessment (Text Models)", fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_text = " | ".join([f"{self.short_names.get(k, k)} = {self.labels[i] if i < len(self.labels) else k}" 
                                  for i, k in enumerate(self.model_keys)])
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic', color='#666666')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables from evaluation results")
    parser.add_argument("--results", required=True, help="Path to consolidated_summary.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for table images")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ComparisonTableGenerator(args.results)
    
    print("🎨 Generating comparison tables...")
    
    # Generate planner metrics table
    planner_path = output_dir / "planner_metrics_table.png"
    generator.generate_planner_metrics_table(planner_path)
    print(f"  ✅ Planner metrics: {planner_path}")
    
    # Generate image metrics table
    image_path = output_dir / "image_metrics_table.png"
    generator.generate_image_metrics_table(image_path)
    print(f"  ✅ Image metrics: {image_path}")
    
    # Generate GPT action judge table
    gpt_action_path = output_dir / "gpt4o_action_judge_table.png"
    generator.generate_gpt_action_judge_table(gpt_action_path)
    print(f"  ✅ GPT action judge: {gpt_action_path}")
    
    # Generate GPT image judge table
    gpt_image_path = output_dir / "gpt4o_image_quality_table.png"
    generator.generate_gpt_image_judge_table(gpt_image_path)
    print(f"  ✅ GPT image judge: {gpt_image_path}")
    
    print("\n✅ All tables generated successfully!")


if __name__ == "__main__":
    main()
