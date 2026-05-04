#!/usr/bin/env python3
"""
Final Metric Summary Generator
Reads evaluation results and generates a comprehensive comparison report
Can also generate table images for presentations
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Optional imports for image generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import font_manager
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - image generation disabled")


class MetricSummaryGenerator:
    """Generate formatted summary from multi-config evaluation results"""
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: Path to consolidated_summary.json
        """
        self.results_path = Path(results_path)
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
        
        # Auto-detect format: old multi-config, new 4-model, or new 3-model
        if "aggregated_metrics" in self.data:
            # Old format (3 models)
            self.configs = ["baseline", "trained_full", "trained_planner"]
            self.config_names = {
                "baseline": "Baseline",
                "trained_full": "Trained (Full)",
                "trained_planner": "Trained (Planner Only)"
            }
            self.use_new_format = False
            self.experiment_type = "old"
        elif "labels" in self.data:
            # New format (3-model or 5-model, text-only or vision)
            labels = self.data["labels"]
            num_models = len(labels)
            
            if num_models == 5:
                # 5-model experiment (Baseline, Standard, RL, RW, DPO)
                # Detect if text or vision based on keys
                if "standard_text" in self.data:
                    # Text-only 5-model experiment
                    self.configs = ["baseline", "standard_text", "rl_text", "rw_text", "dpo_text"]
                    self.config_names = {
                        "baseline": labels[0],
                        "standard_text": labels[1],
                        "rl_text": labels[2],
                        "rw_text": labels[3],
                        "dpo_text": labels[4]
                    }
                    self.experiment_type = "text_only_5model"
                else:
                    # Vision 5-model experiment
                    self.configs = ["baseline", "standard_vision", "rl_vision", "rw_vision", "dpo_vision"]
                    self.config_names = {
                        "baseline": labels[0],
                        "standard_vision": labels[1],
                        "rl_vision": labels[2],
                        "rw_vision": labels[3],
                        "dpo_vision": labels[4]
                    }
                    self.experiment_type = "vision_5model"
            elif num_models == 7:
                # 7-model experiment (Baseline, Edit-Only, Standard, RL, RW, DPO, SW)
                # Detect if text or vision based on keys
                if "standard_text" in self.data:
                    # Text-only 7-model experiment
                    self.configs = ["baseline", "edit_only", "standard_text", "rl_text", "rw_text", "dpo_text", "sw_text"]
                    self.config_names = {
                        "baseline": labels[0],
                        "edit_only": labels[1],
                        "standard_text": labels[2],
                        "rl_text": labels[3],
                        "rw_text": labels[4],
                        "dpo_text": labels[5],
                        "sw_text": labels[6]
                    }
                    self.experiment_type = "text_only_7model"
                else:
                    # Vision 7-model experiment
                    self.configs = ["baseline", "edit_only", "standard_vision", "rl_vision", "rw_vision", "dpo_vision", "sw_vision"]
                    self.config_names = {
                        "baseline": labels[0],
                        "edit_only": labels[1],
                        "standard_vision": labels[2],
                        "rl_vision": labels[3],
                        "rw_vision": labels[4],
                        "dpo_vision": labels[5],
                        "sw_vision": labels[6]
                    }
                    self.experiment_type = "vision_7model"
            elif num_models == 8:
                # 8-model experiment (7 models + GPT-4o)
                # Detect if text or vision based on keys
                if "standard_text" in self.data:
                    # Text-only 8-model experiment
                    self.configs = ["baseline", "edit_only", "standard_text", "rl_text", "rw_text", "dpo_text", "sw_text", "gpt4o"]
                    self.config_names = {
                        "baseline": labels[0],
                        "edit_only": labels[1],
                        "standard_text": labels[2],
                        "rl_text": labels[3],
                        "rw_text": labels[4],
                        "dpo_text": labels[5],
                        "sw_text": labels[6],
                        "gpt4o": labels[7]
                    }
                    self.experiment_type = "text_only_8model"
                else:
                    # Vision 8-model experiment
                    self.configs = ["baseline", "edit_only", "standard_vision", "rl_vision", "rw_vision", "dpo_vision", "sw_vision", "gpt4o"]
                    self.config_names = {
                        "baseline": labels[0],
                        "edit_only": labels[1],
                        "standard_vision": labels[2],
                        "rl_vision": labels[3],
                        "rw_vision": labels[4],
                        "dpo_vision": labels[5],
                        "sw_vision": labels[6],
                        "gpt4o": labels[7]
                    }
                    self.experiment_type = "vision_8model"
            elif num_models == 3:
                # 3-model experiment
                # Detect experiment type and set correct config keys
                if "Text" in labels[1]:
                    # Text-only experiment: baseline, standard_text, rl_text
                    self.configs = ["baseline", "standard_text", "rl_text"]
                    self.config_names = {
                        "baseline": labels[0],
                        "standard_text": labels[1],
                        "rl_text": labels[2]
                    }
                    self.experiment_type = "text_only"
                else:
                    # Vision experiment: baseline, standard_vision, rl_vision
                    self.configs = ["baseline", "standard_vision", "rl_vision"]
                    self.config_names = {
                        "baseline": labels[0],
                        "standard_vision": labels[1],
                        "rl_vision": labels[2]
                    }
                    self.experiment_type = "vision"
            else:
                raise ValueError(f"Unsupported number of models: {num_models}")
            
            self.use_new_format = True
        elif "standard" in self.data and "rl" in self.data and "vision" in self.data:
            # New 4-model consolidated format (deprecated)
            self.configs = ["baseline", "standard", "rl", "vision"]
            self.config_names = {
                "baseline": "Baseline (4B)",
                "standard": "Standard",
                "rl": "RL",
                "vision": "Vision"
            }
            self.use_new_format = True
            self.experiment_type = "4model"
        else:
            raise ValueError("Unknown consolidated_summary.json format")
        
    def extract_metric(self, config: str, metric_type: str, metric_name: str) -> float:
        """Extract a specific metric value"""
        try:
            if self.use_new_format:
                # Special handling for GPT-4o (API format)
                if config == "gpt4o":
                    # GPT-4o uses: data[gpt4o][planner_metrics][action_f1][mean] (same as others)
                    if f"{metric_type}_metrics" in self.data[config]:
                        metric_data = self.data[config][f"{metric_type}_metrics"].get(metric_name, {})
                        if isinstance(metric_data, dict):
                            return metric_data.get("mean", None)
                        elif isinstance(metric_data, (int, float)):
                            return float(metric_data)
                    return None
                
                # Special handling for Edit-Only (simplified format)
                if config == "edit_only" and metric_type == "image":
                    # Edit-Only uses simplified format: data[edit_only][metrics][metric_name][mean]
                    if "metrics" in self.data[config]:
                        # Map gpt_judge_overall to overall_quality for Edit-Only
                        if metric_name == "gpt_judge_overall":
                            return self.data[config]["metrics"].get("overall_quality", {}).get("mean", None)
                        return self.data[config]["metrics"].get(metric_name, {}).get("mean", None)
                    return None
                
                # Standard format: data[config][aggregated_X_metrics][X_metric_mean]
                # Try aggregated format first
                if f"aggregated_{metric_type}_metrics" in self.data[config]:
                    return self.data[config][f"aggregated_{metric_type}_metrics"].get(f"{metric_type}_{metric_name}_mean", None)
                
                # Fallback: direct format data[config][X_metrics][metric_name][mean]
                if f"{metric_type}_metrics" in self.data[config]:
                    metric_data = self.data[config][f"{metric_type}_metrics"].get(metric_name, {})
                    if isinstance(metric_data, dict):
                        return metric_data.get("mean", None)
                    elif isinstance(metric_data, (int, float)):
                        return float(metric_data)
                
                # Special handling for GPT judge metrics
                if metric_type == "image" and metric_name == "gpt_judge_overall":
                    # Try gpt_image_judge.overall_image_score.mean
                    if "gpt_image_judge" in self.data[config]:
                        gpt_data = self.data[config]["gpt_image_judge"].get("overall_image_score", {})
                        if isinstance(gpt_data, dict):
                            return gpt_data.get("mean", None)
                        elif isinstance(gpt_data, (int, float)):
                            return float(gpt_data)
                
                return None
            else:
                # Old format: data[aggregated_metrics][config][metric_type][metric_mean]
                return self.data["aggregated_metrics"][config][metric_type].get(f"{metric_type}_{metric_name}_mean", None)
        except (KeyError, TypeError):
            return None
    
    def format_value(self, value: float, decimals: int = 3) -> str:
        """Format metric value for display"""
        if value is None:
            return "-"
        return f"{value:.{decimals}f}"
    
    def determine_winner(self, values: List[float], higher_is_better: bool = True) -> int:
        """
        Determine which config has the best value
        Returns index of winner
        """
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        if not valid_values:
            return -1
        
        if higher_is_better:
            return max(valid_values, key=lambda x: x[1])[0]
        else:
            return min(valid_values, key=lambda x: x[1])[0]
    
    def create_planner_metrics_table(self) -> str:
        """Create formatted table for planner metrics"""
        metrics = [
            ("Action F1 Score", "action_f1", True),
            ("Action IoU", "action_iou", True),
            ("Action Precision", "action_precision", True),
            ("Action Recall", "action_recall", True),
        ]
        
        lines = []
        lines.append("\n### Planner Metrics")
        lines.append("*Higher is better*\n")
        
        # Dynamic header based on config_names
        header = "| Metric | " + " | ".join(self.config_names[c] for c in self.configs) + " | Winner |"
        separator = "|--------|" + "|".join(["-" * (len(self.config_names[c]) + 2) for c in self.configs]) + "|--------|"
        
        lines.append(header)
        lines.append(separator)
        
        for metric_name, metric_key, higher_is_better in metrics:
            values = [self.extract_metric(config, "planner", metric_key) for config in self.configs]
            formatted_values = [self.format_value(v) for v in values]
            
            # Determine winner (all metrics use higher_is_better=True now)
            winner_idx = self.determine_winner(values, higher_is_better)
            winner = self.config_names[self.configs[winner_idx]] if winner_idx >= 0 else "-"
            
            # Build row dynamically
            row = f"| {metric_name} | " + " | ".join(formatted_values) + f" | **{winner}** |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def create_image_metrics_table(self) -> str:
        """Create formatted table for image quality metrics"""
        metrics = [
            ("LPIPS ↓", "lpips", False),  # Lower is better
            ("PSNR ↑", "psnr", True),     # Higher is better
            ("SSIM ↑", "ssim", True),     # Higher is better
            ("CLIP Score ↑", "clip_score", True),  # Higher is better
            ("GPT-4o ↑", "gpt_judge_overall", True),  # Higher is better (0-10 scale)
        ]
        
        lines = []
        lines.append("\n### Image Quality Metrics")
        lines.append("*↑ = Higher is better, ↓ = Lower is better*\n")
        
        # Dynamic header based on config_names
        header = "| Metric | " + " | ".join(self.config_names[c] for c in self.configs) + " | Winner |"
        separator = "|--------|" + "|".join(["-" * (len(self.config_names[c]) + 2) for c in self.configs]) + "|--------|"
        
        lines.append(header)
        lines.append(separator)
        
        for metric_name, metric_key, higher_is_better in metrics:
            values = [self.extract_metric(config, "image", metric_key) for config in self.configs]
            formatted_values = [self.format_value(v) for v in values]
            
            # Determine winner
            winner_idx = self.determine_winner(values, higher_is_better)
            winner = self.config_names[self.configs[winner_idx]] if winner_idx >= 0 else "-"
            
            # Build row dynamically
            row = f"| {metric_name} | " + " | ".join(formatted_values) + f" | **{winner}** |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def create_gpt_action_judge_table(self) -> str:
        """Create formatted table for GPT-4o action judge metrics"""
        metrics = [
            ("Relevance", "relevance", True),
            ("Theme/Style Focus", "theme_style_focus", True),
            ("Completeness", "completeness", True),
            ("Efficiency", "efficiency", True),
            ("Correctness", "correctness", True),
            ("Overall Reasoning Quality", "overall_reasoning_quality", True),
            ("Overall Score", "overall_score", True),
        ]
        
        lines = []
        lines.append("\n### GPT-4o Action Plan Quality Assessment")
        lines.append("*Higher is better*\n")
        
        # Dynamic header based on config_names
        header = "| Metric | " + " | ".join(self.config_names[c] for c in self.configs) + " | Winner |"
        separator = "|--------|" + "|".join(["-" * (len(self.config_names[c]) + 2) for c in self.configs]) + "|--------|"
        
        lines.append(header)
        lines.append(separator)
        
        for metric_name, metric_key, higher_is_better in metrics:
            values = []
            for config in self.configs:
                # Edit-Only doesn't have action judge
                if config == "edit_only":
                    values.append("N/A")
                    continue
                
                # Extract from gpt_action_judge
                try:
                    if config in self.data and "gpt_action_judge" in self.data[config]:
                        metric_data = self.data[config]["gpt_action_judge"].get(metric_key, {})
                        if isinstance(metric_data, dict):
                            values.append(metric_data.get("mean", None))
                        elif isinstance(metric_data, (int, float)):
                            values.append(float(metric_data))
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                except (KeyError, TypeError):
                    values.append(None)
            
            formatted_values = [self.format_value(v, decimals=2) if v != "N/A" else "N/A" for v in values]
            
            # Determine winner (exclude N/A and None)
            valid_values_for_winner = [v for v in values if v is not None and v != "N/A"]
            if valid_values_for_winner:
                # Create a list with None for N/A values for determine_winner
                values_for_winner = [v if v != "N/A" else None for v in values]
                winner_idx = self.determine_winner(values_for_winner, higher_is_better)
                winner = self.config_names[self.configs[winner_idx]] if winner_idx >= 0 else "-"
            else:
                winner = "-"
            
            # Build row dynamically
            row = f"| {metric_name} | " + " | ".join(formatted_values) + f" | **{winner}** |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def create_gpt_image_judge_table(self) -> str:
        """Create formatted table for GPT-4o image quality judge metrics"""
        metrics = [
            ("Instruction Following", "instruction_following", True),
            ("Visual Quality", "visual_quality", True),
            ("Transformation Strength", "transformation_strength", True),
            ("Coherence", "coherence", True),
            ("Semantic Accuracy", "semantic_accuracy", True),
            ("Technical Execution", "technical_execution", True),
            ("Overall Image Score", "overall_image_score", True),
        ]
        
        lines = []
        lines.append("\n### GPT-4o Image Quality Assessment")
        lines.append("*Higher is better*\n")
        
        # Dynamic header based on config_names
        header = "| Metric | " + " | ".join(self.config_names[c] for c in self.configs) + " | Winner |"
        separator = "|--------|" + "|".join(["-" * (len(self.config_names[c]) + 2) for c in self.configs]) + "|--------|"
        
        lines.append(header)
        lines.append(separator)
        
        for metric_name, metric_key, higher_is_better in metrics:
            values = []
            for config in self.configs:
                # Edit-Only doesn't have plan-related image metrics
                if config == "edit_only" and metric_key in ["transformation_strength", "coherence", "semantic_accuracy", "technical_execution"]:
                    values.append("N/A")
                    continue
                
                # Extract from gpt_image_judge
                try:
                    if config in self.data and "gpt_image_judge" in self.data[config]:
                        metric_data = self.data[config]["gpt_image_judge"].get(metric_key, {})
                        if isinstance(metric_data, dict):
                            values.append(metric_data.get("mean", None))
                        elif isinstance(metric_data, (int, float)):
                            values.append(float(metric_data))
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                except (KeyError, TypeError):
                    values.append(None)
            
            formatted_values = [self.format_value(v, decimals=2) if v != "N/A" else "N/A" for v in values]
            
            # Determine winner (exclude N/A and None)
            valid_values_for_winner = [v for v in values if v is not None and v != "N/A"]
            if valid_values_for_winner:
                # Create a list with None for N/A values for determine_winner
                values_for_winner = [v if v != "N/A" else None for v in values]
                winner_idx = self.determine_winner(values_for_winner, higher_is_better)
                winner = self.config_names[self.configs[winner_idx]] if winner_idx >= 0 else "-"
            else:
                winner = "-"
            
            # Build row dynamically
            row = f"| {metric_name} | " + " | ".join(formatted_values) + f" | **{winner}** |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def calculate_overall_winner(self) -> Tuple[str, Dict[str, int]]:
        """
        Calculate overall winner based on key metrics
        Returns: (winner_name, scores_dict)
        """
        scores = {config: 0 for config in self.configs}
        
        # Planner metrics (weighted)
        planner_metrics = [
            ("action_f1", True, 3),           # Most important
            ("action_iou", True, 2),
            ("action_precision", True, 2),
            ("action_recall", True, 2),
        ]
        
        for metric_key, higher_is_better, weight in planner_metrics:
            values = [self.extract_metric(config, "planner", metric_key) for config in self.configs]
            winner_idx = self.determine_winner(values, higher_is_better)
            if winner_idx >= 0:
                scores[self.configs[winner_idx]] += weight
        
        # Image metrics (weighted less than planner)
        image_metrics = [
            ("lpips", False, 1),
            ("psnr", True, 1),
            ("ssim", True, 1),
            ("clip_score", True, 1),
            ("gpt_judge_overall", True, 2),  # GPT judge weighted higher (more comprehensive)
        ]
        
        for metric_key, higher_is_better, weight in image_metrics:
            values = [self.extract_metric(config, "image", metric_key) for config in self.configs]
            winner_idx = self.determine_winner(values, higher_is_better)
            if winner_idx >= 0:
                scores[self.configs[winner_idx]] += weight
        
        # GPT-4o Action Judge metrics (independent quality assessment)
        # Try to load from separate gpt4o_action_scores_val.json files
        gpt_action_metrics = [
            ("overall_score", True, 3),  # Overall action plan quality (weighted highly)
        ]
        
        for metric_key, higher_is_better, weight in gpt_action_metrics:
            values = []
            for config in self.configs:
                # Try to load GPT action scores
                try:
                    if hasattr(self, 'results_path') and self.results_path:
                        # Get the parent directory where model dirs are
                        results_dir = self.results_path.parent
                        if config in self.data:
                            # Try to find the model directory
                            model_base = results_dir.parent
                            if self.experiment_type == "text_only_5model":
                                model_dirs = {
                                    "baseline": model_base / "baseline",
                                    "standard_text": model_base / "standard_text",
                                    "rl_text": model_base / "rl_text",
                                    "rw_text": model_base / "rw_text",
                                    "dpo_text": model_base / "dpo_text",
                                }
                            elif self.experiment_type == "vision_5model":
                                model_dirs = {
                                    "baseline": model_base / "baseline",
                                    "standard_vision": model_base / "standard_vision",
                                    "rl_vision": model_base / "rl_vision",
                                    "rw_vision": model_base / "rw_vision",
                                    "dpo_vision": model_base / "dpo_vision",
                                }
                            else:
                                values.append(None)
                                continue
                            
                            if config in model_dirs:
                                gpt_file = model_dirs[config] / "gpt4o_action_scores_val.json"
                                if gpt_file.exists():
                                    with open(gpt_file, 'r') as f:
                                        gpt_data = json.load(f)
                                        mean_key = f'{metric_key}_mean'
                                        if 'aggregated_scores' in gpt_data and mean_key in gpt_data['aggregated_scores']:
                                            values.append(gpt_data['aggregated_scores'][mean_key])
                                        else:
                                            values.append(None)
                                else:
                                    values.append(None)
                            else:
                                values.append(None)
                        else:
                            values.append(None)
                    else:
                        values.append(None)
                except Exception as e:
                    values.append(None)
            
            # Only count if we have valid values
            if any(v is not None for v in values):
                winner_idx = self.determine_winner(values, higher_is_better)
                if winner_idx >= 0:
                    scores[self.configs[winner_idx]] += weight
        
        # Find winner
        winner_config = max(scores, key=scores.get)
        return self.config_names[winner_config], scores
    
    def create_insights(self, winner: str, scores: Dict[str, int]) -> str:
        """Generate insights and recommendations"""
        lines = []
        lines.append("\n## 🏆 Overall Winner")
        lines.append(f"\n**{winner}**\n")
        
        # Show scores
        lines.append("### Scoring Breakdown")
        lines.append("*Based on weighted performance across all metrics*\n")
        for config in self.configs:
            config_name = self.config_names[config]
            score = scores[config]
            lines.append(f"- **{config_name}**: {score} points")
        
        lines.append("\n### Why This Approach Wins\n")
        
        # Analyze winner
        if "Full" in winner:
            lines.append("The **Trained (Full)** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Superior Planning Accuracy**: Highest F1 score, IoU, and precision for action prediction")
            lines.append("✅ **Better Priority Understanding**: Best correlation with ground truth action priorities")
            lines.append("✅ **Efficient Planning**: Generates action counts closer to optimal")
            lines.append("✅ **Balanced Performance**: Maintains competitive image quality while excelling at planning")
            lines.append("")
            lines.append("**Recommendation**: Use this for production deployments requiring both accurate planning and high-quality image generation.")
            
        elif "Planner Only" in winner:
            lines.append("The **Trained (Planner Only)** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Good Planning Performance**: Strong F1 score and action prediction accuracy")
            lines.append("✅ **Best Image Quality**: Superior PSNR indicates better pixel-level reconstruction")
            lines.append("✅ **Efficient**: Uses untrained HiDream model, reducing training costs")
            lines.append("✅ **Flexible**: Can swap different HiDream backends without retraining")
            lines.append("")
            lines.append("**Recommendation**: Use this when you need good planning with flexibility in the image generation backend.")
            
        elif winner == "RL":
            lines.append("The **RL** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Highest F1 Score**: Best overall action prediction accuracy")
            lines.append("✅ **Superior IoU**: Better overlap with ground truth actions")
            lines.append("✅ **Improved Recall**: Captures more ground truth actions than baseline and standard")
            lines.append("✅ **Quality-Filtered Training**: Benefits from reward model filtering of high-quality samples")
            lines.append("")
            lines.append("**Recommendation**: Use this for production deployments requiring the most accurate action planning.")
        
        elif winner == "Vision":
            lines.append("The **Vision** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Best Priority Correlation**: Superior understanding of action importance ordering")
            lines.append("✅ **Improved Image Quality**: Highest PSNR and SSIM scores")
            lines.append("✅ **Vision-Language Integration**: Leverages both visual and textual information")
            lines.append("✅ **Better Context Understanding**: Can utilize visual cues for planning")
            lines.append("")
            lines.append("**Recommendation**: Use this when visual context is critical for accurate planning.")
        
        elif "Standard Text" in winner:
            lines.append("The **Standard Text** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Highest Precision**: Best at avoiding false positive actions")
            lines.append("✅ **Balanced Performance**: Good F1 score and IoU across all metrics")
            lines.append("✅ **Efficient Training**: Text-only training is faster and less resource-intensive")
            lines.append("✅ **Solid Baseline**: Establishes strong foundation for further improvements")
            lines.append("")
            lines.append("**Recommendation**: Use this as a reliable baseline for text-only planning tasks.")
        
        elif "RL Text" in winner:
            lines.append("The **RL Text** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Highest F1 Score**: Best overall action prediction accuracy with text-only training")
            lines.append("✅ **Superior IoU**: Better overlap with ground truth actions")
            lines.append("✅ **Quality-Filtered Training**: Benefits from reward model filtering")
            lines.append("✅ **Efficient**: Text-only training is faster than vision-language")
            lines.append("")
            lines.append("**Recommendation**: Use this for text-only deployments requiring accurate planning.")
        
        elif "Standard Vision" in winner:
            lines.append("The **Standard Vision** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Vision-Language Integration**: Leverages both visual and textual information")
            lines.append("✅ **Balanced Performance**: Good F1 score and IoU across all metrics")
            lines.append("✅ **Better Context Understanding**: Can utilize visual cues for planning")
            lines.append("✅ **Solid Foundation**: Establishes strong baseline for vision-language planning")
            lines.append("")
            lines.append("**Recommendation**: Use this when visual context is important for planning accuracy.")
        
        elif "RL Vision" in winner:
            lines.append("The **RL Vision** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Best of Both Worlds**: Combines vision-language training with reward filtering")
            lines.append("✅ **Highest Accuracy**: Best F1 score and IoU with visual context")
            lines.append("✅ **Superior Quality**: Benefits from quality-filtered training data")
            lines.append("✅ **Visual Understanding**: Can leverage image content for better planning")
            lines.append("")
            lines.append("**Recommendation**: Use this for production deployments where visual context is critical.")
        
        elif "RW Text" in winner:
            lines.append("The **RW Text** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Reward-Weighted Learning**: Gives higher importance to high-quality samples during training")
            lines.append("✅ **Balanced Quality**: Learns from all samples while emphasizing better ones")
            lines.append("✅ **Superior Accuracy**: Benefits from weighted supervised learning")
            lines.append("✅ **Efficient Training**: Text-only with reward weighting")
            lines.append("")
            lines.append("**Recommendation**: Use this when you want to leverage reward signals without preference pairs.")
        
        elif "RW Vision" in winner:
            lines.append("The **RW Vision** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Reward-Weighted Learning**: Emphasizes high-quality samples in vision-language training")
            lines.append("✅ **Visual Context + Quality**: Combines visual understanding with reward weighting")
            lines.append("✅ **Balanced Performance**: Learns from diverse samples while prioritizing quality")
            lines.append("✅ **Superior Accuracy**: Best of reward weighting and vision-language integration")
            lines.append("")
            lines.append("**Recommendation**: Use this for vision tasks where sample quality varies significantly.")
        
        elif "DPO Text" in winner:
            lines.append("The **DPO Text** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Preference Optimization**: Learns directly from chosen vs rejected pairs")
            lines.append("✅ **Strong Alignment**: Optimizes for human-preferred outputs")
            lines.append("✅ **Robust Training**: DPO loss provides stable preference learning")
            lines.append("✅ **Efficient**: Text-only training with preference pairs")
            lines.append("")
            lines.append("**Recommendation**: Use this when you have clear preference data for text-only planning.")
        
        elif "DPO Vision" in winner:
            lines.append("The **DPO Vision** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Preference Optimization**: Learns from visual preference pairs")
            lines.append("✅ **Vision-Language Alignment**: Optimizes for preferred visual planning outputs")
            lines.append("✅ **Superior Quality**: Benefits from direct preference optimization")
            lines.append("✅ **Robust Learning**: DPO provides stable training with vision context")
            lines.append("")
            lines.append("**Recommendation**: Use this for vision tasks with clear preference data.")
            
        elif winner == "Standard":
            lines.append("The **Standard** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **Highest Precision**: Best at avoiding false positive actions")
            lines.append("✅ **Balanced Performance**: Good F1 score and IoU across all metrics")
            lines.append("✅ **Efficient Training**: Text-only training is faster and less resource-intensive")
            lines.append("✅ **Solid Baseline**: Establishes strong foundation for further improvements")
            lines.append("")
            lines.append("**Recommendation**: Use this as a reliable baseline for text-only planning tasks.")
            
        else:
            lines.append("The **Baseline (4B)** approach performs best overall because:")
            lines.append("")
            lines.append("✅ **High Recall**: Captures more ground truth actions")
            lines.append("✅ **Zero Training Cost**: No fine-tuning required")
            lines.append("✅ **Good Generalization**: May work better on out-of-distribution samples")
            lines.append("✅ **Fast Deployment**: Ready to use without additional training")
            lines.append("")
            lines.append("**Recommendation**: Consider this for quick prototyping or when training resources are limited.")
        
        return "\n".join(lines)
    
    def create_config_descriptions(self) -> str:
        """Create descriptions of each configuration"""
        lines = []
        lines.append("## Summary of Approaches\n")
        
        if self.experiment_type == "text_only":
            # Text-only 3-model experiment
            lines.append("| Configuration | Training Type | Description |")
            lines.append("|---------------|---------------|-------------|")
            lines.append("| **Baseline (4B)** | Zero-shot | Qwen3-VL-4B-Instruct without any fine-tuning |")
            lines.append("| **Standard Text** | Text-only training | Fine-tuned on text prompts only (vision encoder frozen) |")
            lines.append("| **RL Text** | Text-only + Reward filtering | Trained on high-quality samples filtered by reward model |")
        elif self.experiment_type == "vision":
            # Vision 3-model experiment
            lines.append("| Configuration | Training Type | Description |")
            lines.append("|---------------|---------------|-------------|")
            lines.append("| **Baseline (4B)** | Zero-shot | Qwen3-VL-4B-Instruct without any fine-tuning |")
            lines.append("| **Standard Vision** | Vision-language training | Fine-tuned with both visual and textual information |")
            lines.append("| **RL Vision** | Vision-language + Reward filtering | Trained on high-quality samples with vision-language training |")
        elif self.experiment_type == "4model":
            # 4-model format (deprecated)
            lines.append("| Configuration | Training Type | Description |")
            lines.append("|---------------|---------------|-------------|")
            lines.append("| **Baseline (4B)** | Zero-shot | Qwen3-VL-4B-Instruct without any fine-tuning |")
            lines.append("| **Standard** | Text-only training | Fine-tuned on text prompts only (vision encoder frozen) |")
            lines.append("| **RL** | Text-only + Reward filtering | Trained on high-quality samples filtered by reward model |")
            lines.append("| **Vision** | Vision-language training | Fine-tuned with both visual and textual information using cached embeddings |")
        else:
            # Old 3-model format
            lines.append("| Configuration | Planner | Image Model | Description |")
            lines.append("|---------------|---------|-------------|-------------|")
            lines.append("| **Baseline** | Qwen3-VL (untrained) | HiDream-E1 (untrained) | Standard models without fine-tuning |")
            lines.append("| **Trained (Full)** | Qwen3-VL (fine-tuned) | HiDream-E1 (fine-tuned) | Both planner and image model fine-tuned |")
            lines.append("| **Trained (Planner Only)** | Qwen3-VL (fine-tuned) | HiDream-E1 (untrained) | Only planner fine-tuned |")
        
        return "\n".join(lines)
    
    def create_evaluation_info(self) -> str:
        """Create evaluation metadata section"""
        lines = []
        lines.append("## Evaluation Information\n")
        lines.append(f"- **Dataset Split**: {self.data.get('split', 'N/A')}")
        lines.append(f"- **Number of Samples**: {self.data.get('num_samples', 'N/A')}")
        lines.append(f"- **Successful Evaluations**: {self.data.get('num_successful', 'N/A')}/{self.data.get('num_samples', 'N/A')}")
        lines.append(f"- **End-to-End Success**: {self.data.get('num_end_to_end_success', 'N/A')}/{self.data.get('num_samples', 'N/A')}")
        lines.append(f"- **Errors**: {self.data.get('num_errors', 0)}")
        
        # Add checkpoint info
        lines.append("\n### Model Checkpoints\n")
        lines.append(f"- **Trained Planner**: `{Path(self.data.get('checkpoint', '')).name}`")
        lines.append(f"- **Trained HiDream**: `{Path(self.data.get('hidream_checkpoint', '')).name}`")
        lines.append(f"- **Base HiDream**: `{Path(self.data.get('base_hidream', '')).name}`")
        
        return "\n".join(lines)
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate complete report"""
        lines = []
        
        # Header
        lines.append("# 🎯 ImageAgent Evaluation Summary")
        lines.append("\n*Comprehensive comparison of planner and image generation approaches*\n")
        lines.append("---\n")
        
        # Evaluation info
        lines.append(self.create_evaluation_info())
        lines.append("\n---\n")
        
        # Config descriptions
        lines.append(self.create_config_descriptions())
        lines.append("\n---\n")
        
        # Performance comparison
        lines.append("## 📊 Aggregated Performance Comparison")
        lines.append(self.create_planner_metrics_table())
        lines.append(self.create_image_metrics_table())
        lines.append(self.create_gpt_action_judge_table())
        lines.append(self.create_gpt_image_judge_table())
        lines.append("\n---\n")
        
        # Winner and insights
        winner, scores = self.calculate_overall_winner()
        lines.append(self.create_insights(winner, scores))
        lines.append("\n---\n")
        
        # Footer
        lines.append("\n## 📝 Notes\n")
        lines.append("- **F1 Score**: Harmonic mean of precision and recall")
        lines.append("- **IoU**: Intersection over Union of predicted and ground truth actions")
        lines.append("- **LPIPS**: Learned Perceptual Image Patch Similarity (lower = more similar)")
        lines.append("- **PSNR**: Peak Signal-to-Noise Ratio (higher = better quality)")
        lines.append("- **SSIM**: Structural Similarity Index (higher = more similar)")
        lines.append("- **CLIP Score**: Alignment between image and instruction (higher = better)")
        
        report = "\n".join(lines)
        
        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to: {output_file}")
        
        return report
    
    def create_table_image(self, table_data: List[List[str]], title: str, output_path: str, 
                          col_widths: List[float] = None, highlight_col: int = -1, caption: str = None,
                          bold_cells: List[tuple] = None):
        """
        Create a table image using matplotlib
        
        Args:
            table_data: 2D list of table data (rows x columns)
            title: Title for the table
            output_path: Path to save the image
            col_widths: Relative column widths
            highlight_col: Column index to highlight (-1 for last column)
            caption: Optional caption/legend text to display below table
            bold_cells: List of (row, col) tuples to bold
        """
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib not available - skipping image generation")
            return
        
        n_rows = len(table_data)
        n_cols = len(table_data[0])
        
        # Set default column widths (more compact with abbreviations)
        if col_widths is None:
            col_widths = [0.22, 0.12, 0.12, 0.12, 0.16]  # Absolute widths in figure coordinates
        
        # Create figure with balanced dimensions (add space for caption if present)
        fig_width = sum(col_widths) * 8.5  # Scale factor for readability
        fig_height = max(3.5, n_rows * 0.45 + 1.2)
        if caption:
            fig_height += 0.8  # Add space for caption
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
        
        # Color header row
        for i in range(n_cols):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white', fontsize=11)
        
        # Bold specific cells if provided
        if bold_cells:
            for row, col in bold_cells:
                if 0 <= row < n_rows and 0 <= col < n_cols:
                    cell = table[(row, col)]
                    cell.set_text_props(weight='bold')
        
        # Color winner column
        if highlight_col == -1:
            highlight_col = n_cols - 1
        for i in range(1, n_rows):
            cell = table[(i, highlight_col)]
            cell.set_facecolor('#E8F4F8')
            cell.set_text_props(weight='bold', color='#2E86AB')
        
        # Alternate row colors
        for i in range(1, n_rows):
            for j in range(n_cols):
                if j != highlight_col:
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F5F5F5')
                    else:
                        table[(i, j)].set_facecolor('white')
        
        # Add title
        plt.title(title, fontsize=14, weight='bold', pad=20)
        
        # Add caption/legend if provided
        if caption:
            fig.text(0.5, 0.02, caption, ha='center', fontsize=9, 
                    style='italic', color='#555555', wrap=True)
        
        # Save
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        print(f"  📊 Table image saved: {output_path}")
    
    def generate_table_images(self, output_dir: str):
        """Generate images for planner and image metrics tables"""
        if not HAS_MATPLOTLIB:
            print("⚠️  Skipping table image generation (matplotlib not available)")
            return
        
        output_dir = Path(output_dir)
        
        print("\n🎨 Generating table images...")
        
        # Legend for abbreviations - dynamic based on experiment type
        if self.experiment_type == "text_only_7model":
            legend = "B = Baseline  |  E = Edit-Only  |  S = Standard Text  |  R = RL Text  |  RW = RW Text  |  D = DPO Text  |  SW = SW Text"
            abbrev_map = {
                self.config_names["baseline"]: "B",
                self.config_names["edit_only"]: "E",
                self.config_names["standard_text"]: "S",
                self.config_names["rl_text"]: "R",
                self.config_names["rw_text"]: "RW",
                self.config_names["dpo_text"]: "D",
                self.config_names["sw_text"]: "SW"
            }
            header_abbrevs = ["B", "E", "S", "R", "RW", "D", "SW"]
        elif self.experiment_type == "vision_7model":
            legend = "B = Baseline  |  E = Edit-Only  |  S = Standard Vision  |  R = RL Vision  |  RW = RW Vision  |  D = DPO Vision  |  SW = SW Vision"
            abbrev_map = {
                self.config_names["baseline"]: "B",
                self.config_names["edit_only"]: "E",
                self.config_names["standard_vision"]: "S",
                self.config_names["rl_vision"]: "R",
                self.config_names["rw_vision"]: "RW",
                self.config_names["dpo_vision"]: "D",
                self.config_names["sw_vision"]: "SW"
            }
            header_abbrevs = ["B", "E", "S", "R", "RW", "D", "SW"]
        elif self.experiment_type == "text_only_8model":
            legend = "B = Baseline  |  E = Edit-Only  |  S = Standard Text  |  R = RL Text  |  RW = RW Text  |  D = DPO Text  |  SW = SW Text  |  GPT-4o = GPT-4o Planner"
            abbrev_map = {
                self.config_names["baseline"]: "B",
                self.config_names["edit_only"]: "E",
                self.config_names["standard_text"]: "S",
                self.config_names["rl_text"]: "R",
                self.config_names["rw_text"]: "RW",
                self.config_names["dpo_text"]: "D",
                self.config_names["sw_text"]: "SW",
                self.config_names["gpt4o"]: "GPT-4o"
            }
            header_abbrevs = ["B", "E", "S", "R", "RW", "D", "SW", "GPT-4o"]
        elif self.experiment_type == "vision_8model":
            legend = "B = Baseline  |  E = Edit-Only  |  S = Standard Vision  |  R = RL Vision  |  RW = RW Vision  |  D = DPO Vision  |  SW = SW Vision  |  GPT-4o = GPT-4o Planner"
            abbrev_map = {
                self.config_names["baseline"]: "B",
                self.config_names["edit_only"]: "E",
                self.config_names["standard_vision"]: "S",
                self.config_names["rl_vision"]: "R",
                self.config_names["rw_vision"]: "RW",
                self.config_names["dpo_vision"]: "D",
                self.config_names["sw_vision"]: "SW",
                self.config_names["gpt4o"]: "GPT-4o"
            }
            header_abbrevs = ["B", "E", "S", "R", "RW", "D", "SW", "GPT-4o"]
        elif self.experiment_type == "text_only_5model":
            legend = "B = Baseline  |  S = Standard Text  |  R = RL Text  |  RW = RW Text  |  D = DPO Text"
            abbrev_map = {
                self.config_names["baseline"]: "B",
                self.config_names["standard_text"]: "S",
                self.config_names["rl_text"]: "R",
                self.config_names["rw_text"]: "RW",
                self.config_names["dpo_text"]: "D"
            }
            header_abbrevs = ["B", "S", "R", "RW", "D"]
        elif self.experiment_type == "vision_5model":
            legend = "B = Baseline  |  S = Standard Vision  |  R = RL Vision  |  RW = RW Vision  |  D = DPO Vision"
            abbrev_map = {
                self.config_names["baseline"]: "B",
                self.config_names["standard_vision"]: "S",
                self.config_names["rl_vision"]: "R",
                self.config_names["rw_vision"]: "RW",
                self.config_names["dpo_vision"]: "D"
            }
            header_abbrevs = ["B", "S", "R", "RW", "D"]
        elif self.experiment_type == "text_only":
            legend = "B = Baseline (4B)  |  ST = Standard Text  |  RT = RL Text"
            abbrev_map = {
                "Baseline (4B)": "B",
                "Standard Text": "ST", 
                "RL Text": "RT"
            }
            header_abbrevs = ["B", "ST", "RT"]
        elif self.experiment_type == "vision":
            legend = "B = Baseline (4B)  |  SV = Standard Vision  |  RV = RL Vision"
            abbrev_map = {
                "Baseline (4B)": "B",
                "Standard Vision": "SV", 
                "RL Vision": "RV"
            }
            header_abbrevs = ["B", "SV", "RV"]
        elif self.experiment_type == "4model":
            legend = "B = Baseline (4B)  |  S = Standard  |  R = RL  |  V = Vision"
            abbrev_map = {
                "Baseline (4B)": "B",
                "Standard": "S", 
                "RL": "R",
                "Vision": "V"
            }
            header_abbrevs = ["B", "S", "R", "V"]
        else:
            # Old format
            legend = "B = Baseline  |  TF = Trained (Full)  |  TP = Trained (Planner)"
            abbrev_map = {
                "Baseline": "B",
                "Trained (Full)": "TF",
                "Trained (Planner Only)": "TP"
            }
            header_abbrevs = ["B", "TF", "TP"]
        
        # 1. Planner Metrics Table
        planner_metrics = [
            ("Action F1 Score", "action_f1", True),
            ("Action IoU", "action_iou", True),
            ("Action Precision", "action_precision", True),
            ("Action Recall", "action_recall", True),
            ("Priority Correlation", "priority_correlation", True),
            ("Predicted Actions", "num_predicted_actions", None),
            ("Valid JSON", "valid_json", True),
        ]
        
        # Dynamic header
        planner_data = [["Metric"] + header_abbrevs + ["Winner"]]
        bold_cells = []  # Track cells to bold
        
        row_idx = 1  # Start from 1 (0 is header)
        for metric_name, metric_key, higher_is_better in planner_metrics:
            values = [self.extract_metric(config, "planner", metric_key) for config in self.configs]
            formatted_values = [self.format_value(v) for v in values]
            
            # Determine winner
            if higher_is_better is not None:
                winner_idx = self.determine_winner(values, higher_is_better)
                winner = abbrev_map.get(self.config_names[self.configs[winner_idx]], "-") if winner_idx >= 0 else "-"
            else:
                # For action count, closest to ground truth wins
                gt_count = self.extract_metric("baseline", "planner", "num_ground_truth_actions")
                if gt_count and all(v is not None for v in values):
                    diffs = [abs(v - gt_count) for v in values]
                    winner_idx = diffs.index(min(diffs))
                    winner = abbrev_map.get(self.config_names[self.configs[winner_idx]], "-")
                else:
                    winner_idx = -1
                    winner = "-"
            
            # Mark the winning value cell to bold
            if winner_idx >= 0:
                bold_cells.append((row_idx, winner_idx + 1))  # +1 to skip Metric column
            
            planner_data.append([metric_name] + formatted_values + [winner])
            row_idx += 1
        
        # Dynamic column widths based on number of configs
        if len(self.configs) == 8:
            # For 8 models: B, E, S, R, RW, D, SW, GPT-4o
            col_widths = [0.20, 0.065, 0.065, 0.065, 0.065, 0.07, 0.065, 0.065, 0.09, 0.13]  # Metric, B, E, S, R, RW, D, SW, GPT-4o, Winner
        elif len(self.configs) == 7:
            # For 7 models: B, E, S, R, RW, D, SW
            col_widths = [0.24, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.12]  # Metric, B, E, S, R, RW, D, SW, Winner
        elif len(self.configs) == 5:
            # For 5 models: B, S, R, RW, D
            col_widths = [0.28, 0.08, 0.08, 0.08, 0.09, 0.08, 0.14]  # Metric, B, S, R, RW, D, Winner
        elif len(self.configs) == 4:
            col_widths = [0.30] + [0.08] * 4 + [0.14]  # Metric, B, S, R, V, Winner
        elif len(self.configs) == 3:
            # For 3 models, use slightly wider columns for abbreviations
            if self.experiment_type in ["text_only", "vision"]:
                col_widths = [0.30, 0.10, 0.12, 0.12, 0.16]  # Metric, B, ST/SV, RT/RV, Winner
            else:
                col_widths = [0.30, 0.10, 0.10, 0.10, 0.14]  # Metric, B, TF, TP, Winner
        else:
            col_widths = None  # Use default
        
        self.create_table_image(
            planner_data, 
            "Planner Metrics Comparison",
            str(output_dir / "planner_metrics_table.png"),
            col_widths=col_widths,
            caption=legend,
            bold_cells=bold_cells
        )
        
        # 2. Image Quality Metrics Table
        image_metrics = [
            ("LPIPS ↓", "lpips", False),
            ("PSNR ↑", "psnr", True),
            ("SSIM ↑", "ssim", True),
            ("CLIP Score ↑", "clip_score", True),
            ("GPT-4o ↑", "gpt_judge_overall", True),
        ]
        
        image_data = [["Metric"] + header_abbrevs + ["Winner"]]
        image_bold_cells = []  # Track cells to bold
        
        row_idx = 1  # Start from 1 (0 is header)
        for metric_name, metric_key, higher_is_better in image_metrics:
            values = [self.extract_metric(config, "image", metric_key) for config in self.configs]
            formatted_values = [self.format_value(v) for v in values]
            
            winner_idx = self.determine_winner(values, higher_is_better)
            winner = abbrev_map.get(self.config_names[self.configs[winner_idx]], "-") if winner_idx >= 0 else "-"
            
            # Mark the winning value cell to bold
            if winner_idx >= 0:
                image_bold_cells.append((row_idx, winner_idx + 1))  # +1 to skip Metric column
            
            image_data.append([metric_name] + formatted_values + [winner])
            row_idx += 1
        
        # Dynamic column widths based on number of configs
        if len(self.configs) == 8:
            # For 8 models: B, E, S, R, RW, D, SW, GPT-4o
            col_widths = [0.15, 0.065, 0.065, 0.065, 0.065, 0.07, 0.065, 0.065, 0.09, 0.13]  # Metric, B, E, S, R, RW, D, SW, GPT-4o, Winner
        elif len(self.configs) == 7:
            # For 7 models: B, E, S, R, RW, D, SW
            col_widths = [0.18, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.12]  # Metric, B, E, S, R, RW, D, SW, Winner
        elif len(self.configs) == 5:
            # For 5 models: B, S, R, RW, D
            col_widths = [0.20, 0.08, 0.08, 0.08, 0.09, 0.08, 0.14]  # Metric, B, S, R, RW, D, Winner
        elif len(self.configs) == 4:
            col_widths = [0.21] + [0.08] * 4 + [0.14]  # Metric, B, S, R, V, Winner
        elif len(self.configs) == 3:
            # For 3 models, use slightly wider columns for abbreviations
            if self.experiment_type in ["text_only", "vision"]:
                col_widths = [0.24, 0.10, 0.12, 0.12, 0.16]  # Metric, B, ST/SV, RT/RV, Winner
            else:
                col_widths = [0.21, 0.10, 0.10, 0.10, 0.14]  # Metric, B, TF, TP, Winner
        else:
            col_widths = None  # Use default
        
        self.create_table_image(
            image_data, 
            "Image Quality Metrics Comparison",
            str(output_dir / "image_metrics_table.png"),
            col_widths=col_widths,
            caption=legend,
            bold_cells=image_bold_cells
        )
        
        # 3. Overall Summary Table
        winner, scores = self.calculate_overall_winner()
        winner_abbrev = abbrev_map.get(winner, winner)
        
        # Build summary data dynamically based on configs
        summary_data = [["Config", "Score", "Status"]]
        for config in self.configs:
            config_abbrev = abbrev_map.get(self.config_names[config], self.config_names[config][:2])
            summary_data.append([config_abbrev, str(scores[config]), ""])
        
        summary_bold_cells = []  # Track cells to bold
        
        # Mark winner and bold the winning score
        for i, config in enumerate(self.configs, start=1):
            config_abbrev = abbrev_map.get(self.config_names[config], self.config_names[config][:2])
            if config_abbrev == winner_abbrev:
                summary_data[i][2] = "🏆 WINNER"
                summary_bold_cells.append((i, 1))  # Bold the Score column (column 1)
        
        self.create_table_image(
            summary_data, 
            "Overall Performance Ranking",
            str(output_dir / "overall_summary_table.png"),
            col_widths=[0.20, 0.15, 0.22],  # Config, Score, Status
            highlight_col=2,
            caption=legend,
            bold_cells=summary_bold_cells
        )
        
        print("✅ All table images generated!")


def main():
    parser = argparse.ArgumentParser(description="Generate final metric summary from evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        default="planner_evaluation_results/final_val_5000samples/evaluation_summary_val.json",
        help="Path to evaluation_summary_val.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="planner_evaluation_results/final_val_5000samples/FINAL_SUMMARY.md",
        help="Output path for summary report (markdown)"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print report to console"
    )
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate table images (requires matplotlib)"
    )
    
    args = parser.parse_args()
    
    # Generate report
    generator = MetricSummaryGenerator(args.results)
    report = generator.generate_report(args.output)
    
    # Generate table images if requested
    if args.generate_images:
        output_dir = Path(args.output).parent
        generator.generate_table_images(str(output_dir))
    
    # Print to console if requested
    if args.print:
        print("\n" + "="*80)
        print(report)
        print("="*80 + "\n")
    else:
        print("\n✅ Summary generated successfully!")
        print(f"📄 View the report: {args.output}")
        if args.generate_images:
            print(f"🖼️  Table images saved in: {Path(args.output).parent}")


if __name__ == "__main__":
    main()

