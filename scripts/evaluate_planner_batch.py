#!/usr/bin/env python3
"""
End-to-End Planner Evaluation with Batch Mode Support

This script extends evaluate_planner.py with automatic batch result merging.
When called multiple times with the same output directory, it:
- Loads existing results
- Merges new samples with existing ones (deduplicates by sample_id)
- Recalculates aggregate metrics across ALL samples
- Saves combined results

Usage is IDENTICAL to evaluate_planner.py - just change the script name:
    python scripts/evaluate_planner_batch.py --checkpoint PATH --split val \\
        --sample-ids-file batch_1.txt --output results/
    
    python scripts/evaluate_planner_batch.py --checkpoint PATH --split val \\
        --sample-ids-file batch_2.txt --output results/
    
Result: results/ will contain cumulative metrics from both batches.

Original modes:

Mode 1: Planner-Only Evaluation
- Predict action plans
- Compare with ground truth plans
- Calculate planner metrics (F1, precision, recall)

Mode 2: End-to-End Evaluation (if HiDream checkpoint provided)
- Predict action plans
- Convert to natural language instructions
- Generate edited images with HiDream-E1
- Calculate image quality metrics (LPIPS, SSIM, PSNR, CLIP)
- Save visualizations and comparisons

Usage:
    # Planner-only
    python scripts/evaluate_planner.py --checkpoint PATH --split val
    
    # End-to-end
    python scripts/evaluate_planner.py --checkpoint PATH --split val \\
        --hidream-checkpoint PATH --hidream-config PATH --save-images
"""

# CRITICAL: Disable flash-attn BEFORE any imports
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["DIFFUSERS_ATTN_IMPLEMENTATION"] = "eager"

import sys
import json
import logging
import argparse
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import safetensors.torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from actions.planner_inference import ActionPlanner
from training.models import HiDreamE1LoRA
from training.evaluation.metrics import MetricsCalculator
from training.evaluation.gpt_action_judge import GPT4oActionJudge
from src.reward_model_evaluator import RewardModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_base_model_from_checkpoint(checkpoint_path: str) -> str:
    """
    Auto-detect base model from checkpoint's adapter_config.json.
    
    Args:
        checkpoint_path: Path to checkpoint directory or HuggingFace model ID
        
    Returns:
        Base model name (defaults to 8B for backward compatibility)
    """
    checkpoint_dir = Path(checkpoint_path)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    
    # Default to 8B for backward compatibility
    default_model = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Check if this is a local directory with checkpoint files
    if checkpoint_dir.exists() and checkpoint_dir.is_dir():
        # Try to read from adapter_config.json
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path", default_model)
                    print(f"📋 Auto-detected base model from checkpoint: {base_model}")
                    return base_model
            except Exception as e:
                print(f"⚠️  Could not read adapter_config.json: {e}")
                print(f"   Using default: {default_model}")
                return default_model
        else:
            print(f"⚠️  No adapter_config.json found in {checkpoint_path}")
            print(f"   Using default: {default_model}")
            return default_model
    else:
        # Checkpoint path is a HuggingFace model ID (baseline)
        print(f"📋 Using HuggingFace model ID: {checkpoint_path}")
        return checkpoint_path


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH MODE FUNCTIONS (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_results(output_dir: Path, split: str) -> Tuple[Optional[Dict], List[Dict], Optional[Dict]]:
    """
    Load existing evaluation results if they exist.
    
    Args:
        output_dir: Directory containing result files
        split: Split name (train/val/all)
        
    Returns:
        (summary_dict, detailed_list, gpt_scores_dict)
        - Returns (None, [], None) if no existing results
    """
    summary_file = output_dir / f"evaluation_summary_{split}.json"
    detailed_file = output_dir / f"detailed_results_{split}.json"
    gpt_file = output_dir / f"gpt4o_action_scores_{split}.json"
    
    summary = None
    detailed = []
    gpt_scores = None
    
    # Load summary
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"   ✓ Loaded existing summary: {summary_file.name}")
        except Exception as e:
            print(f"   ⚠️  Failed to load summary: {e}")
    
    # Load detailed results
    if detailed_file.exists():
        try:
            with open(detailed_file, 'r') as f:
                detailed = json.load(f)
            print(f"   ✓ Loaded existing detailed results: {len(detailed)} samples")
        except Exception as e:
            print(f"   ⚠️  Failed to load detailed results: {e}")
    
    # Load GPT scores
    if gpt_file.exists():
        try:
            with open(gpt_file, 'r') as f:
                gpt_scores = json.load(f)
            print(f"   ✓ Loaded existing GPT scores")
        except Exception as e:
            print(f"   ⚠️  Failed to load GPT scores: {e}")
    
    return summary, detailed, gpt_scores


def merge_detailed_results(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """
    Merge detailed results, avoiding duplicates by sample_id.
    New samples replace old ones if duplicate (allows re-running failed samples).
    
    Args:
        existing: List of existing per-sample results
        new: List of new per-sample results
        
    Returns:
        Combined list with unique samples
    """
    # Create dict keyed by sample_id for easy merging
    merged = {}
    
    # Add existing samples
    for sample in existing:
        sample_id = sample.get('sample_id', 'unknown')
        merged[sample_id] = sample
    
    # Add/overwrite with new samples
    for sample in new:
        sample_id = sample.get('sample_id', 'unknown')
        merged[sample_id] = sample
    
    # Convert back to list, sorted by sample_id for consistency
    return sorted(merged.values(), key=lambda x: x.get('sample_id', 'unknown'))


def aggregate_metrics_standalone(all_metrics: List[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    """
    Standalone version of aggregate_metrics.
    Aggregates metrics across all samples.
    
    Args:
        all_metrics: List of per-sample metric dictionaries
        prefix: Optional prefix for metric names
        
    Returns:
        Aggregated metrics with mean/min/max/std
    """
    if not all_metrics:
        return {}
    
    aggregated = {}
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    
    # Remove error keys
    metric_keys = [k for k in all_keys if not k.endswith("_error")]
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and not isinstance(m[key], str)]
        if values:
            metric_name = f"{prefix}_{key}" if prefix else key
            aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
            aggregated[f"{metric_name}_min"] = min(values)
            aggregated[f"{metric_name}_max"] = max(values)
            aggregated[f"{metric_name}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
    
    return aggregated


def aggregate_gpt_action_scores_standalone(all_scores: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Standalone version of _aggregate_gpt_action_scores.
    Aggregates GPT-4o action scores across samples.
    
    Args:
        all_scores: List of per-sample GPT score dictionaries
        
    Returns:
        Aggregated GPT scores with mean/min/max/std
    """
    if not all_scores:
        return {}
    
    aggregated = {}
    # Updated to include all 8 dimensions + 2 overall scores
    dimensions = [
        "relevance", "theme_style_focus", "completeness", "efficiency", "correctness",
        "reasoning_conciseness", "reasoning_completeness", "reasoning_specificity",
        "overall_action_quality", "overall_reasoning_quality", "overall_score"
    ]
    
    for dim in dimensions:
        values = [s[dim] for s in all_scores if dim in s and isinstance(s[dim], (int, float))]
        if values:
            aggregated[f"{dim}_mean"] = sum(values) / len(values)
            aggregated[f"{dim}_min"] = min(values)
            aggregated[f"{dim}_max"] = max(values)
            aggregated[f"{dim}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
    
    aggregated["num_evaluated"] = len(all_scores)
    
    return aggregated


def recalculate_aggregates_from_detailed(
    detailed_results: List[Dict], 
    multi_config: bool
) -> Dict[str, Any]:
    """
    Recalculate aggregate metrics from complete detailed results.
    This is necessary because we need stats (mean/min/max) across ALL samples,
    not just incremental updates.
    
    Args:
        detailed_results: List of per-sample results (merged from all batches)
        multi_config: Whether using multi-config evaluation mode
        
    Returns:
        Dictionary with recalculated aggregates (same structure as evaluate_dataset)
    """
    if multi_config:
        # Multi-config: separate metrics per configuration
        baseline_planner = []
        baseline_image = []
        trained_full_planner = []
        trained_full_image = []
        trained_planner_planner = []
        trained_planner_image = []
        
        for result in detailed_results:
            # Skip error results
            if "error" in result and result["error"]:
                continue
                
            # Baseline metrics
            if "baseline" in result and result["baseline"]:
                if result["baseline"].get("planner_metrics"):
                    baseline_planner.append(result["baseline"]["planner_metrics"])
                if result["baseline"].get("image_metrics"):
                    baseline_image.append(result["baseline"]["image_metrics"])
            
            # Trained full metrics
            if "trained_full" in result and result["trained_full"]:
                if result["trained_full"].get("planner_metrics"):
                    trained_full_planner.append(result["trained_full"]["planner_metrics"])
                if result["trained_full"].get("image_metrics"):
                    trained_full_image.append(result["trained_full"]["image_metrics"])
            
            # Trained planner metrics
            if "trained_planner" in result and result["trained_planner"]:
                if result["trained_planner"].get("planner_metrics"):
                    trained_planner_planner.append(result["trained_planner"]["planner_metrics"])
                if result["trained_planner"].get("image_metrics"):
                    trained_planner_image.append(result["trained_planner"]["image_metrics"])
        
        # Aggregate per configuration
        aggregated = {
            "baseline": {
                "planner": aggregate_metrics_standalone(baseline_planner, prefix="planner"),
                "image": aggregate_metrics_standalone(baseline_image, prefix="image") if baseline_image else {}
            },
            "trained_full": {
                "planner": aggregate_metrics_standalone(trained_full_planner, prefix="planner"),
                "image": aggregate_metrics_standalone(trained_full_image, prefix="image") if trained_full_image else {}
            },
            "trained_planner": {
                "planner": aggregate_metrics_standalone(trained_planner_planner, prefix="planner"),
                "image": aggregate_metrics_standalone(trained_planner_image, prefix="image") if trained_planner_image else {}
            }
        }
        
        num_successful = len(baseline_planner)
        num_end_to_end = len(baseline_image)
        
        return {
            "num_samples": len(detailed_results),
            "num_successful": num_successful,
            "num_errors": len([r for r in detailed_results if "error" in r and r["error"]]),
            "num_end_to_end_success": num_end_to_end,
            "aggregated_metrics": aggregated
        }
    else:
        # Single-config: original behavior
        all_planner_metrics = [r["planner_metrics"] for r in detailed_results if "planner_metrics" in r]
        all_image_metrics = [r["image_metrics"] for r in detailed_results if "image_metrics" in r]
        all_gpt_action_scores = [r["gpt_action_scores"] for r in detailed_results 
                                 if "gpt_action_scores" in r and "error" not in r["gpt_action_scores"]]
        
        aggregated_planner = aggregate_metrics_standalone(all_planner_metrics, prefix="planner")
        aggregated_image = aggregate_metrics_standalone(all_image_metrics, prefix="image") if all_image_metrics else {}
        aggregated_gpt = aggregate_gpt_action_scores_standalone(all_gpt_action_scores)
        
        return {
            "num_samples": len(detailed_results),
            "num_successful": len(all_planner_metrics),
            "num_errors": len([r for r in detailed_results if "error" in r]),
            "num_end_to_end_success": len(all_image_metrics),
            "aggregated_planner_metrics": aggregated_planner,
            "aggregated_image_metrics": aggregated_image,
            "aggregated_gpt_action_scores": aggregated_gpt
        }


class PlannerEvaluator:
    """End-to-end evaluator for action planner with optional image generation."""
    
    def __init__(
        self, 
        checkpoint_path: str,
        model_editor: str,
        hidream_checkpoint: Optional[str] = None,
        hidream_config: Optional[str] = None,
        device: str = "cuda",
        multi_config: bool = False,
        base_hidream_path: Optional[str] = None,
        eval_configs: list = None
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to trained planner checkpoint
            model_editor: Image editor to use ("qwen" or "hidream")
            hidream_checkpoint: Optional path to HiDream checkpoint (required if model_editor="hidream")
            hidream_config: Optional path to HiDream training config
            device: Device to use
            multi_config: If True, run 3-way comparison (baseline, trained planner+system, trained planner only)
            base_hidream_path: Path to base (untrained) HiDream model for multi-config mode
            eval_configs: List of configs to evaluate (baseline, trained_full, trained_planner)
        """
        # Validate model_editor
        if model_editor.lower() not in ["qwen", "hidream"]:
            raise ValueError(f"Invalid model_editor: '{model_editor}'. Must be 'qwen' or 'hidream'")
        
        if model_editor.lower() == "hidream" and hidream_checkpoint is None:
            raise ValueError("--hidream-checkpoint is REQUIRED when using --model-editor hidream")
        
        self.checkpoint_path = checkpoint_path
        self.model_editor = model_editor.lower()
        self.hidream_checkpoint = hidream_checkpoint
        self.hidream_config = hidream_config
        self.base_hidream_path = base_hidream_path
        self.device = device
        self.end_to_end = True  # Always True now (editor always provided)
        self.multi_config = multi_config
        self.eval_configs = eval_configs if eval_configs else ["baseline", "trained_full", "trained_planner"]
        self.use_gpt_action_judge = True  # ALWAYS ENABLED
        print(f"🐛 DEBUG: use_gpt_action_judge = {self.use_gpt_action_judge} (ALWAYS ON)")
        
        # Store model instances (loaded on-demand in multi-config mode)
        self.planner_trained = None
        self.planner_untrained = None
        self.hidream_trained = None
        self.hidream_untrained = None
        self.metrics_calculator = None
        self.gpt_action_judge = None
        self.reward_evaluator = None  # Will be initialized lazily
        
        if multi_config:
            # Multi-config mode: Load models on-demand to save memory
            print(f"\n🔄 Mode: Multi-Config Evaluation (sequential loading)")
            print(f"   Models will be loaded/unloaded per configuration to save GPU memory")
            
            if self.end_to_end:
                print(f"\n📊 Initializing metrics calculator...")
                self.metrics_calculator = MetricsCalculator(device=device)
                print("✅ Metrics calculator ready")
                
                print(f"\n🏆 Initializing reward evaluator...")
                print("   (Will reuse planner model when first loaded)")
                # reward_evaluator will be initialized lazily in evaluate_sample
                print("✅ Reward evaluator configured")
        else:
            # Single config mode: Load models upfront
            print(f"🔧 Loading trained planner from: {checkpoint_path}")
            
            # Auto-detect base model from checkpoint
            base_model = get_base_model_from_checkpoint(checkpoint_path)
            print(f"🤖 Planner Base Model: {base_model}")
            
            self.planner_trained = ActionPlanner(
                model_name=base_model,
                lora_checkpoint=checkpoint_path,
                device=device
            )
            print("✅ Trained planner loaded successfully")
            
            if self.end_to_end:
                print(f"\n🎨 Loading {self.model_editor.upper()} image editor...")
                if self.model_editor == "hidream":
                    print(f"   From: {hidream_checkpoint}")
                self.hidream_trained = self._load_editor(self.model_editor, hidream_checkpoint, hidream_config)
                print(f"✅ {self.model_editor.upper()} image editor loaded successfully")
                
                print(f"\n📊 Initializing metrics calculator...")
                self.metrics_calculator = MetricsCalculator(device=device)
                print("✅ Metrics calculator ready")
                
                print(f"\n🏆 Initializing reward evaluator (reusing planner model)...")
                self.reward_evaluator = RewardModelEvaluator(existing_model=self.planner_trained)
                print("✅ Reward evaluator ready")
                
                print(f"\n🔄 Mode: End-to-End Evaluation (Planner + Image Generation)")
            else:
                print(f"\n🔄 Mode: Planner-Only Evaluation")
        
        # Initialize GPT-4o Action Judge if requested
        if self.use_gpt_action_judge:
            print(f"\n🤖 Initializing GPT-4o Action Judge...")
            print(f"🐛 DEBUG: About to create GPT4oActionJudge instance")
            try:
                self.gpt_action_judge = GPT4oActionJudge()
                print("✅ GPT-4o Action Judge ready")
                print("   ⚠️  Note: GPT-4o calls will be made per sample (may incur costs)")
            except Exception as e:
                print(f"❌ Failed to initialize GPT-4o Action Judge: {e}")
                print("   Continuing without GPT action judging")
                self.use_gpt_action_judge = False
                self.gpt_action_judge = None
        else:
            print(f"🐛 DEBUG: GPT-4o Action Judge NOT requested (use_gpt_action_judge={self.use_gpt_action_judge})")
    
    def _load_editor(self, editor_type: str, checkpoint_path: Optional[str] = None, config_path: str = None, base_path: Optional[str] = None):
        """Load image editor (Qwen-Image-Edit or HiDream-E1)."""
        if editor_type == "qwen":
            print(f"   Loading Qwen-Image-Edit (20B parameters)...")
            from diffusers import DiffusionPipeline
            import torch
            
            model = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            
            model.set_progress_bar_config(disable=True)
            print(f"   ✅ Qwen-Image-Edit loaded successfully")
            return model
            
        elif editor_type == "hidream":
            print(f"   Loading HiDream-E1 (17B parameters)...")
            # Load config
            if config_path is None:
                config_path = project_root / "training" / "config" / "training_config.yaml"
            
            config = OmegaConf.load(config_path)
            
            # Override base model path if provided (for untrained model)
            if base_path:
                config.model.pretrained_model_path = base_path
            
            # Load base model
            model = HiDreamE1LoRA(config, device=self.device)
            
            # Load LoRA weights (if checkpoint provided)
            if checkpoint_path:
                checkpoint_path = Path(checkpoint_path)
                lora_weights_path = checkpoint_path / "adapter_model.safetensors"
                if not lora_weights_path.exists():
                    lora_weights_path = checkpoint_path / "lora_weights.safetensors"
                
                if lora_weights_path.exists():
                    print(f"   Loading LoRA weights from {lora_weights_path}")
                    lora_state = safetensors.torch.load_file(str(lora_weights_path))
                    missing, unexpected = model.transformer.load_state_dict(lora_state, strict=False)
                    print(f"   ✅ Loaded {len(lora_state)} LoRA parameters")
                else:
                    print(f"   ⚠️  No LoRA weights found at {lora_weights_path}")
            else:
                print(f"   Using base model without LoRA")
            
            # Set to eval mode
            model.transformer.eval()
            model.vae.eval()
            for encoder in model.text_encoders.values():
                if encoder is not None:
                    encoder.eval()
            
            print(f"   ✅ HiDream-E1 loaded successfully")
            return model
        
        else:
            raise ValueError(f"Invalid editor_type: {editor_type}")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory."""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    def _load_planner_on_demand(self, use_trained: bool):
        """Load planner on-demand for multi-config mode."""
        if use_trained:
            if self.planner_trained is None:
                print(f"    Loading trained planner...")
                # Auto-detect base model from checkpoint
                base_model = get_base_model_from_checkpoint(self.checkpoint_path)
                print(f"    🤖 Planner Base Model: {base_model}")
                self.planner_trained = ActionPlanner(
                    model_name=base_model,
                    lora_checkpoint=self.checkpoint_path,
                    device=self.device
                )
            return self.planner_trained
        else:
            if self.planner_untrained is None:
                print(f"    Loading untrained planner...")
                print(f"    🤖 Planner Base Model: Qwen/Qwen3-VL-8B-Instruct (default baseline)")
                self.planner_untrained = ActionPlanner(
                    model_name="Qwen/Qwen3-VL-8B-Instruct",
                    lora_checkpoint=None,
                    device=self.device
                )
            return self.planner_untrained
    
    def _load_editor_on_demand(self, use_trained: bool):
        """Load image editor on-demand for multi-config mode."""
        if use_trained or self.model_editor == "qwen":
            # Qwen doesn't support LoRA, so always use "trained" (base model)
            if self.hidream_trained is None:
                print(f"    Loading {self.model_editor.upper()} editor...")
                if self.model_editor == "hidream" and use_trained:
                    self.hidream_trained = self._load_editor(self.model_editor, self.hidream_checkpoint, self.hidream_config)
                else:
                    self.hidream_trained = self._load_editor(self.model_editor)
            return self.hidream_trained
        else:
            # Only for HiDream untrained case
            if self.hidream_untrained is None:
                print(f"    Loading untrained HiDream...")
                self.hidream_untrained = self._load_editor("hidream", None, self.hidream_config, base_path=self.base_hidream_path)
            return self.hidream_untrained
    
    def _unload_models(self):
        """Unload all models to free GPU memory."""
        import gc
        
        # Move models to CPU first to free GPU memory
        if self.planner_trained is not None:
            if hasattr(self.planner_trained, 'model'):
                self.planner_trained.model.to('cpu')
            del self.planner_trained
            self.planner_trained = None
            
        if self.planner_untrained is not None:
            if hasattr(self.planner_untrained, 'model'):
                self.planner_untrained.model.to('cpu')
            del self.planner_untrained
            self.planner_untrained = None
            
        if self.hidream_trained is not None:
            if hasattr(self.hidream_trained, 'transformer'):
                self.hidream_trained.transformer.to('cpu')
            if hasattr(self.hidream_trained, 'vae'):
                self.hidream_trained.vae.to('cpu')
            if hasattr(self.hidream_trained, 'text_encoders'):
                for encoder in self.hidream_trained.text_encoders.values():
                    if encoder is not None:
                        encoder.to('cpu')
            del self.hidream_trained
            self.hidream_trained = None
            
        if self.hidream_untrained is not None:
            if hasattr(self.hidream_untrained, 'transformer'):
                self.hidream_untrained.transformer.to('cpu')
            if hasattr(self.hidream_untrained, 'vae'):
                self.hidream_untrained.vae.to('cpu')
            if hasattr(self.hidream_untrained, 'text_encoders'):
                for encoder in self.hidream_untrained.text_encoders.values():
                    if encoder is not None:
                        encoder.to('cpu')
            del self.hidream_untrained
            self.hidream_untrained = None
        
        # Aggressive memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force another round
        gc.collect()
        torch.cuda.empty_cache()
    
    def evaluate_sample_multi_config(self, sample: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate a single sample with 3 configurations for comparison.
        
        Configurations:
        1. Baseline: Untrained Qwen3 + Untrained HiDream
        2. Trained (planner+system): Trained Qwen3 + Trained HiDream
        3. Trained (planner): Trained Qwen3 + Untrained HiDream
        
        Args:
            sample: Sample dictionary from training data
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary with evaluation results for all 3 configs
        """
        start_time = time.time()
        
        # Get paths (support both "image_path" and "image" keys)
        image_path = Path(project_root) / sample.get("image_path", sample.get("image"))
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}
        
        # Get user instruction - try overall_instruction first, fallback to user_prompt
        user_prompt = sample.get("overall_instruction", sample.get("user_prompt", ""))
        sample_id = sample.get("sample_id", sample.get("metadata", {}).get("folder_name", "unknown"))
        
        # Load analysis.json (construct path from metadata or analysis_path)
        if "analysis_path" in sample:
            analysis_path = Path(project_root) / sample["analysis_path"]
        else:
            source_dir = Path(project_root) / sample["metadata"]["source_dir"]
            analysis_path = source_dir / "analysis.json"
        
        if analysis_path.exists():
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
        else:
            analysis = None  # Analysis optional for evaluation
        
        # Load ground truth
        ground_truth_plan = sample["target_action_plan"]
        
        # Get edited image path
        if "edited_image_path" in sample:
            gt_edit_path = Path(project_root) / sample["edited_image_path"]
        else:
            source_dir = Path(project_root) / sample["metadata"]["source_dir"]
            gt_edit_path = source_dir / "edited.png"
        
        if not gt_edit_path.exists():
            return {"error": f"Ground truth image not found: {gt_edit_path}"}
        
        gt_edit = Image.open(gt_edit_path).convert("RGB")
        original_image = Image.open(image_path).convert("RGB")
        
        # ========== CONFIG 1: Baseline (Untrained + Untrained) ==========
        if "baseline" in self.eval_configs:
            print(f"  [1/{len(self.eval_configs)}] Baseline (Untrained Qwen3 + Untrained HiDream)...")
        else:
            print(f"  [Skipping] Baseline (not in eval_configs)")
            baseline_plan, baseline_instruction, baseline_edit, baseline_metrics = None, None, None, {}
            baseline_time = 0
            baseline_hidream_prompt = None
        
        if "baseline" in self.eval_configs:
            try:
                planner = self._load_planner_on_demand(use_trained=False)
                # Using default temperature=0.1 to match data generation
                # Pass analysis for better CoT reasoning
                baseline_plan = planner.predict_action_plan(
                    image_path=str(image_path),
                    user_prompt=user_prompt,
                    analysis=analysis
                )
                # Keep model's predicted overall_instruction (don't overwrite)
                
                # PREFER: Model-generated hidream_prompt (if available)
                # FALLBACK: Rule-based merge (for backward compatibility)
                if "hidream_prompt" in baseline_plan and baseline_plan["hidream_prompt"]:
                    baseline_instruction = baseline_plan["hidream_prompt"]
                    logger.info(f"    Using model-generated hidream_prompt for baseline")
                else:
                    baseline_instruction = baseline_plan.get("overall_instruction", "style_transformation_mode")
                    logger.warning(f"    hidream_prompt not found in baseline_plan, using overall_instruction fallback")
                
                hidream = self._load_editor_on_demand(use_trained=False)
                baseline_edit, baseline_time, baseline_hidream_prompt = self._generate_edited_image_with_model(
                    image_path, baseline_instruction, hidream
                )
                baseline_metrics = self._calculate_image_metrics(
                    baseline_edit, 
                    gt_edit, 
                    baseline_instruction,
                    original_image=original_image
                )
                
                # Unload baseline models
                print(f"    Unloading baseline models...")
                self._unload_models()
                
                # Report memory status
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"    GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                print(f"  ⚠️  Baseline failed: {e}")
                baseline_plan, baseline_instruction, baseline_edit, baseline_metrics = None, None, None, {}
                baseline_time = 0
                baseline_hidream_prompt = None
                self._unload_models()  # Ensure cleanup
        
        # ========== CONFIG 2: Trained (planner+system) ==========
        if "trained_full" in self.eval_configs:
            config_num = list(self.eval_configs).index("trained_full") + 1
            print(f"  [{config_num}/{len(self.eval_configs)}] Trained (Trained Qwen3 + Trained HiDream)...")
        else:
            print(f"  [Skipping] Trained (planner+system) (not in eval_configs)")
            trained_full_plan, trained_full_instruction, trained_full_edit, trained_full_metrics = None, None, None, {}
            trained_full_time = 0
            trained_full_hidream_prompt = None
        
        if "trained_full" in self.eval_configs:
            try:
                planner = self._load_planner_on_demand(use_trained=True)
                # Using default temperature=0.1 to match data generation
                # Pass analysis for better CoT reasoning
                trained_full_plan = planner.predict_action_plan(
                    image_path=str(image_path),
                    user_prompt=user_prompt,
                    analysis=analysis
                )
                # Keep model's predicted overall_instruction (don't overwrite)
                
                # PREFER: Model-generated hidream_prompt (if available)
                # FALLBACK: Rule-based merge (for backward compatibility)
                if "hidream_prompt" in trained_full_plan and trained_full_plan["hidream_prompt"]:
                    trained_full_instruction = trained_full_plan["hidream_prompt"]
                    logger.info(f"    Using model-generated hidream_prompt for trained_full")
                else:
                    trained_full_instruction = trained_full_plan.get("overall_instruction", "style_transformation_mode")
                    logger.warning(f"    hidream_prompt not found in trained_full_plan, using overall_instruction fallback")
                
                hidream = self._load_editor_on_demand(use_trained=True)
                trained_full_edit, trained_full_time, trained_full_hidream_prompt = self._generate_edited_image_with_model(
                    image_path, trained_full_instruction, hidream
                )
                trained_full_metrics = self._calculate_image_metrics(
                    trained_full_edit, 
                    gt_edit, 
                    trained_full_instruction,
                    original_image=original_image
                )
                
                # Unload trained full models
                print(f"    Unloading trained full models...")
                self._unload_models()
                
                # Report memory status
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"    GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                print(f"  ⚠️  Trained (full) failed: {e}")
                trained_full_plan, trained_full_instruction, trained_full_edit, trained_full_metrics = None, None, None, {}
                trained_full_time = 0
                trained_full_hidream_prompt = None
                self._unload_models()  # Ensure cleanup
        
        # ========== CONFIG 3: Trained (planner only) ==========
        if "trained_planner" in self.eval_configs:
            config_num = list(self.eval_configs).index("trained_planner") + 1
            print(f"  [{config_num}/{len(self.eval_configs)}] Trained (Trained Qwen3 + Untrained HiDream)...")
        else:
            print(f"  [Skipping] Trained (planner) (not in eval_configs)")
            trained_planner_plan, trained_planner_instruction, trained_planner_edit, trained_planner_metrics = None, None, None, {}
            trained_planner_time = 0
            trained_planner_hidream_prompt = None
        
        if "trained_planner" in self.eval_configs:
            try:
                planner = self._load_planner_on_demand(use_trained=True)
                # Using default temperature=0.1 to match data generation
                # Pass analysis for better CoT reasoning
                trained_planner_plan = planner.predict_action_plan(
                    image_path=str(image_path),
                    user_prompt=user_prompt,
                    analysis=analysis
                )
                # Keep model's predicted overall_instruction (don't overwrite)
                
                # PREFER: Model-generated hidream_prompt (if available)
                # FALLBACK: Rule-based merge (for backward compatibility)
                if "hidream_prompt" in trained_planner_plan and trained_planner_plan["hidream_prompt"]:
                    trained_planner_instruction = trained_planner_plan["hidream_prompt"]
                    logger.info(f"    Using model-generated hidream_prompt for trained_planner")
                else:
                    trained_planner_instruction = trained_planner_plan.get("overall_instruction", "style_transformation_mode")
                    logger.warning(f"    hidream_prompt not found in trained_planner_plan, using overall_instruction fallback")
                
                hidream = self._load_editor_on_demand(use_trained=False)
                trained_planner_edit, trained_planner_time, trained_planner_hidream_prompt = self._generate_edited_image_with_model(
                    image_path, trained_planner_instruction, hidream
                )
                trained_planner_metrics = self._calculate_image_metrics(
                    trained_planner_edit, 
                    gt_edit, 
                    trained_planner_instruction,
                    original_image=original_image
                )
                
                # Unload trained planner models
                print(f"    Unloading trained planner models...")
                self._unload_models()
                
                # Report memory status
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"    GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                print(f"  ⚠️  Trained (planner only) failed: {e}")
                trained_planner_plan, trained_planner_instruction, trained_planner_edit, trained_planner_metrics = None, None, None, {}
                trained_planner_time = 0
                trained_planner_hidream_prompt = None
                self._unload_models()  # Ensure cleanup
        
        # ========== Calculate Planner Metrics ==========
        baseline_planner_metrics = self.calculate_planner_metrics(baseline_plan, ground_truth_plan) if baseline_plan else {}
        trained_full_planner_metrics = self.calculate_planner_metrics(trained_full_plan, ground_truth_plan) if trained_full_plan else {}
        trained_planner_planner_metrics = self.calculate_planner_metrics(trained_planner_plan, ground_truth_plan) if trained_planner_plan else {}
        
        # ========== Save Outputs ==========
        if output_dir:
            sample_dir = output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original and ground truth images (same for all configs)
            original_image.save(sample_dir / "original.png")
            gt_edit.save(sample_dir / "ground_truth.png")
            
            # Save plans
            if baseline_plan:
                with open(sample_dir / "baseline_plan.json", 'w') as f:
                    json.dump(baseline_plan, f, indent=2)
            if trained_full_plan:
                with open(sample_dir / "trained_full_plan.json", 'w') as f:
                    json.dump(trained_full_plan, f, indent=2)
            if trained_planner_plan:
                with open(sample_dir / "trained_planner_plan.json", 'w') as f:
                    json.dump(trained_planner_plan, f, indent=2)
            
            # Copy ground truth action plan for comparison
            gt_results_dir = Path("imageagent_results_16000_cot") / sample_id
            gt_action_plan = gt_results_dir / "action_plan.json"
            if gt_action_plan.exists():
                shutil.copy(gt_action_plan, sample_dir / "ground_truth_plan.json")
            
            # Save instructions
            if baseline_instruction:
                with open(sample_dir / "baseline_instruction.txt", 'w') as f:
                    f.write(baseline_instruction)
            if trained_full_instruction:
                with open(sample_dir / "trained_full_instruction.txt", 'w') as f:
                    f.write(trained_full_instruction)
            if trained_planner_instruction:
                with open(sample_dir / "trained_planner_instruction.txt", 'w') as f:
                    f.write(trained_planner_instruction)
            
            # Save HiDream prompts (with wrapper)
            if baseline_hidream_prompt:
                with open(sample_dir / "baseline_edit_prompt.txt", 'w') as f:
                    f.write(baseline_hidream_prompt)
            if trained_full_hidream_prompt:
                with open(sample_dir / "trained_full_edit_prompt.txt", 'w') as f:
                    f.write(trained_full_hidream_prompt)
            if trained_planner_hidream_prompt:
                with open(sample_dir / "trained_planner_edit_prompt.txt", 'w') as f:
                    f.write(trained_planner_hidream_prompt)
            
            # Save images
            if baseline_edit:
                baseline_edit.save(sample_dir / "baseline_edit.png")
            if trained_full_edit:
                trained_full_edit.save(sample_dir / "trained_full_edit.png")
            if trained_planner_edit:
                trained_planner_edit.save(sample_dir / "trained_planner_edit.png")
            
            # ============ COMPUTE REWARD SCORES FOR ALL CONFIGS ============
            # Lazy initialization of reward evaluator (first time only)
            if self.reward_evaluator is None:
                print(f"  🏆 Initializing reward evaluator (one-time setup)...")
                # Find first available planner
                if self.planner_trained is not None:
                    self.reward_evaluator = RewardModelEvaluator(existing_model=self.planner_trained)
                elif self.planner_untrained is not None:
                    self.reward_evaluator = RewardModelEvaluator(existing_model=self.planner_untrained)
                else:
                    # Load a temporary planner just for reward evaluation
                    temp_planner = self._load_planner_on_demand(use_trained=False)
                    self.reward_evaluator = RewardModelEvaluator(existing_model=temp_planner)
                print(f"  ✅ Reward evaluator ready")
            
            # Compute reward scores for each config
            reward_results = {}
            
            # Baseline rewards
            if baseline_edit and baseline_plan:
                print(f"  🏆 Computing reward scores for baseline...")
                try:
                    reward_results['baseline'] = self.reward_evaluator.evaluate_transformation(
                        original_image_path=str(image_path),
                        edited_image_path=str(sample_dir / "baseline_edit.png"),
                        user_prompt=user_prompt,
                        action_plan=baseline_plan,
                        analysis=analysis
                    )
                    with open(sample_dir / "baseline_reward_scores.json", 'w') as f:
                        json.dump(reward_results['baseline'], f, indent=2)
                except Exception as e:
                    print(f"  ⚠️  Failed to compute baseline rewards: {e}")
                    reward_results['baseline'] = None
            
            # Trained full rewards
            if trained_full_edit and trained_full_plan:
                print(f"  🏆 Computing reward scores for trained_full...")
                try:
                    reward_results['trained_full'] = self.reward_evaluator.evaluate_transformation(
                        original_image_path=str(image_path),
                        edited_image_path=str(sample_dir / "trained_full_edit.png"),
                        user_prompt=user_prompt,
                        action_plan=trained_full_plan,
                        analysis=analysis
                    )
                    with open(sample_dir / "trained_full_reward_scores.json", 'w') as f:
                        json.dump(reward_results['trained_full'], f, indent=2)
                except Exception as e:
                    print(f"  ⚠️  Failed to compute trained_full rewards: {e}")
                    reward_results['trained_full'] = None
            
            # Trained planner rewards
            if trained_planner_edit and trained_planner_plan:
                print(f"  🏆 Computing reward scores for trained_planner...")
                try:
                    reward_results['trained_planner'] = self.reward_evaluator.evaluate_transformation(
                        original_image_path=str(image_path),
                        edited_image_path=str(sample_dir / "trained_planner_edit.png"),
                        user_prompt=user_prompt,
                        action_plan=trained_planner_plan,
                        analysis=analysis
                    )
                    with open(sample_dir / "trained_planner_reward_scores.json", 'w') as f:
                        json.dump(reward_results['trained_planner'], f, indent=2)
                except Exception as e:
                    print(f"  ⚠️  Failed to compute trained_planner rewards: {e}")
                    reward_results['trained_planner'] = None
            
            print(f"  ✓ Reward scores computed for all configs")
            # ==========================================
            
            # Save comparison (flexible based on which configs were evaluated)
            # At least one edited image must exist (besides original and ground truth)
            available_edits = [baseline_edit, trained_full_edit, trained_planner_edit]
            if any(available_edits):
                # Use user_prompt for caption (more informative than planner's generic output)
                self._save_comparison(
                    original=original_image,
                    predicted=trained_full_edit,  # Trained (planner+system), may be None
                    ground_truth=gt_edit,
                    output_path=sample_dir / "comparison.png",
                    overall_instruction=user_prompt,  # Use actual user request
                    baseline=baseline_edit,  # May be None
                    trained_planner_only=trained_planner_edit  # May be None
                )
            
            # Save metrics
            metrics_summary = {
                "baseline": {
                    "planner_metrics": baseline_planner_metrics,
                    "image_metrics": baseline_metrics,
                    "generation_time": baseline_time
                },
                "trained_full": {
                    "planner_metrics": trained_full_planner_metrics,
                    "image_metrics": trained_full_metrics,
                    "generation_time": trained_full_time
                },
                "trained_planner": {
                    "planner_metrics": trained_planner_planner_metrics,
                    "image_metrics": trained_planner_metrics,
                    "generation_time": trained_planner_time
                }
            }
            with open(sample_dir / "metrics.json", 'w') as f:
                json.dump(metrics_summary, f, indent=2)
        
        total_time = time.time() - start_time
        
        return {
            "sample_id": sample_id,
            "user_prompt": user_prompt,
            "total_time": total_time,
            "baseline": {
                "plan": baseline_plan,
                "instruction": baseline_instruction,
                "planner_metrics": baseline_planner_metrics,
                "image_metrics": baseline_metrics,
                "generation_time": baseline_time
            },
            "trained_full": {
                "plan": trained_full_plan,
                "instruction": trained_full_instruction,
                "planner_metrics": trained_full_planner_metrics,
                "image_metrics": trained_full_metrics,
                "generation_time": trained_full_time
            },
            "trained_planner": {
                "plan": trained_planner_plan,
                "instruction": trained_planner_instruction,
                "planner_metrics": trained_planner_planner_metrics,
                "image_metrics": trained_planner_metrics,
                "generation_time": trained_planner_time
            }
        }
    
    def evaluate_sample(self, sample: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate a single sample with planner and optionally generate images.
        
        Args:
            sample: Sample dictionary from training data
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary with evaluation results
        """
        if self.multi_config:
            return self.evaluate_sample_multi_config(sample, output_dir)
        
        start_time = time.time()
        
        # Get paths
        image_path = Path(project_root) / sample["original_image_path"]
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}
        
        # Get user instruction - try overall_instruction first, fallback to user_prompt
        user_instruction = sample.get("overall_instruction", sample.get("user_prompt", ""))
        sample_id = sample["sample_id"]
        
        # Step 1: Predict action plan
        try:
            # Using default temperature=0.1 to match data generation
            predicted_plan = self.planner_trained.predict_action_plan(
                image_path=str(image_path),
                user_prompt=user_instruction
            )
            # Don't overwrite model's prediction - keep what it generated
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
        
        # Step 2: Calculate planner metrics
        ground_truth_plan = sample["target_action_plan"]
        planner_metrics = self.calculate_planner_metrics(predicted_plan, ground_truth_plan)
        
        result = {
            "sample_id": sample_id,
            "user_prompt": user_instruction,
            "predicted_plan": predicted_plan,
            "ground_truth_plan": ground_truth_plan,
            "planner_metrics": planner_metrics
        }
        
        # Step 2.5: GPT-4o Action Judge (if enabled) - SEPARATE from teacher metrics
        print(f"🐛 DEBUG: Checking GPT Action Judge - use_gpt_action_judge={self.use_gpt_action_judge}, gpt_action_judge={self.gpt_action_judge is not None}")
        if self.use_gpt_action_judge and self.gpt_action_judge is not None:
            print(f"🐛 DEBUG: Calling GPT-4o Action Judge for sample {sample_id}")
            try:
                original_image = Image.open(image_path).convert("RGB")
                gpt_action_scores = self.gpt_action_judge.judge_action_plan(
                    original_image=original_image,
                    user_prompt=user_instruction,
                    predicted_plan=predicted_plan,
                    teacher_plan=ground_truth_plan  # Provide as reference but judge independently
                )
                result["gpt_action_scores"] = gpt_action_scores
                print(f"✅ GPT Action Judge completed for {sample_id}")
            except Exception as e:
                print(f"⚠️  GPT action judge failed for {sample_id}: {e}")
                result["gpt_action_scores"] = {"error": str(e)}
        
        # Step 3: End-to-end evaluation (if editor loaded)
        if self.end_to_end:
            try:
                # Load analysis.json
                analysis_path = Path(project_root) / sample["analysis_path"]
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                
                # Convert action plan to natural language instruction
                # PREFER: Model-generated hidream_prompt (if available)
                # FALLBACK: Rule-based merge (for backward compatibility)
                if "hidream_prompt" in predicted_plan and predicted_plan["hidream_prompt"]:
                    instruction = predicted_plan["hidream_prompt"]
                    logger.info(f"    Using model-generated hidream_prompt for end-to-end eval")
                else:
                    instruction = predicted_plan.get("overall_instruction", "style_transformation_mode")
                    logger.warning(f"    hidream_prompt not found in predicted_plan, using overall_instruction fallback")
                result["instruction"] = instruction
                result["instruction_length"] = len(instruction.split())
                
                # Generate edited image (editor is stored in hidream_trained regardless of type)
                editor = self.hidream_trained if self.hidream_trained else self.hidream_model
                predicted_edit, gen_time, hidream_prompt = self._generate_edited_image_with_model(
                    image_path=image_path,
                    instruction=instruction,
                    editor_model=editor
                )
                result["generation_time"] = gen_time
                
                # Load ground truth edited image
                # Check if we have edited_image_path directly (new format)
                if "edited_image_path" in sample:
                    gt_edit_path = Path(project_root) / sample["edited_image_path"]
                else:
                    # Fallback: use source_dir (old format)
                    source_dir = Path(project_root) / sample["metadata"]["source_dir"]
                    gt_edit_path = source_dir / "edited.png"
                
                if gt_edit_path.exists():
                    gt_edit = Image.open(gt_edit_path).convert("RGB")
                    
                    # Load original image for GPT judge
                    original = Image.open(image_path).convert("RGB")
                    
                    # Calculate image metrics (pass all images for GPT judge)
                    image_metrics = self._calculate_image_metrics(
                        predicted_edit, 
                        gt_edit, 
                        instruction,
                        original_image=original
                    )
                    result["image_metrics"] = image_metrics
                    
                    # Save outputs if directory provided
                    if output_dir:
                        sample_dir = output_dir / sample_id
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save original and ground truth images
                        original = Image.open(image_path).convert("RGB")
                        original.save(sample_dir / "original.png")
                        gt_edit.save(sample_dir / "ground_truth.png")
                        
                        # Save predicted plan
                        with open(sample_dir / "predicted_plan.json", 'w') as f:
                            json.dump(predicted_plan, f, indent=2)
                        
                        # Copy ground truth action plan for comparison
                        gt_results_dir = Path("imageagent_results_16000_cot") / sample_id
                        gt_action_plan = gt_results_dir / "action_plan.json"
                        if gt_action_plan.exists():
                            shutil.copy(gt_action_plan, sample_dir / "ground_truth_plan.json")
                        
                        # Save instruction
                        with open(sample_dir / "instruction.txt", 'w') as f:
                            f.write(instruction)
                        
                        # Save HiDream prompt (with wrapper)
                        with open(sample_dir / "edit_prompt.txt", 'w') as f:
                            f.write(hidream_prompt)
                        
                        # Save predicted edit
                        predicted_edit.save(sample_dir / "predicted_edit.png")
                        
                        # ============ COMPUTE REWARD SCORES ============
                        # Lazy initialization of reward evaluator (multi-config mode)
                        if self.reward_evaluator is None and self.multi_config:
                            print(f"  🏆 Initializing reward evaluator (one-time setup)...")
                            # Find first available planner
                            if self.planner_trained is not None:
                                self.reward_evaluator = RewardModelEvaluator(existing_model=self.planner_trained)
                            elif self.planner_untrained is not None:
                                self.reward_evaluator = RewardModelEvaluator(existing_model=self.planner_untrained)
                            print("  ✅ Reward evaluator ready")
                        
                        # Compute reward scores
                        if self.reward_evaluator is not None:
                            print(f"  🏆 Computing reward scores...")
                            try:
                                # Load analysis if available
                                analysis = None
                                gt_results_dir = Path("imageagent_results_16000_cot") / sample_id
                                analysis_path = gt_results_dir / "analysis.json"
                                if analysis_path.exists():
                                    with open(analysis_path, 'r') as f:
                                        analysis = json.load(f)
                                
                                reward_result = self.reward_evaluator.evaluate_transformation(
                                    original_image_path=str(image_path),
                                    edited_image_path=str(sample_dir / "predicted_edit.png"),
                                    user_prompt=user_instruction,
                                    action_plan=predicted_plan,
                                    analysis=analysis
                                )
                                
                                # Save reward scores
                                with open(sample_dir / "reward_scores.json", 'w') as f:
                                    json.dump(reward_result, f, indent=2)
                                
                                # Extract individual scores for aggregation
                                scores = reward_result.get('scores', {})
                                result["reward_action_plan_quality"] = scores.get('action_plan_quality', {}).get('score', 0)
                                result["reward_plan_reasoning"] = scores.get('plan_reasoning', {}).get('score', 0)
                                result["reward_reasoning_quality"] = scores.get('reasoning_quality', {}).get('score', 0)
                                result["reward_final_image_quality"] = scores.get('final_image_quality', {}).get('score', 0)
                                result["reward_adherence_to_plan"] = scores.get('adherence_to_plan', {}).get('score', 0)
                                result["reward_adherence_to_prompt"] = scores.get('adherence_to_prompt', {}).get('score', 0)
                                result["reward_overall_quality"] = scores.get('overall_quality', {}).get('score', 0)
                                
                                print(f"  ✓ Reward scores computed")
                                
                            except Exception as e:
                                print(f"  ⚠️  Failed to compute reward scores: {e}")
                                import traceback
                                traceback.print_exc()
                                # Add zero scores if evaluation fails
                                for metric in ['action_plan_quality', 'plan_reasoning', 'reasoning_quality', 
                                               'final_image_quality', 'adherence_to_plan', 'adherence_to_prompt', 
                                               'overall_quality']:
                                    result[f'reward_{metric}'] = 0
                        # ==========================================
                        
                        # Save comparison
                        self._save_comparison(
                            original,
                            predicted_edit,
                            gt_edit,
                            sample_dir / "comparison.png",
                            overall_instruction=user_instruction,  # Use actual user request
                            predicted_label="Predicted"
                        )
                        
                        # Save metrics
                        with open(sample_dir / "metrics.json", 'w') as f:
                            json.dump({
                                "planner_metrics": planner_metrics,
                                "image_metrics": image_metrics,
                                "instruction_length": result["instruction_length"],
                                "generation_time": gen_time
                            }, f, indent=2)
                else:
                    result["warning"] = f"Ground truth edited image not found: {gt_edit_path}"
                    
            except Exception as e:
                result["end_to_end_error"] = f"End-to-end evaluation failed: {str(e)}"
                import traceback
                result["end_to_end_traceback"] = traceback.format_exc()
        
        result["total_time"] = time.time() - start_time
        return result
    
    @torch.no_grad()
    def _generate_edited_image(self, image_path: Path, instruction: str) -> Tuple[Image.Image, float, str]:
        """
        Generate edited image using HiDream-E1.
        
        Returns:
            (edited_image, generation_time, formatted_prompt)
        """
        start_time = time.time()
        
        # Load and prepare image
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = pil_image.resize((768, 768), Image.Resampling.LANCZOS)
        
        # Format prompt for HiDream-E1 (simplified wrapper - no duplication)
        # Instruction appears once, not twice
        formatted_prompt = f"Editing Instruction: {instruction}. Maintain high quality, original composition and style."
        
        # Use pipeline
        hidream = self.hidream_trained if self.hidream_trained else self.hidream_model
        if hidream.pipeline is None:
            raise RuntimeError("Pipeline not available")
        
        output = hidream.pipeline(
            prompt=formatted_prompt,
            image=pil_image,
            height=768,
            width=768,
            num_inference_steps=28,
            guidance_scale=5.0,
            image_guidance_scale=4.0,
        )
        
        edited_image = output.images[0]
        gen_time = time.time() - start_time
        
        return edited_image, gen_time, formatted_prompt
    
    def _generate_edited_image_with_model(self, image_path: Path, instruction: str, editor_model) -> Tuple[Image.Image, float, str]:
        """
        Generate edited image using a specific editor model (Qwen or HiDream).
        
        Args:
            image_path: Path to source image
            instruction: Natural language editing instruction
            editor_model: Specific editor model to use (Qwen or HiDream)
            
        Returns:
            (edited_image, generation_time, formatted_prompt)
        """
        start_time = time.time()
        
        # Load and prepare image
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = pil_image.resize((768, 768), Image.Resampling.LANCZOS)
        
        # Format prompt (simplified wrapper - no duplication)
        formatted_prompt = f"Editing Instruction: {instruction}. Maintain high quality, original composition and style."
        
        # Generate with appropriate API based on model type
        if self.model_editor == "qwen":
            # Qwen-Image-Edit API
            output = editor_model(
                prompt=formatted_prompt,
                image=pil_image,
                height=768,
                width=768,
                num_inference_steps=28,
                guidance_scale=7.5,
            )
        elif self.model_editor == "hidream":
            # HiDream-E1 API
            if editor_model.pipeline is None:
                raise RuntimeError("Pipeline not available")
            
            output = editor_model.pipeline(
                prompt=formatted_prompt,
                image=pil_image,
                height=768,
                width=768,
                num_inference_steps=28,
                guidance_scale=3.5,
                image_guidance_scale=4.0,
            )
        else:
            raise ValueError(f"Unknown model_editor: {self.model_editor}")
        
        edited_image = output.images[0]
        gen_time = time.time() - start_time
        
        return edited_image, gen_time, formatted_prompt
    
    def _calculate_image_metrics(
        self, 
        predicted: Image.Image, 
        ground_truth: Image.Image, 
        instruction: str = None,
        original_image: Image.Image = None
    ) -> Dict[str, float]:
        """Calculate image quality metrics including GPT-4o judge."""
        # Resize images to match dimensions (use ground truth size as reference)
        if predicted.size != ground_truth.size:
            print(f"    ⚠️ Resizing predicted image from {predicted.size} to {ground_truth.size}")
            predicted = predicted.resize(ground_truth.size, Image.Resampling.LANCZOS)
        
        # Convert to tensors
        pred_tensor = self._pil_to_tensor(predicted).unsqueeze(0).to(self.device)
        gt_tensor = self._pil_to_tensor(ground_truth).unsqueeze(0).to(self.device)
        
        # Calculate metrics
        metrics = {}
        
        try:
            lpips_score = self.metrics_calculator.compute_lpips(pred_tensor, gt_tensor)
            metrics["lpips"] = float(lpips_score)
        except Exception as e:
            metrics["lpips_error"] = str(e)
        
        try:
            ssim_score = self.metrics_calculator.compute_ssim(pred_tensor, gt_tensor)
            metrics["ssim"] = float(ssim_score)
        except Exception as e:
            metrics["ssim_error"] = str(e)
        
        try:
            psnr_score = self.metrics_calculator.compute_psnr(pred_tensor, gt_tensor)
            metrics["psnr"] = float(psnr_score)
        except Exception as e:
            metrics["psnr_error"] = str(e)
        
        try:
            # CLIP score requires instruction text
            if instruction:
                clip_score = self.metrics_calculator.compute_clip_score(pred_tensor, [instruction])
                metrics["clip_score"] = float(clip_score)
            else:
                metrics["clip_score"] = 0.0  # No instruction provided
        except Exception as e:
            metrics["clip_error"] = str(e)
        
        # GPT-4o Judge (if original image and instruction provided)
        if original_image is not None and instruction:
            try:
                gpt_scores = self.metrics_calculator.compute_gpt_judge_score(
                    predicted,
                    original_image,
                    instruction
                )
                metrics.update(gpt_scores)
            except Exception as e:
                print(f"    ⚠️ GPT judge error: {e}")
                metrics["gpt_judge_error"] = str(e)
        
        return metrics
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor in range [-1, 1]."""
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return img_tensor
    
    def _save_comparison(
        self, 
        original: Image.Image,
        predicted: Image.Image,
        ground_truth: Image.Image,
        output_path: Path,
        overall_instruction: str = None,
        baseline: Image.Image = None,
        trained_planner_only: Image.Image = None,
        predicted_label: str = "Predicted"
    ):
        """Save side-by-side comparison image (supports 2-4 images dynamically)."""
        # Resize all to same size
        size = (384, 384)
        
        # Build list of images and labels dynamically
        images = [("Original", original)]
        
        if baseline:
            images.append(("Baseline", baseline))
        if predicted:
            images.append((predicted_label, predicted))
        if trained_planner_only:
            images.append(("Trained Planner", trained_planner_only))
        
        # Always add ground truth at the end
        images.append(("Ground Truth", ground_truth))
        
        # Resize all images
        resized_images = [(label, img.resize(size, Image.Resampling.LANCZOS)) for label, img in images]
        num_images = len(resized_images)
        
        # Add space at top and bottom for labels and instruction
        top_margin = 40
        bottom_margin = 80 if overall_instruction else 10
        
        # Create comparison image with extra space
        comparison = Image.new('RGB', (size[0] * num_images, size[1] + top_margin + bottom_margin), color='white')
        
        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Paste images and add labels dynamically
        for idx, (label, img) in enumerate(resized_images):
            x_pos = idx * size[0]
            comparison.paste(img, (x_pos, top_margin))
            
            # Center the label text
            # Estimate text width (rough approximation)
            text_width = len(label) * 10
            text_x = x_pos + (size[0] - text_width) // 2
            draw.text((text_x, 5), label, fill="black", font=font_large)
        
        # Add overall instruction at the bottom if provided
        if overall_instruction:
            instruction_y = size[1] + top_margin + 10
            # Truncate if too long
            max_chars = 150 if num_images == 4 else 120
            if len(overall_instruction) > max_chars:
                overall_instruction = overall_instruction[:max_chars-3] + "..."
            draw.text((10, instruction_y), f"Instruction: {overall_instruction}", fill="black", font=font_small)
        
        comparison.save(output_path)
    
    def calculate_planner_metrics(self, predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Calculate planner evaluation metrics."""
        metrics = {}
        
        # 1. Action Selection Accuracy
        pred_actions = {a["action_id"] for a in predicted.get("actions", [])}
        gt_actions = {a["action_id"] for a in ground_truth.get("actions", [])}
        
        if len(gt_actions) > 0:
            intersection = pred_actions & gt_actions
            union = pred_actions | gt_actions
            
            metrics["action_precision"] = len(intersection) / len(pred_actions) if pred_actions else 0
            metrics["action_recall"] = len(intersection) / len(gt_actions)
            metrics["action_f1"] = (2 * metrics["action_precision"] * metrics["action_recall"] / 
                                   (metrics["action_precision"] + metrics["action_recall"]))                                    if (metrics["action_precision"] + metrics["action_recall"]) > 0 else 0
            metrics["action_iou"] = len(intersection) / len(union) if union else 0
        
        # 2. Priority Correlation (Spearman)
        pred_priorities = {a["action_id"]: a.get("priority", 1) 
                          for a in predicted.get("actions", [])}
        gt_priorities = {a["action_id"]: a.get("priority", 1) 
                        for a in ground_truth.get("actions", [])}
        
        common_actions = pred_actions & gt_actions
        if len(common_actions) >= 2:
            from scipy.stats import spearmanr
            pred_ranks = [pred_priorities[a] for a in sorted(common_actions)]
            gt_ranks = [gt_priorities[a] for a in sorted(common_actions)]
            correlation, _ = spearmanr(pred_ranks, gt_ranks)
            # Handle NaN (can occur with constant/degenerate ranks)
            if np.isnan(correlation):
                metrics["priority_correlation"] = 0.0
            else:
                metrics["priority_correlation"] = float(correlation)
        else:
            metrics["priority_correlation"] = 0.0
        
        # 3. JSON Validity
        metrics["valid_json"] = 1.0 if isinstance(predicted, dict) else 0.0
        metrics["has_actions"] = 1.0 if "actions" in predicted and len(predicted["actions"]) > 0 else 0.0
        
        # 4. Number of actions
        metrics["num_predicted_actions"] = len(predicted.get("actions", []))
        metrics["num_ground_truth_actions"] = len(ground_truth.get("actions", []))
        
        return metrics
    
    def calculate_metrics(self, predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Backward compatibility: redirect to calculate_planner_metrics."""
        return self.calculate_planner_metrics(predicted, ground_truth)
    
    def evaluate_dataset(
        self, 
        data_path: str,
        num_samples: int = None, 
        split: str = "val",
        output_dir: Optional[Path] = None,
        save_images: bool = False,
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset.
        
        Args:
            data_path: Path to training data JSON
            num_samples: Number of samples to evaluate (None = all)
            split: Which split to use (train/val/all)
            output_dir: Where to save outputs
            save_images: Whether to save generated images
            sample_ids: Optional list of specific sample IDs to evaluate (overrides num_samples)
            
        Returns:
            Dictionary with aggregated results
        """
        print(f"\n📊 Evaluating on {split} set...")
        if self.end_to_end:
            print(f"✏️  IMAGE EDITOR: {self.model_editor.upper()} ({'Qwen-Image-Edit 20B' if self.model_editor == 'qwen' else 'HiDream-E1 17B'})")
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Handle both dict with "samples" key and direct list
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        elif not isinstance(data, list):
            raise ValueError(f"Invalid data format in {data_path}")
        
        # Split data (90/10 train/val as in training)
        split_idx = int(len(data) * 0.9)
        if split == "val":
            eval_data = data[split_idx:]
        elif split == "train":
            eval_data = data[:split_idx]
        else:
            eval_data = data
        
        # Filter by sample IDs if provided (overrides num_samples)
        if sample_ids:
            sample_ids_set = set(sample_ids)
            eval_data = [s for s in eval_data if s.get('id', s.get('sample_id', s.get('metadata', {}).get('folder_name'))) in sample_ids_set]
            print(f"🔍 Filtered to {len(eval_data)} samples matching provided IDs")
        elif num_samples:
            # Limit samples if specified (only if no sample_ids provided)
            eval_data = eval_data[:num_samples]
        
        print(f"📁 Evaluating {len(eval_data)} samples...")
        
        # Create output directory for images
        samples_dir = None
        if save_images and output_dir:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate each sample
        results = []
        all_planner_metrics = []
        all_image_metrics = []
        all_gpt_action_scores = []  # NEW: Collect GPT action scores separately
        
        # For multi-config mode, we'll collect metrics per config
        if self.multi_config:
            baseline_planner = []
            baseline_image = []
            trained_full_planner = []
            trained_full_image = []
            trained_planner_planner = []
            trained_planner_image = []
        
        for sample in tqdm(eval_data, desc="Evaluating"):
            result = self.evaluate_sample(sample, output_dir=samples_dir)
            results.append(result)
            
            # Handle multi-config results (nested structure)
            if self.multi_config:
                if "baseline" in result and result["baseline"].get("planner_metrics"):
                    baseline_planner.append(result["baseline"]["planner_metrics"])
                    if result["baseline"].get("image_metrics"):
                        baseline_image.append(result["baseline"]["image_metrics"])
                
                if "trained_full" in result and result["trained_full"].get("planner_metrics"):
                    trained_full_planner.append(result["trained_full"]["planner_metrics"])
                    if result["trained_full"].get("image_metrics"):
                        trained_full_image.append(result["trained_full"]["image_metrics"])
                
                if "trained_planner" in result and result["trained_planner"].get("planner_metrics"):
                    trained_planner_planner.append(result["trained_planner"]["planner_metrics"])
                    if result["trained_planner"].get("image_metrics"):
                        trained_planner_image.append(result["trained_planner"]["image_metrics"])
            else:
                # Single-config mode (original behavior)
                if "planner_metrics" in result:
                    all_planner_metrics.append(result["planner_metrics"])
                
                if "image_metrics" in result:
                    all_image_metrics.append(result["image_metrics"])
                
                # NEW: Collect GPT action scores
                if "gpt_action_scores" in result and "error" not in result["gpt_action_scores"]:
                    all_gpt_action_scores.append(result["gpt_action_scores"])
        
        # Aggregate metrics
        if self.multi_config:
            # Aggregate per configuration
            aggregated_baseline = {
                "planner": self.aggregate_metrics(baseline_planner, prefix="planner"),
                "image": self.aggregate_metrics(baseline_image, prefix="image") if baseline_image else {}
            }
            aggregated_trained_full = {
                "planner": self.aggregate_metrics(trained_full_planner, prefix="planner"),
                "image": self.aggregate_metrics(trained_full_image, prefix="image") if trained_full_image else {}
            }
            aggregated_trained_planner = {
                "planner": self.aggregate_metrics(trained_planner_planner, prefix="planner"),
                "image": self.aggregate_metrics(trained_planner_image, prefix="image") if trained_planner_image else {}
            }
            
            # Calculate success counts
            num_successful = len(baseline_planner)  # Baseline planner successes
            num_end_to_end = len(baseline_image)    # Baseline end-to-end successes
            
            return {
                "num_samples": len(eval_data),
                "num_successful": num_successful,
                "num_errors": len([r for r in results if "error" in r and r["error"]]),
                "num_end_to_end_success": num_end_to_end,
                "aggregated_metrics": {
                    "baseline": aggregated_baseline,
                    "trained_full": aggregated_trained_full,
                    "trained_planner": aggregated_trained_planner
                },
                "detailed_results": results
            }
        else:
            # Single-config mode (original behavior)
            aggregated_planner = self.aggregate_metrics(all_planner_metrics, prefix="planner")
            aggregated_image = self.aggregate_metrics(all_image_metrics, prefix="image") if all_image_metrics else {}
            
            # NEW: Aggregate GPT action scores separately
            aggregated_gpt_action = {}
            if all_gpt_action_scores:
                aggregated_gpt_action = self._aggregate_gpt_action_scores(all_gpt_action_scores)
            
            # Pipeline metrics
            pipeline_metrics = {}
            if self.end_to_end:
                gen_times = [r["generation_time"] for r in results if "generation_time" in r]
                inst_lens = [r["instruction_length"] for r in results if "instruction_length" in r]
                
                if gen_times:
                    pipeline_metrics["avg_generation_time"] = sum(gen_times) / len(gen_times)
                    pipeline_metrics["total_generation_time"] = sum(gen_times)
                
                if inst_lens:
                    pipeline_metrics["avg_instruction_length"] = sum(inst_lens) / len(inst_lens)
            
            return {
                "num_samples": len(eval_data),
                "num_successful": len([r for r in results if "planner_metrics" in r]),
                "num_errors": len([r for r in results if "error" in r]),
                "num_end_to_end_success": len([r for r in results if "image_metrics" in r]),
                "aggregated_planner_metrics": aggregated_planner,
                "aggregated_image_metrics": aggregated_image,
                "aggregated_gpt_action_scores": aggregated_gpt_action,  # NEW: Separate GPT scores
                "pipeline_metrics": pipeline_metrics,
                "detailed_results": results
            }
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
        """Aggregate metrics across all samples."""
        if not all_metrics:
            return {}
        
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        
        # Remove error keys
        metric_keys = [k for k in all_keys if not k.endswith("_error")]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m and not isinstance(m[key], str)]
            if values:
                metric_name = f"{prefix}_{key}" if prefix else key
                aggregated[f"{metric_name}_mean"] = sum(values) / len(values)
                aggregated[f"{metric_name}_min"] = min(values)
                aggregated[f"{metric_name}_max"] = max(values)
                aggregated[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
        
        return aggregated
    
    def _aggregate_gpt_action_scores(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate GPT-4o action scores across samples."""
        if not all_scores:
            return {}
        
        aggregated = {}
        
        # GPT score dimensions
        dimensions = ["relevance", "completeness", "efficiency", "correctness", "overall_score"]
        
        for dim in dimensions:
            values = [s[dim] for s in all_scores if dim in s and isinstance(s[dim], (int, float))]
            if values:
                aggregated[f"{dim}_mean"] = sum(values) / len(values)
                aggregated[f"{dim}_min"] = min(values)
                aggregated[f"{dim}_max"] = max(values)
                aggregated[f"{dim}_std"] = np.std(values) if len(values) > 1 else 0.0
        
        aggregated["num_evaluated"] = len(all_scores)
        
        return aggregated


def main():
    parser = argparse.ArgumentParser(description="End-to-End Planner Evaluation")
    
    # Planner arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints/qwen3_vl_action_planner/final",
        help="Path to trained planner checkpoint"
    )
    
    # Image Editor arguments (for end-to-end eval)
    parser.add_argument(
        "--model-editor",
        type=str,
        required=True,
        choices=["qwen", "hidream"],
        help='Image editor to use: "qwen" (Qwen-Image-Edit) or "hidream" (HiDream-E1) - REQUIRED'
    )
    parser.add_argument(
        "--hidream-checkpoint",
        type=str,
        default=None,
        help="Path to HiDream checkpoint (REQUIRED if --model-editor is hidream)"
    )
    parser.add_argument(
        "--hidream-config",
        type=str,
        default=None,
        help="Path to HiDream training config"
    )
    
    # Multi-config evaluation arguments
    parser.add_argument(
        "--multi-config",
        action="store_true",
        help="Run 3-way comparison (baseline, trained full, trained planner only)"
    )
    parser.add_argument(
        "--base-hidream",
        type=str,
        default=None,
        help="Path to base (untrained) HiDream model for multi-config mode"
    )
    parser.add_argument(
        "--eval-configs",
        type=str,
        nargs="+",
        default=["baseline", "trained_full", "trained_planner"],
        choices=["baseline", "trained_full", "trained_planner"],
        help="Which configurations to evaluate in multi-config mode (default: all three)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "training_data/standard/planner_training_data.json",
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="Which split to evaluate"
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated sample IDs to evaluate (overrides num_samples)"
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default=None,
        help="File with sample IDs to evaluate, one per line (overrides num_samples)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save detailed predictions"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images and comparisons (end-to-end mode)"
    )
    
    # Device argument
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    # GPT-4o Action Judge argument (NEW)
    parser.add_argument(
        "--use-gpt-judge-action",
        action="store_true",
        help="Use GPT-4o to judge action plan quality (generates separate table, may incur costs)"
    )
    # Legacy alias
    parser.add_argument(
        "--use-gpt-judge",
        action="store_true",
        dest="use_gpt_judge_action",
        help=argparse.SUPPRESS  # Hidden alias for backward compatibility
    )
    
    args = parser.parse_args()
    
    # DEBUG: Print args
    print(f"🐛 DEBUG: Parsed arguments:")
    print(f"   --use-gpt-judge-action: {args.use_gpt_judge_action}")
    print(f"   --checkpoint: {args.checkpoint}")
    print(f"   --output: {args.output}")
    
    # Convert num_samples to int if provided
    num_samples = int(args.num_samples) if args.num_samples else None
    
    # Handle sample IDs
    sample_ids = None
    if args.sample_ids_file:
        # Load from file
        with open(args.sample_ids_file, 'r') as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        print(f"📋 Loaded {len(sample_ids)} sample IDs from {args.sample_ids_file}")
    elif args.sample_ids:
        # Parse from command line
        sample_ids = [sid.strip() for sid in args.sample_ids.split(',')]
        print(f"📋 Using {len(sample_ids)} sample IDs from command line")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW: Check for existing results (batch mode detection)
    # ═══════════════════════════════════════════════════════════════════════════════
    summary_file = output_dir / f"evaluation_summary_{args.split}.json"
    detailed_file = output_dir / f"detailed_results_{args.split}.json"
    
    existing_summary, existing_detailed, existing_gpt = None, [], None
    
    if summary_file.exists() or detailed_file.exists():
        print("\n" + "="*70)
        print("📂 BATCH MODE DETECTED")
        print("="*70)
        print(f"   Found existing evaluation results in {output_dir}")
        
        # Load existing data
        existing_summary, existing_detailed, existing_gpt = load_existing_results(
            output_dir, args.split
        )
        
        print(f"   Existing samples: {len(existing_detailed)}")
        if sample_ids:
            print(f"   New samples in this batch: {len(sample_ids)}")
        elif num_samples:
            print(f"   New samples in this batch: {num_samples}")
        print(f"   Will merge results after evaluation")
        print("="*70 + "\n")
    else:
        print(f"\n✨ Creating new evaluation results\n")
    
    # Initialize evaluator
    evaluator = PlannerEvaluator(
        checkpoint_path=args.checkpoint,
        model_editor=args.model_editor,
        hidream_checkpoint=args.hidream_checkpoint,
        hidream_config=args.hidream_config,
        device=args.device,
        multi_config=args.multi_config,
        base_hidream_path=args.base_hidream,
        eval_configs=args.eval_configs
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        data_path=args.data,
        num_samples=num_samples,
        split=args.split,
        output_dir=output_dir,
        save_images=args.save_images,
        sample_ids=sample_ids
    )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW: Merge with existing results (if batch mode)
    # ═══════════════════════════════════════════════════════════════════════════════
    if existing_detailed:
        print(f"\n{'='*70}")
        print("🔄 MERGING BATCH RESULTS")
        print(f"{'='*70}")
        
        # Merge detailed results
        combined_detailed = merge_detailed_results(existing_detailed, results['detailed_results'])
        print(f"   Previous batches: {len(existing_detailed)} samples")
        print(f"   This batch: {len(results['detailed_results'])} samples")
        print(f"   Combined total: {len(combined_detailed)} samples")
        
        # Recalculate aggregates from ALL samples
        print(f"   Recalculating aggregate metrics...")
        recalculated = recalculate_aggregates_from_detailed(combined_detailed, args.multi_config)
        
        # Update results dictionary
        results['detailed_results'] = combined_detailed
        results['num_samples'] = recalculated['num_samples']
        results['num_successful'] = recalculated['num_successful']
        results['num_errors'] = recalculated['num_errors']
        results['num_end_to_end_success'] = recalculated['num_end_to_end_success']
        
        if args.multi_config:
            results['aggregated_metrics'] = recalculated['aggregated_metrics']
        else:
            results['aggregated_planner_metrics'] = recalculated['aggregated_planner_metrics']
            results['aggregated_image_metrics'] = recalculated['aggregated_image_metrics']
            results['aggregated_gpt_action_scores'] = recalculated['aggregated_gpt_action_scores']
        
        print(f"✅ Merge complete!")
        print(f"{'='*70}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"\n✅ Successfully evaluated: {results['num_successful']}/{results['num_samples']} samples")
    print(f"❌ Errors: {results['num_errors']}")
    
    # Handle multi-config vs single-config output
    if args.multi_config and 'aggregated_metrics' in results:
        # Multi-config mode: show all 3 configurations
        for config_name in ["baseline", "trained_full", "trained_planner"]:
            config_metrics = results['aggregated_metrics'][config_name]
            
            print(f"\n{'='*70}")
            print(f"📊 {config_name.upper().replace('_', ' ')} CONFIGURATION")
            print(f"{'='*70}")
            
            # Print planner metrics
            if config_metrics['planner']:
                print("\n📈 Planner Metrics:")
                planner_metrics = config_metrics['planner']
                for key, value in sorted(planner_metrics.items()):
                    if "_mean" in key:
                        metric_name = key.replace("planner_", "").replace("_mean", "")
                        mean_val = value
                        min_val = planner_metrics.get(key.replace("_mean", "_min"), 0)
                        max_val = planner_metrics.get(key.replace("_mean", "_max"), 0)
                        print(f"  {metric_name:30s}: {mean_val:.4f}  (min: {min_val:.4f}, max: {max_val:.4f})")
            
            # Print image metrics
            if config_metrics['image']:
                print(f"\n📸 Image Quality Metrics:")
                image_metrics = config_metrics['image']
                for key, value in sorted(image_metrics.items()):
                    if "_mean" in key:
                        metric_name = key.replace("image_", "").replace("_mean", "")
                        mean_val = value
                        min_val = image_metrics.get(key.replace("_mean", "_min"), 0)
                        max_val = image_metrics.get(key.replace("_mean", "_max"), 0)
                        
                        # Add interpretation hints
                        hint = ""
                        if "lpips" in metric_name:
                            hint = " (lower is better)"
                        elif "ssim" in metric_name or "psnr" in metric_name or "clip" in metric_name:
                            hint = " (higher is better)"
                        
                        print(f"  {metric_name:30s}: {mean_val:.4f}  (min: {min_val:.4f}, max: {max_val:.4f}){hint}")
    else:
        # Single-config mode (original behavior)
        print("\n📈 Planner Metrics:")
        planner_metrics = results['aggregated_planner_metrics']
        for key, value in sorted(planner_metrics.items()):
            if "_mean" in key:
                metric_name = key.replace("planner_", "").replace("_mean", "")
                mean_val = value
                min_val = planner_metrics.get(key.replace("_mean", "_min"), 0)
                max_val = planner_metrics.get(key.replace("_mean", "_max"), 0)
                print(f"  {metric_name:30s}: {mean_val:.4f}  (min: {min_val:.4f}, max: {max_val:.4f})")
        
        # Print image metrics if available
        if results['aggregated_image_metrics']:
            print(f"\n📸 Image Quality Metrics:")
            print(f"   Successfully generated: {results['num_end_to_end_success']}/{results['num_successful']} samples")
            
            image_metrics = results['aggregated_image_metrics']
            for key, value in sorted(image_metrics.items()):
                if "_mean" in key:
                    metric_name = key.replace("image_", "").replace("_mean", "")
                    mean_val = value
                    min_val = image_metrics.get(key.replace("_mean", "_min"), 0)
                    max_val = image_metrics.get(key.replace("_mean", "_max"), 0)
                    
                    # Add interpretation hints
                    hint = ""
                    if "lpips" in metric_name:
                        hint = " (lower is better)"
                    elif "ssim" in metric_name or "psnr" in metric_name or "clip" in metric_name:
                        hint = " (higher is better)"
                    
                    print(f"  {metric_name:30s}: {mean_val:.4f}  (min: {min_val:.4f}, max: {max_val:.4f}){hint}")
        
        # Print pipeline metrics if available
        if results.get('pipeline_metrics'):
            print(f"\n⏱️ Pipeline Metrics:")
            pipeline_metrics = results['pipeline_metrics']
            for key, value in sorted(pipeline_metrics.items()):
                print(f"  {key:30s}: {value:.2f}")
        
        # NEW: Print GPT-4o action scores if available
        if results.get('aggregated_gpt_action_scores'):
            print(f"\n🤖 GPT-4o Action Plan Quality Scores (0-10 scale):")
            gpt_scores = results['aggregated_gpt_action_scores']
            print(f"   Evaluated: {gpt_scores.get('num_evaluated', 0)}/{results['num_successful']} samples")
            
            for dim in ["relevance", "completeness", "efficiency", "correctness", "overall_score"]:
                if f"{dim}_mean" in gpt_scores:
                    mean_val = gpt_scores[f"{dim}_mean"]
                    min_val = gpt_scores.get(f"{dim}_min", 0)
                    max_val = gpt_scores.get(f"{dim}_max", 0)
                    print(f"  {dim:15s}: {mean_val:.2f}/10  (min: {min_val:.2f}, max: {max_val:.2f})")
    
    # Save results
    summary_path = output_dir / f"evaluation_summary_{args.split}.json"
    
    if args.multi_config:
        # Save multi-config structure
        with open(summary_path, 'w') as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "hidream_checkpoint": args.hidream_checkpoint,
                "base_hidream": args.base_hidream,
                "data_path": args.data,
                "split": args.split,
                "num_samples": results["num_samples"],
                "num_successful": results["num_successful"],
                "num_errors": results["num_errors"],
                "num_end_to_end_success": results.get("num_end_to_end_success", 0),
                "aggregated_metrics": results["aggregated_metrics"]
            }, f, indent=2)
    else:
        # Save single-config structure (original)
        with open(summary_path, 'w') as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "hidream_checkpoint": args.hidream_checkpoint,
                "data_path": args.data,
                "split": args.split,
                "num_samples": results["num_samples"],
                "num_successful": results["num_successful"],
                "num_errors": results["num_errors"],
                "num_end_to_end_success": results.get("num_end_to_end_success", 0),
                "aggregated_planner_metrics": results.get("aggregated_planner_metrics", {}),
                "aggregated_image_metrics": results.get("aggregated_image_metrics", {}),
                "pipeline_metrics": results.get("pipeline_metrics", {})
            }, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    # NEW: Save GPT-4o action scores to SEPARATE file (if available)
    if results.get("aggregated_gpt_action_scores"):
        gpt_scores_path = output_dir / f"gpt4o_action_scores_{args.split}.json"
        with open(gpt_scores_path, 'w') as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "data_path": args.data,
                "split": args.split,
                "num_samples": results["num_samples"],
                "num_evaluated": results["aggregated_gpt_action_scores"].get("num_evaluated", 0),
                "aggregated_scores": results["aggregated_gpt_action_scores"]
            }, f, indent=2)
        print(f"💾 GPT-4o action scores saved to: {gpt_scores_path}")
    
    # Save detailed results if requested
    if args.save_predictions:
        detailed_path = output_dir / f"detailed_results_{args.split}.json"
        # Don't save full results if too large - just save sample IDs and metrics
        simplified_results = []
        
        if args.multi_config:
            # Multi-config: extract metrics from nested structure
            for r in results["detailed_results"]:
                simplified = {
                    "sample_id": r.get("sample_id"),
                    "user_prompt": r.get("user_prompt"),
                    "baseline": {
                        "planner_metrics": r.get("baseline", {}).get("planner_metrics"),
                        "image_metrics": r.get("baseline", {}).get("image_metrics")
                    },
                    "trained_full": {
                        "planner_metrics": r.get("trained_full", {}).get("planner_metrics"),
                        "image_metrics": r.get("trained_full", {}).get("image_metrics")
                    },
                    "trained_planner": {
                        "planner_metrics": r.get("trained_planner", {}).get("planner_metrics"),
                        "image_metrics": r.get("trained_planner", {}).get("image_metrics")
                    },
                    "gpt_action_scores": r.get("gpt_action_scores"),
                    "error": r.get("error")
                }
                simplified_results.append(simplified)
        else:
            # Single-config: original format
            for r in results["detailed_results"]:
                simplified = {
                    "sample_id": r.get("sample_id"),
                    "user_prompt": r.get("user_prompt"),
                    "planner_metrics": r.get("planner_metrics"),
                    "image_metrics": r.get("image_metrics"),
                    "gpt_action_scores": r.get("gpt_action_scores"),
                    "error": r.get("error"),
                    "end_to_end_error": r.get("end_to_end_error")
                }
                simplified_results.append(simplified)
        
        with open(detailed_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print(f"💾 Detailed results saved to: {detailed_path}")
    
    # Show where images were saved
    if args.save_images and results.get("num_end_to_end_success", 0) > 0:
        samples_dir = output_dir / "samples"
        print(f"\n🖼️  Generated images saved to: {samples_dir}")
        print(f"   View comparisons: ls {samples_dir}/*/comparison.png")
    
    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

