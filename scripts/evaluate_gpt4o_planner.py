#!/usr/bin/env python3
"""
GPT-4o Planner Evaluation with Batch Mode Support

This script evaluates the GPT-4o Vision planner using the same evaluation pipeline
as other planners. It's adapted from evaluate_planner_batch.py but uses API calls
instead of local model loading.

Key Differences from evaluate_planner_batch.py:
- Uses GPT4oPlanner instead of ActionPlanner
- No GPU/device management (API-based)
- Adds rate limiting between samples
- No checkpoint loading

Usage:
    python scripts/evaluate_gpt4o_planner.py \
        --data training_data/cot_8b_trajectory/full_dataset_for_eval.json \
        --output evaluation_results/gpt4o \
        --sample-ids-file selected_samples.txt \
        --save-images \
        --save-predictions
"""

import os
import sys
import json
import logging
import argparse
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from actions.gpt4o_planner import GPT4oPlanner
from training.models import HiDreamE1LoRA
from training.evaluation.metrics import MetricsCalculator
from training.evaluation.gpt_action_judge import GPT4oActionJudge
from training.evaluation.gpt_judge import GPT4oJudge
from src.reward_model_evaluator import RewardModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class GPT4oPlannerEvaluator:
    """Evaluator for GPT-4o planner with optional image generation."""
    
    def __init__(
        self,
        azure_config: Optional[Dict] = None,
        image_editor_type: str = "qwen",
        hidream_checkpoint: Optional[str] = None,
        hidream_config: Optional[str] = None,
        save_images: bool = False,
        save_predictions: bool = False,
        use_gpt_action_judge: bool = True,
        rate_limit_delay: float = 0.5,
        action_library_path: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            azure_config: Azure OpenAI credentials
            image_editor_type: "qwen" or "hidream"
            hidream_checkpoint: Path to HiDream checkpoint (if using hidream)
            hidream_config: Path to HiDream config (if using hidream)
            save_images: Whether to save generated images
            save_predictions: Whether to save predictions
            use_gpt_action_judge: Whether to use GPT-4o action judge
            rate_limit_delay: Delay between API calls (seconds)
            action_library_path: Path to action library JSON (for custom themes like complex)
        """
        # Initialize GPT-4o planner
        self.planner = GPT4oPlanner(azure_config=azure_config, action_library_path=action_library_path)
        
        # Image editor setup
        self.image_editor_type = image_editor_type
        self.save_images = save_images
        self.save_predictions = save_predictions
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize image editor if requested
        self.editor = None
        self.editor_type = image_editor_type
        if save_images:
            if image_editor_type == "hidream":
                if hidream_checkpoint is None:
                    raise ValueError("--hidream-checkpoint required when using hidream editor")
                from omegaconf import OmegaConf
                import safetensors.torch
                logger.info(f"Loading HiDream editor from {hidream_checkpoint}")
                
                # Load config
                if hidream_config:
                    config = OmegaConf.load(hidream_config)
                else:
                    config = OmegaConf.create({})
                
                # Load model
                self.editor = HiDreamE1LoRA(checkpoint_path=hidream_checkpoint, config=config)
                logger.info("✅ HiDream editor loaded")
            elif image_editor_type == "qwen":
                from diffusers import DiffusionPipeline
                import torch
                logger.info("Loading Qwen-Image-Edit (20B parameters)...")
                self.editor = DiffusionPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit",
                    torch_dtype=torch.bfloat16,
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                self.editor.set_progress_bar_config(disable=True)
                logger.info("✅ Qwen-Image-Edit loaded")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize GPT action judge (if enabled)
        self.use_gpt_action_judge = use_gpt_action_judge
        self.gpt_action_judge = None
        if use_gpt_action_judge:
            self.gpt_action_judge = GPT4oActionJudge(azure_config=azure_config)
        
        # Initialize GPT-4o Image Judge (if enabled)
        self.use_gpt_image_judge = use_gpt_action_judge  # Same flag controls both
        self.gpt_image_judge = None
        if self.use_gpt_image_judge:
            try:
                self.gpt_image_judge = GPT4oJudge()
                logger.info("✅ GPT-4o Image Judge initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GPT-4o Image Judge: {e}")
                self.use_gpt_image_judge = False
        
        # Initialize reward evaluator (lazy loading to save memory)
        self.reward_evaluator = None  # Will be initialized on first use
    
    def calculate_planner_metrics(self, predicted_plan: Dict, ground_truth_plan: Dict) -> Dict[str, float]:
        """Calculate planner metrics comparing predicted vs ground truth plans."""
        metrics = {}
        
        # Extract action sets
        pred_actions = set(a["action_id"] for a in predicted_plan.get("actions", []))
        gt_actions = set(a["action_id"] for a in ground_truth_plan.get("actions", []))
        
        # Action selection metrics
        if len(pred_actions) == 0 and len(gt_actions) == 0:
            metrics["precision"] = 1.0
            metrics["recall"] = 1.0
            metrics["f1"] = 1.0
        elif len(pred_actions) == 0:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        elif len(gt_actions) == 0:
            metrics["precision"] = 0.0
            metrics["recall"] = 1.0  # All predictions are false positives
            metrics["f1"] = 0.0
        else:
            tp = len(pred_actions & gt_actions)
            fp = len(pred_actions - gt_actions)
            fn = len(gt_actions - pred_actions)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            metrics["true_positives"] = tp
            metrics["false_positives"] = fp
            metrics["false_negatives"] = fn
        
        # IoU (Jaccard similarity)
        union = len(pred_actions | gt_actions)
        metrics["iou"] = len(pred_actions & gt_actions) / union if union > 0 else 0.0
        
        # Action counts
        metrics["num_predicted_actions"] = len(pred_actions)
        metrics["num_ground_truth_actions"] = len(gt_actions)
        
        return metrics
    
    def _pil_to_tensor(self, image: Image.Image):
        """Convert PIL Image to tensor."""
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
        ])
        return transform(image)
    
    def _calculate_image_metrics(
        self, 
        predicted: Image.Image, 
        ground_truth: Image.Image, 
        original_image: Image.Image = None
    ) -> Dict[str, float]:
        """Calculate image quality metrics."""
        import torch
        
        # Resize images to match dimensions
        if predicted.size != ground_truth.size:
            logger.info(f"    ⚠️ Resizing predicted image from {predicted.size} to {ground_truth.size}")
            predicted = predicted.resize(ground_truth.size, Image.Resampling.LANCZOS)
        
        # Convert to tensors
        pred_tensor = self._pil_to_tensor(predicted).unsqueeze(0)
        gt_tensor = self._pil_to_tensor(ground_truth).unsqueeze(0)
        
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
        
        return metrics
    
    def _save_comparison(
        self, 
        original: Image.Image,
        predicted: Image.Image,
        ground_truth: Image.Image,
        output_path: Path,
        overall_instruction: str = None,
        predicted_label: str = "Predicted"
    ):
        """Save side-by-side comparison image (Original | Predicted | Ground Truth)."""
        # Resize all to same size
        size = (384, 384)
        
        # Build list of images and labels
        images = [
            ("Original", original),
            (predicted_label, predicted),
            ("Ground Truth", ground_truth)
        ]
        
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
        
        # Paste images and add labels
        for idx, (label, img) in enumerate(resized_images):
            x_pos = idx * size[0]
            comparison.paste(img, (x_pos, top_margin))
            
            # Center the label text
            text_width = len(label) * 10
            text_x = x_pos + (size[0] - text_width) // 2
            draw.text((text_x, 5), label, fill="black", font=font_large)
        
        # Add overall instruction at the bottom if provided
        if overall_instruction:
            instruction_y = size[1] + top_margin + 10
            # Truncate if too long
            max_chars = 120
            if len(overall_instruction) > max_chars:
                overall_instruction = overall_instruction[:max_chars-3] + "..."
            draw.text((10, instruction_y), f"Instruction: {overall_instruction}", fill="black", font=font_small)
        
        comparison.save(output_path)
    
    def evaluate_sample(self, sample: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate a single sample with GPT-4o planner.
        
        Args:
            sample: Sample dictionary from training data
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        # Get paths
        image_path = Path(project_root) / sample["original_image_path"]
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}
        
        # Get user instruction
        user_instruction = sample.get("overall_instruction", sample.get("user_prompt", ""))
        sample_id = sample["sample_id"]
        
        # Step 1: Predict action plan with GPT-4o
        logger.info(f"  Predicting action plan for {sample_id}...")
        try:
            predicted_plan = self.planner.predict_action_plan(
                image_path=str(image_path),
                user_prompt=user_instruction,
                temperature=0.1
            )
        except Exception as e:
            import traceback
            logger.error(f"❌ Prediction error for {sample_id}:")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
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
        
        # Step 2.5: GPT-4o Action Judge (if enabled)
        if self.use_gpt_action_judge and self.gpt_action_judge is not None:
            logger.info(f"  Running GPT-4o Action Judge for {sample_id}...")
            try:
                original_image = Image.open(image_path).convert("RGB")
                gpt_action_scores = self.gpt_action_judge.judge_action_plan(
                    original_image=original_image,
                    user_prompt=user_instruction,
                    predicted_plan=predicted_plan,
                    teacher_plan=ground_truth_plan
                )
                result["gpt_action_scores"] = gpt_action_scores
                logger.info(f"  ✅ GPT Action Judge completed")
            except Exception as e:
                logger.warning(f"  ⚠️  GPT action judge failed: {e}")
                result["gpt_action_scores"] = {"error": str(e)}
        
        # Step 3: End-to-end evaluation (if editor loaded)
        if self.editor is not None and self.save_images:
            logger.info(f"  Generating edited image for {sample_id}...")
            try:
                # PREFER: Model-generated hidream_prompt (if available)
                # FALLBACK: overall_instruction or user_prompt
                if "hidream_prompt" in predicted_plan and predicted_plan["hidream_prompt"]:
                    edit_instruction = predicted_plan["hidream_prompt"]
                    logger.info(f"    Using model-generated hidream_prompt")
                else:
                    edit_instruction = predicted_plan.get("overall_instruction", user_instruction)
                    logger.warning(f"    hidream_prompt not found in predicted_plan, using overall_instruction fallback")
                
                # Generate edited image
                original_image = Image.open(image_path).convert("RGB")
                if self.editor_type == "hidream":
                    edited_image = self.editor.edit_image(original_image, edit_instruction)
                else:  # qwen
                    # Qwen uses DiffusionPipeline API
                    output = self.editor(
                        prompt=edit_instruction,
                        image=original_image,
                        height=768,
                        width=768,
                        num_inference_steps=28,
                        guidance_scale=7.5,
                    )
                    edited_image = output.images[0]
                
                # Calculate image metrics
                gt_image_path = Path(project_root) / sample["edited_image_path"]
                if gt_image_path.exists():
                    gt_image = Image.open(gt_image_path).convert("RGB")
                    image_metrics = self._calculate_image_metrics(
                        edited_image, gt_image, original_image
                    )
                    result["image_metrics"] = image_metrics
                
                # Save images if output_dir provided
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save predicted edit
                    edited_image.save(output_dir / "predicted_edit.png")
                    
                    # Copy original and ground truth
                    shutil.copy2(image_path, output_dir / "original.png")
                    if gt_image_path.exists():
                        shutil.copy2(gt_image_path, output_dir / "ground_truth.png")
                    
                    # Create comparison image (Original | Predicted | Ground Truth)
                    original_image = Image.open(image_path).convert("RGB")
                    gt_image = Image.open(gt_image_path).convert("RGB") if gt_image_path.exists() else edited_image
                    self._save_comparison(
                        original=original_image,
                        predicted=edited_image,
                        ground_truth=gt_image,
                        output_path=output_dir / "comparison.png",
                        overall_instruction=user_instruction,
                        predicted_label="GPT-4o Predicted"
                    )
                
                logger.info(f"  ✅ Image generation complete")
                
                # Step 3.5: GPT-4o Image Judge (if enabled)
                if self.use_gpt_image_judge and self.gpt_image_judge is not None:
                    logger.info(f"  Running GPT-4o Image Judge for {sample_id}...")
                    try:
                        gpt_image_scores = self.gpt_image_judge.judge_single_edit(
                            original_image=original_image,
                            generated_image=edited_image,
                            instruction=edit_instruction
                        )
                        
                        # Add to image_metrics (if they exist)
                        if "image_metrics" in result:
                            result["image_metrics"]["gpt_judge_instruction_following"] = gpt_image_scores.get("instruction_following", 0.0)
                            result["image_metrics"]["gpt_judge_visual_quality"] = gpt_image_scores.get("visual_quality", 0.0)
                            result["image_metrics"]["gpt_judge_overall"] = gpt_image_scores.get("overall_image_score", 0.0)
                        else:
                            result["image_metrics"] = {
                                "gpt_judge_instruction_following": gpt_image_scores.get("instruction_following", 0.0),
                                "gpt_judge_visual_quality": gpt_image_scores.get("visual_quality", 0.0),
                                "gpt_judge_overall": gpt_image_scores.get("overall_image_score", 0.0)
                            }
                        
                        logger.info(f"  ✅ GPT Image Judge completed")
                    except Exception as e:
                        logger.warning(f"  ⚠️  GPT image judge failed: {e}")
                
                # Step 4: Compute reward scores with Qwen3-VL-8B (after image generation)
                logger.info(f"  Computing Qwen3-VL-8B reward scores...")
                try:
                    # Initialize reward evaluator on first use (lazy loading)
                    if self.reward_evaluator is None:
                        logger.info(f"  🏆 Initializing Qwen3-VL-8B Reward Evaluator (one-time setup)...")
                        self.reward_evaluator = RewardModelEvaluator()
                        logger.info(f"  ✅ Reward evaluator ready")
                    
                    # Load analysis.json if available
                    analysis = None
                    analysis_path = Path(project_root) / sample.get("analysis_path", "")
                    if analysis_path.exists():
                        with open(analysis_path, 'r') as f:
                            analysis = json.load(f)
                    
                    # Compute 7D reward scores
                    reward_result = self.reward_evaluator.evaluate_transformation(
                        original_image_path=str(image_path),
                        edited_image_path=str(output_dir / "predicted_edit.png"),
                        user_prompt=user_instruction,
                        action_plan=predicted_plan,
                        analysis=analysis
                    )
                    
                    # Save reward scores to separate file (matches other models)
                    if output_dir:
                        with open(output_dir / "reward_scores.json", 'w') as f:
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
                    
                    logger.info(f"  ✅ Reward scores computed and saved")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  Reward evaluation failed: {e}")
                    result["reward_error"] = str(e)
                
            except Exception as e:
                logger.warning(f"  ⚠️  Image generation failed: {e}")
                result["image_generation_error"] = str(e)
        
        # Save predictions if requested
        if self.save_predictions and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predicted plan
            with open(output_dir / "predicted_plan.json", 'w') as f:
                json.dump(predicted_plan, f, indent=2)
            
            # Save full result
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(result, f, indent=2)
        
        # Calculate total time
        result["evaluation_time"] = time.time() - start_time
        
        return result
    
    def evaluate_dataset(
        self,
        samples: List[Dict],
        output_dir: Path,
        split: str = "all"
    ) -> Tuple[Dict, List[Dict]]:
        """
        Evaluate multiple samples.
        
        Args:
            samples: List of sample dictionaries
            output_dir: Output directory for results
            split: Split name (for file naming)
            
        Returns:
            Tuple of (summary_metrics, detailed_results)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples directory
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        logger.info(f"Evaluating {len(samples)} samples...")
        logger.info(f"Rate limit delay: {self.rate_limit_delay}s between samples")
        
        detailed_results = []
        all_planner_metrics = []
        all_image_metrics = []
        all_gpt_action_scores = []
        
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            sample_id = sample["sample_id"]
            sample_output_dir = samples_dir / sample_id
            
            # Evaluate sample
            result = self.evaluate_sample(sample, sample_output_dir)
            
            # Skip if error
            if "error" in result:
                logger.warning(f"⚠️  Sample {sample_id} failed: {result['error']}")
                continue
            
            detailed_results.append(result)
            
            # Collect metrics
            if "planner_metrics" in result:
                all_planner_metrics.append(result["planner_metrics"])
            
            if "image_metrics" in result:
                all_image_metrics.append(result["image_metrics"])
            
            if "gpt_action_scores" in result and "error" not in result["gpt_action_scores"]:
                all_gpt_action_scores.append(result["gpt_action_scores"])
            
            # Rate limiting (avoid API throttling)
            if i < len(samples) - 1:  # Don't sleep after last sample
                time.sleep(self.rate_limit_delay)
        
        # Aggregate metrics
        summary = {
            "num_samples": len(detailed_results),
            "split": split,
            "model": "GPT-4o Vision (API)",
            "planner_metrics": self._aggregate_metrics(all_planner_metrics),
        }
        
        if all_image_metrics:
            summary["image_metrics"] = self._aggregate_metrics(all_image_metrics)
        
        if all_gpt_action_scores:
            summary["gpt_action_scores"] = self._aggregate_gpt_scores(all_gpt_action_scores)
        
        # Save results
        summary_file = output_dir / f"evaluation_summary_{split}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        detailed_file = output_dir / f"detailed_results_{split}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"\n✅ Evaluation complete!")
        logger.info(f"   Summary: {summary_file}")
        logger.info(f"   Detailed: {detailed_file}")
        
        return summary, detailed_results
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across samples."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
            if values:
                aggregated[f"{key}_mean"] = sum(values) / len(values)
                aggregated[f"{key}_min"] = min(values)
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
        
        return aggregated
    
    def _aggregate_gpt_scores(self, scores_list: List[Dict]) -> Dict[str, float]:
        """Aggregate GPT-4o action scores."""
        if not scores_list:
            return {}
        
        aggregated = {}
        dimensions = ["relevance", "theme_style_focus", "completeness", "efficiency", 
                     "correctness", "overall_score"]
        
        for dim in dimensions:
            values = [s[dim] for s in scores_list if dim in s and isinstance(s[dim], (int, float))]
            if values:
                aggregated[f"{dim}_mean"] = sum(values) / len(values)
                aggregated[f"{dim}_min"] = min(values)
                aggregated[f"{dim}_max"] = max(values)
                aggregated[f"{dim}_std"] = np.std(values) if len(values) > 1 else 0.0
        
        aggregated["num_evaluated"] = len(scores_list)
        
        return aggregated


def load_samples_from_ids(data_path: str, sample_ids_file: str) -> List[Dict]:
    """Load samples based on sample IDs file."""
    # Read sample IDs
    with open(sample_ids_file, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    
    # Load full dataset
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    
    # Filter by sample IDs
    id_set = set(sample_ids)
    filtered_samples = [s for s in samples if s["sample_id"] in id_set]
    
    logger.info(f"Loaded {len(filtered_samples)} samples (out of {len(sample_ids)} requested IDs)")
    
    return filtered_samples


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Planner Evaluation")
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation data JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default=None,
        help="File with sample IDs to evaluate (one per line)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Which split to evaluate"
    )
    
    # Image editor arguments
    parser.add_argument(
        "--model-editor",
        type=str,
        default="qwen",
        choices=["qwen", "hidream"],
        help="Image editor to use"
    )
    parser.add_argument(
        "--hidream-checkpoint",
        type=str,
        default=None,
        help="Path to HiDream checkpoint (if using hidream)"
    )
    parser.add_argument(
        "--hidream-config",
        type=str,
        default=None,
        help="Path to HiDream config (if using hidream)"
    )
    parser.add_argument(
        "--action-library",
        type=str,
        default=None,
        help="Path to action library JSON (default: action_library_v2.json for normal theme, use action_library_complex.json for complex theme)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Generate and save edited images"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted action plans"
    )
    parser.add_argument(
        "--no-gpt-judge",
        action="store_true",
        help="Disable GPT-4o action judge (enabled by default)"
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    
    # Azure config (optional overrides)
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Azure OpenAI endpoint (optional override)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Azure OpenAI API key (optional override)"
    )
    
    args = parser.parse_args()
    
    # Prepare Azure config (use environment variables or command line args)
    azure_config = None
    if args.azure_endpoint or args.api_key:
        azure_endpoint = args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        
        if not azure_endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. Please set environment variables:\n"
                "  - AZURE_OPENAI_ENDPOINT\n"
                "  - AZURE_OPENAI_API_KEY\n"
                "Or use --azure-endpoint and --api-key arguments\n"
                "Or export the required environment variables: export the required environment variables"
            )
        
        azure_config = {
            "azure_endpoint": azure_endpoint,
            "api_key": api_key,
            "api_version": api_version,
            "model": model,
        }
    
    # Load samples
    if args.sample_ids_file:
        samples = load_samples_from_ids(args.data, args.sample_ids_file)
    else:
        with open(args.data, 'r') as f:
            data = json.load(f)
        
        samples = data.get("samples", [])
        
        # Filter by split if needed
        if args.split != "all":
            samples = [s for s in samples if s.get("split", "val") == args.split]
        
        logger.info(f"Loaded {len(samples)} samples from {args.split} split")
    
    if not samples:
        logger.error("No samples to evaluate!")
        return
    
    # Initialize evaluator
    logger.info("Initializing GPT-4o evaluator...")
    evaluator = GPT4oPlannerEvaluator(
        azure_config=azure_config,
        image_editor_type=args.model_editor,
        hidream_checkpoint=args.hidream_checkpoint,
        hidream_config=args.hidream_config,
        save_images=args.save_images,
        save_predictions=args.save_predictions,
        use_gpt_action_judge=not args.no_gpt_judge,
        rate_limit_delay=args.rate_limit_delay,
        action_library_path=args.action_library
    )
    
    # Run evaluation
    logger.info(f"\nStarting evaluation...")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Save images: {args.save_images}")
    logger.info(f"  Save predictions: {args.save_predictions}")
    logger.info(f"  GPT action judge: {not args.no_gpt_judge}")
    logger.info(f"  Estimated time: ~{len(samples) * 3 / 60:.1f} minutes")
    logger.info(f"  Estimated cost: ~${len(samples) * 0.015:.2f}")
    logger.info("")
    
    summary, detailed = evaluator.evaluate_dataset(
        samples=samples,
        output_dir=Path(args.output),
        split=args.split
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: GPT-4o Vision (API)")
    logger.info(f"Samples evaluated: {summary['num_samples']}")
    logger.info("")
    
    if "planner_metrics" in summary:
        logger.info("Planner Metrics:")
        pm = summary["planner_metrics"]
        logger.info(f"  F1 Score: {pm.get('f1_mean', 0):.3f} ± {pm.get('f1_std', 0):.3f}")
        logger.info(f"  Precision: {pm.get('precision_mean', 0):.3f} ± {pm.get('precision_std', 0):.3f}")
        logger.info(f"  Recall: {pm.get('recall_mean', 0):.3f} ± {pm.get('recall_std', 0):.3f}")
        logger.info(f"  IoU: {pm.get('iou_mean', 0):.3f} ± {pm.get('iou_std', 0):.3f}")
        logger.info("")
    
    if "image_metrics" in summary:
        logger.info("Image Metrics:")
        im = summary["image_metrics"]
        logger.info(f"  LPIPS: {im.get('lpips_mean', 0):.4f} ± {im.get('lpips_std', 0):.4f}")
        logger.info(f"  SSIM: {im.get('ssim_mean', 0):.4f} ± {im.get('ssim_std', 0):.4f}")
        logger.info(f"  PSNR: {im.get('psnr_mean', 0):.2f} ± {im.get('psnr_std', 0):.2f}")
        logger.info(f"  CLIP: {im.get('clip_score_mean', 0):.4f} ± {im.get('clip_score_std', 0):.4f}")
        logger.info("")
    
    if "gpt_action_scores" in summary:
        logger.info("GPT-4o Action Judge Scores:")
        gs = summary["gpt_action_scores"]
        logger.info(f"  Overall: {gs.get('overall_score_mean', 0):.2f} ± {gs.get('overall_score_std', 0):.2f}")
        logger.info(f"  Relevance: {gs.get('relevance_mean', 0):.2f} ± {gs.get('relevance_std', 0):.2f}")
        logger.info(f"  Completeness: {gs.get('completeness_mean', 0):.2f} ± {gs.get('completeness_std', 0):.2f}")
        logger.info(f"  Efficiency: {gs.get('efficiency_mean', 0):.2f} ± {gs.get('efficiency_std', 0):.2f}")
        logger.info(f"  Correctness: {gs.get('correctness_mean', 0):.2f} ± {gs.get('correctness_std', 0):.2f}")
        logger.info("")
    
    logger.info("="*80)
    logger.info(f"✅ Results saved to: {args.output}")
    logger.info("="*80)


if __name__ == "__main__":
    main()


