#!/usr/bin/env python3
"""
Gemini 2.5 Planner Evaluation

This script evaluates the Gemini 2.5 Vision planner using the same evaluation pipeline
as evaluate_gpt4o_planner.py. Uses GeminiPlanner instead of GPT4oPlanner.

Key Differences from evaluate_gpt4o_planner.py:
- Uses GeminiPlanner instead of GPT4oPlanner
- No Azure config needed — uses GOOGLE_API_KEY / GEMINI_API_KEY from credentials.sh
- No checkpoint loading

Usage:
    python scripts/evaluate_gemini_planner.py \
        --data training_data/simple/cot_4b_trajectory/full_dataset_for_eval.json \
        --output evaluation_results/simple/text_parallel_cot_4b_trajectory/gemini25 \
        --sample-ids-file training_data/simple/cot_4b_trajectory/test_samples_cot_4b.txt \
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

from actions.gemini_planner import GeminiPlanner
from training.models import HiDreamE1LoRA
from training.evaluation.metrics import MetricsCalculator
from training.evaluation.gpt_action_judge import GPT4oActionJudge
from training.evaluation.gpt_judge import GPT4oJudge
from src.reward_model_evaluator import RewardModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class GeminiPlannerEvaluator:
    """Evaluator for Gemini 2.5 planner with optional image generation."""

    def __init__(
        self,
        image_editor_type: str = "qwen",
        hidream_checkpoint: Optional[str] = None,
        hidream_config: Optional[str] = None,
        save_images: bool = False,
        save_predictions: bool = False,
        use_gpt_action_judge: bool = False,
        rate_limit_delay: float = 0.5,
        action_library_path: Optional[str] = None
    ):
        # Initialize Gemini planner (reads GOOGLE_API_KEY / GEMINI_API_KEY from env)
        self.planner = GeminiPlanner(action_library_path=action_library_path)

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

                if hidream_config:
                    config = OmegaConf.load(hidream_config)
                else:
                    config = OmegaConf.create({})

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
            self.gpt_action_judge = GPT4oActionJudge()

        # Initialize GPT-4o Image Judge (if enabled)
        self.use_gpt_image_judge = use_gpt_action_judge
        self.gpt_image_judge = None
        if self.use_gpt_image_judge:
            try:
                self.gpt_image_judge = GPT4oJudge()
                logger.info("✅ GPT-4o Image Judge initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GPT-4o Image Judge: {e}")
                self.use_gpt_image_judge = False

        # Initialize reward evaluator (lazy loading to save memory)
        self.reward_evaluator = None

    def calculate_planner_metrics(self, predicted_plan: Dict, ground_truth_plan: Dict) -> Dict[str, float]:
        """Calculate planner metrics comparing predicted vs ground truth plans."""
        metrics = {}

        pred_actions = set(a["action_id"] for a in predicted_plan.get("actions", []))
        gt_actions = set(a["action_id"] for a in ground_truth_plan.get("actions", []))

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
            metrics["recall"] = 1.0
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

        union = len(pred_actions | gt_actions)
        metrics["iou"] = len(pred_actions & gt_actions) / union if union > 0 else 0.0

        metrics["num_predicted_actions"] = len(pred_actions)
        metrics["num_ground_truth_actions"] = len(gt_actions)

        return metrics

    def _pil_to_tensor(self, image: Image.Image):
        """Convert PIL Image to tensor."""
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        return transform(image)

    def _calculate_image_metrics(
        self,
        predicted: Image.Image,
        ground_truth: Image.Image,
        original_image: Image.Image = None
    ) -> Dict[str, float]:
        """Calculate image quality metrics."""
        import torch

        if predicted.size != ground_truth.size:
            logger.info(f"    ⚠️ Resizing predicted image from {predicted.size} to {ground_truth.size}")
            predicted = predicted.resize(ground_truth.size, Image.Resampling.LANCZOS)

        pred_tensor = self._pil_to_tensor(predicted).unsqueeze(0)
        gt_tensor = self._pil_to_tensor(ground_truth).unsqueeze(0)

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
        size = (384, 384)

        images = [
            ("Original", original),
            (predicted_label, predicted),
            ("Ground Truth", ground_truth)
        ]

        resized_images = [(label, img.resize(size, Image.Resampling.LANCZOS)) for label, img in images]
        num_images = len(resized_images)

        top_margin = 40
        bottom_margin = 80 if overall_instruction else 10

        comparison = Image.new('RGB', (size[0] * num_images, size[1] + top_margin + bottom_margin), color='white')

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        for idx, (label, img) in enumerate(resized_images):
            x_pos = idx * size[0]
            comparison.paste(img, (x_pos, top_margin))
            text_width = len(label) * 10
            text_x = x_pos + (size[0] - text_width) // 2
            draw.text((text_x, 5), label, fill="black", font=font_large)

        if overall_instruction:
            instruction_y = size[1] + top_margin + 10
            max_chars = 120
            if len(overall_instruction) > max_chars:
                overall_instruction = overall_instruction[:max_chars-3] + "..."
            draw.text((10, instruction_y), f"Instruction: {overall_instruction}", fill="black", font=font_small)

        comparison.save(output_path)

    def evaluate_sample(self, sample: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Evaluate a single sample with Gemini 2.5 planner."""
        start_time = time.time()

        image_path = Path(project_root) / sample["original_image_path"]
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        user_instruction = sample.get("overall_instruction", sample.get("user_prompt", ""))
        sample_id = sample["sample_id"]

        # Step 1: Predict action plan with Gemini 2.5
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
                if "hidream_prompt" in predicted_plan and predicted_plan["hidream_prompt"]:
                    edit_instruction = predicted_plan["hidream_prompt"]
                    logger.info(f"    Using model-generated hidream_prompt")
                else:
                    edit_instruction = predicted_plan.get("overall_instruction", user_instruction)
                    logger.warning(f"    hidream_prompt not found in predicted_plan, using overall_instruction fallback")

                original_image = Image.open(image_path).convert("RGB")
                if self.editor_type == "hidream":
                    edited_image = self.editor.edit_image(original_image, edit_instruction)
                else:  # qwen
                    output = self.editor(
                        prompt=edit_instruction,
                        image=original_image,
                        height=768,
                        width=768,
                        num_inference_steps=28,
                        guidance_scale=7.5,
                    )
                    edited_image = output.images[0]

                gt_image_path = Path(project_root) / sample["edited_image_path"]
                if gt_image_path.exists():
                    gt_image = Image.open(gt_image_path).convert("RGB")
                    image_metrics = self._calculate_image_metrics(
                        edited_image, gt_image, original_image
                    )
                    result["image_metrics"] = image_metrics

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)

                    edited_image.save(output_dir / "predicted_edit.png")
                    shutil.copy2(image_path, output_dir / "original.png")
                    if gt_image_path.exists():
                        shutil.copy2(gt_image_path, output_dir / "ground_truth.png")

                    original_image = Image.open(image_path).convert("RGB")
                    gt_image = Image.open(gt_image_path).convert("RGB") if gt_image_path.exists() else edited_image
                    self._save_comparison(
                        original=original_image,
                        predicted=edited_image,
                        ground_truth=gt_image,
                        output_path=output_dir / "comparison.png",
                        overall_instruction=user_instruction,
                        predicted_label="Gemini 2.5 Predicted"
                    )

                logger.info(f"  ✅ Image generation complete")

                if self.use_gpt_image_judge and self.gpt_image_judge is not None:
                    logger.info(f"  Running GPT-4o Image Judge for {sample_id}...")
                    try:
                        gpt_image_scores = self.gpt_image_judge.judge_single_edit(
                            original_image=original_image,
                            generated_image=edited_image,
                            instruction=edit_instruction
                        )

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

                logger.info(f"  Computing Qwen3-VL-8B reward scores...")
                try:
                    if self.reward_evaluator is None:
                        logger.info(f"  🏆 Initializing Qwen3-VL-8B Reward Evaluator (one-time setup)...")
                        self.reward_evaluator = RewardModelEvaluator()
                        logger.info(f"  ✅ Reward evaluator ready")

                    analysis = None
                    analysis_path = Path(project_root) / sample.get("analysis_path", "")
                    if analysis_path.exists():
                        with open(analysis_path, 'r') as f:
                            analysis = json.load(f)

                    reward_result = self.reward_evaluator.evaluate_transformation(
                        original_image_path=str(image_path),
                        edited_image_path=str(output_dir / "predicted_edit.png"),
                        user_prompt=user_instruction,
                        action_plan=predicted_plan,
                        analysis=analysis
                    )

                    if output_dir:
                        with open(output_dir / "reward_scores.json", 'w') as f:
                            json.dump(reward_result, f, indent=2)

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

            with open(output_dir / "predicted_plan.json", 'w') as f:
                json.dump(predicted_plan, f, indent=2)

            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(result, f, indent=2)

        result["evaluation_time"] = time.time() - start_time

        return result

    def evaluate_dataset(
        self,
        samples: List[Dict],
        output_dir: Path,
        split: str = "all"
    ) -> Tuple[Dict, List[Dict]]:
        """Evaluate multiple samples."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

            result = self.evaluate_sample(sample, sample_output_dir)

            if "error" in result:
                logger.warning(f"⚠️  Sample {sample_id} failed: {result['error']}")
                continue

            detailed_results.append(result)

            if "planner_metrics" in result:
                all_planner_metrics.append(result["planner_metrics"])

            if "image_metrics" in result:
                all_image_metrics.append(result["image_metrics"])

            if "gpt_action_scores" in result and "error" not in result["gpt_action_scores"]:
                all_gpt_action_scores.append(result["gpt_action_scores"])

            if i < len(samples) - 1:
                time.sleep(self.rate_limit_delay)

        summary = {
            "num_samples": len(detailed_results),
            "split": split,
            "model": "Gemini 2.5 Flash (API)",
            "planner_metrics": self._aggregate_metrics(all_planner_metrics),
        }

        if all_image_metrics:
            summary["image_metrics"] = self._aggregate_metrics(all_image_metrics)

        if all_gpt_action_scores:
            summary["gpt_action_scores"] = self._aggregate_gpt_scores(all_gpt_action_scores)

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
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

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
    with open(sample_ids_file, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data.get("samples", [])
    id_set = set(sample_ids)
    filtered_samples = [s for s in samples if s["sample_id"] in id_set]

    logger.info(f"Loaded {len(filtered_samples)} samples (out of {len(sample_ids)} requested IDs)")

    return filtered_samples


def main():
    parser = argparse.ArgumentParser(description="Gemini 2.5 Planner Evaluation")

    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--sample-ids-file", type=str, default=None,
                        help="File with sample IDs to evaluate (one per line)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"],
                        help="Which split to evaluate")
    parser.add_argument("--model-editor", type=str, default="qwen", choices=["qwen", "hidream"],
                        help="Image editor to use")
    parser.add_argument("--hidream-checkpoint", type=str, default=None,
                        help="Path to HiDream checkpoint (if using hidream)")
    parser.add_argument("--hidream-config", type=str, default=None,
                        help="Path to HiDream config (if using hidream)")
    parser.add_argument("--action-library", type=str, default=None,
                        help="Path to action library JSON")
    parser.add_argument("--save-images", action="store_true",
                        help="Generate and save edited images")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save predicted action plans")
    parser.add_argument("--no-gpt-judge", action="store_true",
                        help="Disable GPT-4o action judge (disabled by default)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")

    args = parser.parse_args()

    # Load samples
    if args.sample_ids_file:
        samples = load_samples_from_ids(args.data, args.sample_ids_file)
    else:
        with open(args.data, 'r') as f:
            data = json.load(f)

        samples = data.get("samples", [])

        if args.split != "all":
            samples = [s for s in samples if s.get("split", "val") == args.split]

        logger.info(f"Loaded {len(samples)} samples from {args.split} split")

    if not samples:
        logger.error("No samples to evaluate!")
        return

    # Initialize evaluator
    logger.info("Initializing Gemini 2.5 evaluator...")
    evaluator = GeminiPlannerEvaluator(
        image_editor_type=args.model_editor,
        hidream_checkpoint=args.hidream_checkpoint,
        hidream_config=args.hidream_config,
        save_images=args.save_images,
        save_predictions=args.save_predictions,
        use_gpt_action_judge=not args.no_gpt_judge if hasattr(args, 'no_gpt_judge') else False,
        rate_limit_delay=args.rate_limit_delay,
        action_library_path=args.action_library
    )

    logger.info(f"\nStarting evaluation...")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Save images: {args.save_images}")
    logger.info(f"  Save predictions: {args.save_predictions}")
    logger.info(f"  Rate limit: {args.rate_limit_delay}s")
    logger.info("")

    summary, detailed = evaluator.evaluate_dataset(
        samples=samples,
        output_dir=Path(args.output),
        split=args.split
    )

    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: Gemini 2.5 Flash (API)")
    logger.info(f"Samples evaluated: {summary['num_samples']}")
    logger.info("")

    if "planner_metrics" in summary:
        logger.info("Planner Metrics:")
        pm = summary["planner_metrics"]
        logger.info(f"  F1 Score:  {pm.get('f1_mean', 0):.3f} ± {pm.get('f1_std', 0):.3f}")
        logger.info(f"  Precision: {pm.get('precision_mean', 0):.3f} ± {pm.get('precision_std', 0):.3f}")
        logger.info(f"  Recall:    {pm.get('recall_mean', 0):.3f} ± {pm.get('recall_std', 0):.3f}")
        logger.info(f"  IoU:       {pm.get('iou_mean', 0):.3f} ± {pm.get('iou_std', 0):.3f}")
        logger.info("")

    logger.info("="*80)
    logger.info(f"✅ Results saved to: {args.output}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
