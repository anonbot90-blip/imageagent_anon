#!/usr/bin/env python3
"""
Unified Model Evaluator

This script provides a single unified interface for evaluating all model types:
- Baseline: Qwen3-VL planner (no RL)
- Standard/RL/RW/DPO/SW: Fine-tuned planners
- Edit-Only: Direct image editing (no planning)
- GPT-4o: API-based planner

Usage:
    python scripts/evaluation/evaluate_model.py \\
        --model-type baseline \\
        --checkpoint PATH \\
        --data DATASET.json \\
        --output OUTPUT_DIR \\
        --sample-ids-file IDS.txt \\
        --editor-type qwen \\
        --gpu 0 \\
        --save-images

Model Types:
    - baseline: Base Qwen3-VL planner (no fine-tuning)
    - standard: Standard fine-tuned planner
    - rl: RL fine-tuned planner  
    - rw: Reward-weighted fine-tuned planner
    - dpo: DPO fine-tuned planner
    - sw: Sample-weighted fine-tuned planner
    - edit_only: Direct editing baseline (no planning)
    - gpt4o: GPT-4o Vision API planner
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from actions.planner_inference import ActionPlanner
from actions.gpt4o_planner import GPT4oPlanner
from training.evaluation.metrics import MetricsCalculator
from training.evaluation.gpt_action_judge import GPT4oActionJudge
from training.evaluation.gpt_judge import GPT4oJudge
from src.reward_model_evaluator import RewardModelEvaluator
from diffusers import DiffusionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_base_model_from_checkpoint(checkpoint_path: str) -> str:
    """Auto-detect base model from checkpoint's adapter_config.json."""
    checkpoint_dir = Path(checkpoint_path)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    
    # Default to 8B for backward compatibility
    default_model = "Qwen/Qwen3-VL-8B-Instruct"
    
    if checkpoint_dir.exists() and checkpoint_dir.is_dir():
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path", default_model)
                logger.info(f"Detected base model: {base_model}")
                return base_model
            except Exception as e:
                logger.warning(f"Could not read adapter_config.json: {e}")
    
    # If HuggingFace model ID provided directly
    if "Qwen3-VL-4B" in checkpoint_path or "4b" in checkpoint_path.lower():
        return "Qwen/Qwen3-VL-4B-Instruct"
    elif "Qwen3-VL-8B" in checkpoint_path or "8b" in checkpoint_path.lower():
        return "Qwen/Qwen3-VL-8B-Instruct"
    
    logger.info(f"Using default base model: {default_model}")
    return default_model


class UnifiedModelEvaluator:
    """Unified evaluator for all model types."""
    
    def __init__(
        self,
        model_type: str,
        checkpoint: Optional[str] = None,
        data_path: str = None,
        output_dir: str = None,
        sample_ids_file: Optional[str] = None,
        results_dir: Optional[str] = None,  # For edit_only mode
        editor_type: str = "qwen",
        hidream_checkpoint: Optional[str] = None,
        hidream_config: Optional[str] = None,
        action_library: Optional[str] = None,
        device: str = "cuda",
        save_images: bool = False,
        save_predictions: bool = False,
        use_gpt_judge: bool = True
    ):
        """
        Initialize unified evaluator.
        
        Args:
            model_type: Model type (baseline, standard, rl, rw, dpo, sw, edit_only, gpt4o)
            checkpoint: Path to model checkpoint (for planner models)
            data_path: Path to evaluation dataset JSON
            output_dir: Output directory
            sample_ids_file: File with sample IDs to evaluate
            results_dir: Results directory (for edit_only mode)
            editor_type: Image editor type (qwen or hidream)
            hidream_checkpoint: HiDream checkpoint path
            hidream_config: HiDream config path
            action_library: Action library JSON path (for GPT-4o)
            device: Device to use
            save_images: Save comparison images
            save_predictions: Save predictions
            use_gpt_judge: Use GPT-4o judges
        """
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.data_path = Path(data_path) if data_path else None
        self.output_dir = Path(output_dir)
        self.sample_ids_file = sample_ids_file
        self.results_dir = Path(results_dir) if results_dir else None
        self.editor_type = editor_type
        self.device = device
        self.save_images = save_images
        self.save_predictions = save_predictions
        self.use_gpt_judge = use_gpt_judge
        self.action_library = action_library
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models based on type
        self.planner = None
        self.editor = None
        self.metrics_calculator = None
        self.gpt_action_judge = None
        self.gpt_image_judge = None
        self.reward_model = None
        
        logger.info(f"Initializing {model_type} evaluator...")
        self._setup_models(hidream_checkpoint, hidream_config)
        logger.info(f"✓ {model_type} evaluator initialized")
    
    def _setup_models(self, hidream_checkpoint: Optional[str], hidream_config: Optional[str]):
        """Setup models based on model type."""
        # Setup planner (except for edit_only)
        if self.model_type != "edit_only":
            if self.model_type == "gpt4o":
                self._setup_gpt4o_planner()
            else:
                self._setup_local_planner()
        
        # Setup image editor if needed
        if self.save_images:
            self._setup_image_editor(hidream_checkpoint, hidream_config)
        
        # Setup metrics calculator
        if self.save_images:
            logger.info("Initializing metrics calculator...")
            self.metrics_calculator = MetricsCalculator(device=self.device)
        
        # Setup GPT judges
        if self.use_gpt_judge:
            logger.info("Initializing GPT-4o judges...")
            try:
                self.gpt_action_judge = GPT4oActionJudge()
                if self.save_images:
                    self.gpt_image_judge = GPT4oJudge()
            except Exception as e:
                logger.warning(f"Could not initialize GPT judges: {e}")
        
        # Setup reward model (reuse planner to save memory - like old script)
        if self.save_images:
            logger.info("Initializing reward model (reusing planner model)...")
            try:
                # Reuse planner for reward evaluation (memory efficient)
                # This matches the old evaluate_planner_batch.py approach
                # RewardModelEvaluator can accept ActionPlanner as existing_model
                if self.planner:
                    self.reward_model = RewardModelEvaluator(existing_model=self.planner)
                    logger.info("✓ Reward model reusing planner (memory efficient - no separate 8B model)")
                else:
                    # Fallback: load separate model if planner not available
                    logger.warning("Planner not available for reuse, loading separate reward model")
                    self.reward_model = RewardModelEvaluator()
            except Exception as e:
                logger.warning(f"Could not initialize reward model: {e}")
                self.reward_model = None
    
    def _setup_local_planner(self):
        """Setup local planner model."""
        if self.checkpoint is None:
            raise ValueError(f"--checkpoint required for {self.model_type} model type")
        
        logger.info(f"Loading {self.model_type} planner from {self.checkpoint}")
        base_model = get_base_model_from_checkpoint(self.checkpoint)
        
        self.planner = ActionPlanner(
            model_name=base_model,
            lora_checkpoint=self.checkpoint,
            device=self.device
        )
        logger.info(f"✓ Planner loaded")
    
    def _setup_gpt4o_planner(self):
        """Setup GPT-4o API planner."""
        logger.info("Initializing GPT-4o planner...")
        self.planner = GPT4oPlanner(
            azure_config=None,  # Uses environment variables
            action_library_path=self.action_library
        )
        logger.info("✓ GPT-4o planner initialized")
    
    def _setup_image_editor(self, hidream_checkpoint: Optional[str], hidream_config: Optional[str]):
        """Setup image editor (Qwen or HiDream)."""
        if self.editor_type == "qwen":
            logger.info("Loading Qwen-Image-Edit model...")
            self.editor = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.editor.set_progress_bar_config(disable=True)
            logger.info("✓ Qwen editor loaded")
        
        elif self.editor_type == "hidream":
            if hidream_checkpoint is None:
                raise ValueError("--hidream-checkpoint required for hidream editor")
            logger.info(f"Loading HiDream editor from {hidream_checkpoint}")
            
            config = OmegaConf.load(hidream_config) if hidream_config else None
            self.editor = HiDreamE1LoRA(config=config, device=self.device)
            
            state_dict = safetensors.torch.load_file(hidream_checkpoint)
            self.editor.load_state_dict(state_dict, strict=False)
            self.editor.eval()
            logger.info("✓ HiDream editor loaded")
        
        else:
            raise ValueError(f"Unknown editor type: {self.editor_type}")
    
    def load_samples(self) -> List[Dict]:
        """Load samples to evaluate."""
        # Edit-only mode: load from results_dir
        if self.model_type == "edit_only" and self.results_dir:
            with open(self.sample_ids_file) as f:
                sample_ids = [line.strip() for line in f if line.strip()]
            
            # Create minimal sample dictionaries for edit-only
            samples = []
            for sid in sample_ids:
                sample_dir = self.results_dir / sid
                if sample_dir.exists():
                    samples.append({
                        "sample_id": sid,
                        "results_path": str(sample_dir)
                    })
            
            logger.info(f"Loaded {len(samples)} samples from results_dir")
            return samples
        
        # Standard mode: load from dataset JSON
        if not self.data_path:
            raise ValueError("--data required for non-edit_only models")
        
        with open(self.data_path) as f:
            data = json.load(f)
        
        all_samples = {s["sample_id"]: s for s in data["samples"]}
        
        # Filter by sample IDs if provided
        if self.sample_ids_file:
            with open(self.sample_ids_file) as f:
                sample_ids = [line.strip() for line in f if line.strip()]
            
            samples = [all_samples[sid] for sid in sample_ids if sid in all_samples]
            logger.info(f"Loaded {len(samples)} samples from {self.sample_ids_file}")
        else:
            samples = list(all_samples.values())
            logger.info(f"Loaded all {len(samples)} samples")
        
        return samples
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample based on model type."""
        if self.model_type == "edit_only":
            return self._evaluate_edit_only(sample)
        elif self.model_type == "gpt4o":
            return self._evaluate_gpt4o(sample)
        else:
            return self._evaluate_planner(sample)
    
    def _evaluate_planner(self, sample: Dict) -> Dict:
        """Evaluate sample using local planner model."""
        sample_id = sample["sample_id"]
        sample_dir = self.output_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Get ground truth
        gt_action_plan = sample["target_action_plan"]
        original_image_path = project_root / sample["original_image_path"]
        edited_image_path = project_root / sample["edited_image_path"]
        
        # Predict action plan
        overall_instruction = sample.get("overall_instruction") or sample.get("user_prompt", "")
        
        # Load analysis if available
        analysis = None
        if "analysis_path" in sample:
            analysis_path = project_root / sample["analysis_path"]
            if analysis_path.exists():
                with open(analysis_path) as f:
                    analysis = json.load(f)
        
        with torch.no_grad():
            predicted_plan = self.planner.predict_action_plan(
                image_path=str(original_image_path),
                user_prompt=overall_instruction,
                analysis=analysis
            )
        
        # Calculate planner metrics
        planner_metrics = self._calculate_planner_metrics(predicted_plan, gt_action_plan)
        
        result = {
            "sample_id": sample_id,
            "planner_metrics": planner_metrics,
            "predicted_plan": predicted_plan,
            "ground_truth_plan": gt_action_plan
        }
        
        # GPT action judge
        if self.gpt_action_judge:
            try:
                original_img = Image.open(original_image_path).convert("RGB")
                gpt_action_result = self.gpt_action_judge.judge_action_plan(
                    original_image=original_img,
                    user_prompt=overall_instruction,
                    predicted_plan=predicted_plan,
                    teacher_plan=gt_action_plan
                )
                result["gpt_action_judge"] = gpt_action_result
            except Exception as e:
                logger.warning(f"GPT action judge failed for {sample_id}: {e}")
        
        # Save predicted plan (match old script structure)
        with open(sample_dir / "predicted_plan.json", 'w') as f:
            json.dump(predicted_plan, f, indent=2)
        
        # Save original and ground truth images (match old script structure)
        original_img = Image.open(original_image_path).convert("RGB")
        original_img.save(sample_dir / "original.png")
        
        if edited_image_path.exists():
            target_img = Image.open(edited_image_path).convert("RGB")
            target_img.save(sample_dir / "ground_truth.png")
        else:
            target_img = None
        
        # Generate edited image if requested
        if self.save_images and self.editor:
            try:
                # Clear CUDA cache BEFORE image generation to free memory
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                generated_image = self._generate_edited_image(
                    original_image_path,
                    predicted_plan,
                    sample_dir
                )
                
                # Clear CUDA cache after image generation to free memory
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # Calculate image metrics
                if target_img is None:
                    target_img = Image.open(edited_image_path).convert("RGB") if edited_image_path.exists() else None
                
                if target_img:
                    image_metrics = self._calculate_image_metrics(
                        generated_image,
                        target_img,
                        original_img,
                        overall_instruction
                    )
                    result["image_metrics"] = image_metrics
                
                # GPT image judge
                if self.gpt_image_judge:
                    try:
                        gpt_image_result = self.gpt_image_judge.judge_single_edit(
                            original_image=original_img,
                            generated_image=generated_image,
                            instruction=overall_instruction
                        )
                        result["gpt_image_judge"] = gpt_image_result
                    except Exception as e:
                        logger.warning(f"GPT image judge failed for {sample_id}: {e}")
                
                # Reward model
                if self.reward_model:
                    try:
                        reward_score = self.reward_model.evaluate(
                            original_image=original_img,
                            edited_image=generated_image,
                            instruction=overall_instruction
                        )
                        result["reward_score"] = reward_score
                        
                        # Save reward scores (match old script structure)
                        with open(sample_dir / "reward_scores.json", 'w') as f:
                            json.dump(reward_score, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Reward model failed for {sample_id}: {e}")
                
                # Save comparison image
                if target_img:
                    self._save_comparison_image(
                        original_img,
                        target_img,
                        generated_image,
                        sample_dir
                    )
                
            except Exception as e:
                logger.warning(f"Image generation failed for {sample_id}: {e}")
        
        # Save per-sample results (match old script naming)
        with open(sample_dir / "metrics.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _evaluate_edit_only(self, sample: Dict) -> Dict:
        """Evaluate sample using direct image editing (no planning)."""
        sample_id = sample["sample_id"]
        sample_output_dir = self.output_dir / "samples" / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load from results_dir if available (original mode)
        if "results_path" in sample:
            results_sample_dir = Path(sample["results_path"])
            
            # Load original image
            original_image_path = results_sample_dir / "original.png"
            if not original_image_path.exists():
                logger.warning(f"Skipping {sample_id}: No original image")
                return {"sample_id": sample_id, "error": "No original image"}
            
            # Load edit instruction from prompt.json
            prompt_path = results_sample_dir / "prompt.json"
            if not prompt_path.exists():
                logger.warning(f"Skipping {sample_id}: No prompt file")
                return {"sample_id": sample_id, "error": "No prompt file"}
            
            with open(prompt_path) as f:
                prompt_data = json.load(f)
            
            # Extract edit instruction
            if 'edit_info' in prompt_data and 'text' in prompt_data['edit_info']:
                overall_instruction = prompt_data['edit_info']['text']
            elif 'edit' in prompt_data and 'text' in prompt_data['edit']:
                overall_instruction = prompt_data['edit']['text']
            else:
                logger.warning(f"Skipping {sample_id}: No edit instruction")
                return {"sample_id": sample_id, "error": "No edit instruction"}
            
            edited_image_path = results_sample_dir / "edited.png"
        else:
            # Load from dataset JSON (new mode)
            original_image_path = project_root / sample["original_image_path"]
            edited_image_path = project_root / sample["edited_image_path"]
            overall_instruction = sample.get("overall_instruction") or sample.get("user_prompt", "")
        
        # Load images
        original_img = Image.open(original_image_path).convert("RGB")
        target_img = Image.open(edited_image_path).convert("RGB") if edited_image_path.exists() else None
        
        # Direct image editing
        pil_image = original_img.resize((768, 768), Image.Resampling.LANCZOS)
        formatted_prompt = f"Editing Instruction: {overall_instruction}. Maintain high quality, original composition and style."
        
        try:
            with torch.no_grad():
                if self.editor_type == "qwen":
                    output = self.editor(
                        prompt=formatted_prompt,
                        image=pil_image,
                        height=768,
                        width=768,
                        num_inference_steps=28,
                        guidance_scale=7.5,
                    )
                    generated_image = output.images[0]
                else:
                    # HiDream
                    generated_image = self.editor.edit_image(
                        image=pil_image,
                        instruction=formatted_prompt
                    )
            
            # Save generated image
            generated_image.save(sample_output_dir / "predicted_edit.png")
            original_img.save(sample_output_dir / "original.png")
            
            # Clear CUDA cache immediately after generation
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            # Clear cache and re-raise
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            raise
        
        # Save instruction
        with open(sample_output_dir / "edit_prompt.txt", 'w') as f:
            f.write(overall_instruction)
        
        # Calculate metrics
        result = {
            "sample_id": sample_id,
            "planner_metrics": None,  # No planning
            "gpt_action_judge": None  # No planning
        }
        
        if target_img and self.metrics_calculator:
            target_img.save(sample_output_dir / "ground_truth.png")
            
            image_metrics = self._calculate_image_metrics(
                generated_image,
                target_img,
                original_img,
                overall_instruction
            )
            result["image_metrics"] = image_metrics
        
        # GPT image judge
        if self.gpt_image_judge:
            try:
                gpt_image_result = self.gpt_image_judge.judge_single_edit(
                    original_image=original_img,
                    generated_image=generated_image,
                    instruction=overall_instruction
                )
                result["gpt_image_judge"] = gpt_image_result
            except Exception as e:
                logger.warning(f"GPT image judge failed for {sample_id}: {e}")
        
        # Reward model
        if self.reward_model:
            try:
                reward_score = self.reward_model.evaluate(
                    original_image=original_img,
                    edited_image=generated_image,
                    instruction=overall_instruction
                )
                result["reward_score"] = reward_score
                
                # Save reward scores
                with open(sample_output_dir / "reward_scores.json", 'w') as f:
                    json.dump({"reward_score": reward_score}, f, indent=2)
            except Exception as e:
                logger.warning(f"Reward model failed for {sample_id}: {e}")
        
        # Save comparison if we have target
        if target_img:
            self._save_comparison_image(original_img, target_img, generated_image, sample_output_dir)
        
        # Save result
        with open(sample_output_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _evaluate_gpt4o(self, sample: Dict) -> Dict:
        """Evaluate sample using GPT-4o planner."""
        # Similar to _evaluate_planner but with rate limiting
        time.sleep(0.5)  # Rate limit
        return self._evaluate_planner(sample)
    
    def _generate_edited_image(self, original_image_path: Path, predicted_plan: Dict, sample_dir: Path) -> Image.Image:
        """Generate edited image from predicted plan."""
        # Determine instruction (match old script behavior)
        # PREFER: Model-generated hidream_prompt (if available)
        # FALLBACK: overall_instruction or style_transformation_mode
        if "hidream_prompt" in predicted_plan and predicted_plan["hidream_prompt"]:
            instruction = predicted_plan["hidream_prompt"]
            logger.info(f"    Using model-generated hidream_prompt")
        else:
            instruction = predicted_plan.get("overall_instruction", "style_transformation_mode")
            logger.warning(f"    hidream_prompt not found in predicted_plan, using overall_instruction fallback")
        
        # Format prompt for Qwen editor (match old script)
        formatted_prompt = f"Editing Instruction: {instruction}. Maintain high quality, original composition and style."
        
        # Save instruction files (match old script structure)
        with open(sample_dir / "instruction.txt", 'w') as f:
            f.write(instruction)
        with open(sample_dir / "edit_prompt.txt", 'w') as f:
            f.write(formatted_prompt)
        
        # Load and edit image
        original_img = Image.open(original_image_path).convert("RGB")
        pil_image = original_img.resize((768, 768), Image.Resampling.LANCZOS)
        
        try:
            with torch.no_grad():
                if self.editor_type == "qwen":
                    output = self.editor(
                        prompt=formatted_prompt,
                        image=pil_image,
                        height=768,
                        width=768,
                        num_inference_steps=28,
                        guidance_scale=7.5,
                    )
                    generated_image = output.images[0]
                else:
                    generated_image = self.editor.edit_image(
                        image=pil_image,
                        instruction=formatted_prompt
                    )
            
            # Save generated image (match old script naming)
            generated_image.save(sample_dir / "predicted_edit.png")
            
            # Clear CUDA cache immediately after generation
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            return generated_image
        except torch.cuda.OutOfMemoryError as e:
            # Clear cache and re-raise
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            raise
    
    def _plan_to_instruction(self, plan: Dict) -> str:
        """Convert action plan to natural language instruction."""
        actions = plan.get("action_plan", [])
        if not actions:
            return "Edit the image."
        
        instructions = []
        for action in actions:
            action_type = action.get("action")
            params = action.get("parameters", {})
            
            if action_type == "add_object":
                obj = params.get("object")
                loc = params.get("location")
                instructions.append(f"Add {obj} at {loc}")
            elif action_type == "remove_object":
                obj = params.get("object")
                instructions.append(f"Remove {obj}")
            elif action_type == "change_color":
                obj = params.get("object")
                color = params.get("color")
                instructions.append(f"Change {obj} to {color}")
            elif action_type == "modify_attribute":
                obj = params.get("object")
                attr = params.get("attribute")
                val = params.get("value")
                instructions.append(f"Modify {obj}'s {attr} to {val}")
            else:
                # Generic fallback
                instructions.append(f"{action_type}: {params}")
        
        return ". ".join(instructions) + "."
    
    def _calculate_planner_metrics(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """Calculate planner evaluation metrics (F1, precision, recall, IoU)."""
        metrics = {}
        
        # Extract action plans - check both "actions" and "action_plan" keys
        pred_actions = predicted.get("actions", predicted.get("action_plan", []))
        gt_actions = ground_truth.get("actions", ground_truth.get("action_plan", []))
        
        if not gt_actions:
            return metrics
        
        # 1. Action Type Matching - check both "action" and "action_id" keys
        pred_action_types = set()
        for a in pred_actions:
            action_name = a.get("action", a.get("action_id", "")).lower()
            if action_name:
                pred_action_types.add(action_name)
        
        gt_action_types = set()
        for a in gt_actions:
            action_name = a.get("action", a.get("action_id", "")).lower()
            if action_name:
                gt_action_types.add(action_name)
        
        if gt_action_types:
            intersection = pred_action_types & gt_action_types
            union = pred_action_types | gt_action_types
            
            metrics["action_precision"] = len(intersection) / len(pred_action_types) if pred_action_types else 0.0
            metrics["action_recall"] = len(intersection) / len(gt_action_types)
            metrics["action_f1"] = (2 * metrics["action_precision"] * metrics["action_recall"] / 
                                   (metrics["action_precision"] + metrics["action_recall"])) \
                                    if (metrics["action_precision"] + metrics["action_recall"]) > 0 else 0.0
            metrics["action_iou"] = len(intersection) / len(union) if union else 0.0
        else:
            metrics["action_precision"] = 0.0
            metrics["action_recall"] = 0.0
            metrics["action_f1"] = 0.0
            metrics["action_iou"] = 0.0
        
        # 2. Object Matching (if available)
        try:
            pred_objects = set()
            for action in pred_actions:
                params = action.get("parameters", {})
                if "object" in params:
                    pred_objects.add(params["object"].lower())
                if "target" in params:
                    pred_objects.add(params["target"].lower())
            
            gt_objects = set()
            for action in gt_actions:
                params = action.get("parameters", {})
                if "object" in params:
                    gt_objects.add(params["object"].lower())
                if "target" in params:
                    gt_objects.add(params["target"].lower())
            
            if gt_objects:
                obj_intersection = pred_objects & gt_objects
                obj_union = pred_objects | gt_objects
                metrics["object_precision"] = len(obj_intersection) / len(pred_objects) if pred_objects else 0.0
                metrics["object_recall"] = len(obj_intersection) / len(gt_objects)
                metrics["object_f1"] = (2 * metrics["object_precision"] * metrics["object_recall"] / 
                                       (metrics["object_precision"] + metrics["object_recall"])) \
                                        if (metrics["object_precision"] + metrics["object_recall"]) > 0 else 0.0
        except:
            pass
        
        # 3. Priority Correlation (Spearman)
        try:
            from scipy.stats import spearmanr
            
            pred_priorities = {a.get("action", a.get("action_id", "")).lower(): a.get("priority", 1) 
                              for a in pred_actions if a.get("action") or a.get("action_id")}
            gt_priorities = {a.get("action", a.get("action_id", "")).lower(): a.get("priority", 1) 
                            for a in gt_actions if a.get("action") or a.get("action_id")}
            
            common_actions = set(pred_priorities.keys()) & set(gt_priorities.keys())
            if len(common_actions) >= 2:
                pred_ranks = [pred_priorities[a] for a in sorted(common_actions)]
                gt_ranks = [gt_priorities[a] for a in sorted(common_actions)]
                correlation, _ = spearmanr(pred_ranks, gt_ranks)
                if not np.isnan(correlation):
                    metrics["priority_correlation"] = float(correlation)
        except:
            pass
        
        # 4. JSON Validity and Action Counts
        metrics["valid_json"] = 1.0 if isinstance(predicted, dict) else 0.0
        metrics["has_actions"] = 1.0 if pred_actions else 0.0
        metrics["num_predicted_actions"] = len(pred_actions)
        metrics["num_ground_truth_actions"] = len(gt_actions)
        
        return metrics
    
    def _calculate_image_metrics(self, generated: Image.Image, target: Image.Image, original: Image.Image, instruction: str = "") -> Dict:
        """Calculate image quality metrics."""
        if not self.metrics_calculator:
            return {}
        
        metrics = {}
        
        # Convert PIL Images to tensors
        pred_tensor = torch.from_numpy(np.array(generated)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gt_tensor = torch.from_numpy(np.array(target)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pred_tensor = pred_tensor.to(self.device)
        gt_tensor = gt_tensor.to(self.device)
        
        # LPIPS
        try:
            lpips_score = self.metrics_calculator.compute_lpips(pred_tensor, gt_tensor)
            metrics["lpips"] = float(lpips_score)
        except Exception as e:
            logger.warning(f"LPIPS calculation failed: {e}")
        
        # SSIM
        try:
            ssim_score = self.metrics_calculator.compute_ssim(pred_tensor, gt_tensor)
            metrics["ssim"] = float(ssim_score)
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
        
        # PSNR
        try:
            psnr_score = self.metrics_calculator.compute_psnr(pred_tensor, gt_tensor)
            metrics["psnr"] = float(psnr_score)
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")
        
        # CLIP score (text-image alignment)
        if instruction:
            try:
                clip_score = self.metrics_calculator.compute_clip_score([generated], [instruction])
                metrics["clip_score"] = float(clip_score)
            except Exception as e:
                logger.warning(f"CLIP calculation failed: {e}")
        
        return metrics
    
    def _save_comparison_image(self, original: Image.Image, target: Image.Image, generated: Image.Image, sample_dir: Path):
        """Save comparison visualization."""
        # Create side-by-side comparison
        width, height = original.size
        comparison = Image.new("RGB", (width * 3, height))
        comparison.paste(original, (0, 0))
        comparison.paste(target, (width, 0))
        comparison.paste(generated, (width * 2, 0))
        
        comparison.save(sample_dir / "comparison.png")
    
    def run(self):
        """Run evaluation on all samples."""
        logger.info(f"Starting {self.model_type} evaluation...")
        logger.info(f"Output: {self.output_dir}")
        
        # Load samples
        samples = self.load_samples()
        
        # Load existing results for batch merging
        existing_results = self._load_existing_results()
        
        # Evaluate samples
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {self.model_type}"):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
                
                # Clear CUDA cache after each sample to prevent OOM
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            except Exception as e:
                logger.error(f"Failed to evaluate {sample['sample_id']}: {e}")
                import traceback
                traceback.print_exc()
                
                # Clear cache even on error
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
        
        # Merge with existing results
        all_results = self._merge_results(existing_results, results)
        
        # Calculate aggregate metrics
        summary = self._calculate_summary(all_results)
        
        # Save results
        self._save_results(all_results, summary)
        
        logger.info(f"✓ Evaluation complete: {len(all_results)} samples")
        logger.info(f"Results saved to: {self.output_dir}")
    
    def _load_existing_results(self) -> List[Dict]:
        """Load existing evaluation results for batch merging."""
        results_file = self.output_dir / "evaluation_results_all.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                return data.get("results", [])
        return []
    
    def _merge_results(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge new results with existing results (deduplicate by sample_id)."""
        # Create lookup by sample_id
        merged = {r["sample_id"]: r for r in existing}
        
        # Update with new results
        for result in new:
            merged[result["sample_id"]] = result
        
        return list(merged.values())
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all samples."""
        if not results:
            return {}
        
        summary = {
            "num_samples": len(results),
            "model_type": self.model_type
        }
        
        # Planner metrics
        planner_metrics_list = [r["planner_metrics"] for r in results if r.get("planner_metrics")]
        if planner_metrics_list:
            summary["planner_metrics"] = self._aggregate_metrics(planner_metrics_list)
        
        # Image metrics
        image_metrics_list = [r["image_metrics"] for r in results if r.get("image_metrics")]
        if image_metrics_list:
            summary["image_metrics"] = self._aggregate_metrics(image_metrics_list)
        
        # GPT action judge
        gpt_action_list = [r["gpt_action_judge"] for r in results if r.get("gpt_action_judge")]
        if gpt_action_list:
            summary["gpt_action_judge"] = self._aggregate_gpt_action(gpt_action_list)
        
        # GPT image judge
        gpt_image_list = [r["gpt_image_judge"] for r in results if r.get("gpt_image_judge")]
        if gpt_image_list:
            summary["gpt_image_judge"] = self._aggregate_gpt_image(gpt_image_list)
        
        # Reward scores
        reward_scores = [r["reward_score"] for r in results if r.get("reward_score")]
        if reward_scores:
            summary["reward_score"] = {
                "mean": float(np.mean(reward_scores)),
                "std": float(np.std(reward_scores)),
                "min": float(np.min(reward_scores)),
                "max": float(np.max(reward_scores))
            }
        
        return summary
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate numeric metrics."""
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        # Calculate mean/std for each metric
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return aggregated
    
    def _aggregate_gpt_action(self, gpt_list: List[Dict]) -> Dict:
        """Aggregate GPT action judge results."""
        if not gpt_list:
            return {}
        
        # Define all GPT action metrics
        metric_keys = [
            "relevance",
            "theme_style_focus",
            "completeness",
            "efficiency",
            "correctness",
            "reasoning_conciseness",
            "reasoning_completeness",
            "reasoning_specificity",
            "overall_action_quality",
            "overall_reasoning_quality",
            "overall_score"
        ]
        
        aggregated = {}
        for key in metric_keys:
            values = [g.get(key) for g in gpt_list if g.get(key) is not None]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
        
        aggregated["num_samples"] = len(gpt_list)
        return aggregated
    
    def _aggregate_gpt_image(self, gpt_list: List[Dict]) -> Dict:
        """Aggregate GPT image judge results."""
        if not gpt_list:
            return {}
        
        # Define all GPT image metrics
        metric_keys = [
            "instruction_following",
            "visual_quality",
            "transformation_strength",
            "coherence",
            "semantic_accuracy",
            "technical_execution",
            "overall_image_score"
        ]
        
        aggregated = {}
        for key in metric_keys:
            values = [g.get(key) for g in gpt_list if g.get(key) is not None]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
        
        aggregated["num_samples"] = len(gpt_list)
        return aggregated
    
    def _save_results(self, results: List[Dict], summary: Dict):
        """Save evaluation results and summary."""
        # Save detailed results
        results_file = self.output_dir / "evaluation_results_all.json"
        with open(results_file, 'w') as f:
            json.dump({
                "model_type": self.model_type,
                "num_samples": len(results),
                "results": results
            }, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "evaluation_summary_all.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Saved results to {results_file}")
        logger.info(f"✓ Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Unified Model Evaluator")
    
    # Model configuration
    parser.add_argument("--model-type", required=True,
                       choices=["baseline", "standard", "rl", "rw", "dpo", "sw", "edit_only", "gpt4o"],
                       help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path (for planner models)")
    
    # Data configuration
    parser.add_argument("--data", type=str, help="Path to evaluation dataset JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sample-ids-file", type=str, help="File with sample IDs to evaluate")
    parser.add_argument("--results-dir", type=str, help="Results directory (for edit_only mode)")
    
    # Image editor configuration
    parser.add_argument("--editor-type", default="qwen", choices=["qwen", "hidream"],
                       help="Image editor type")
    parser.add_argument("--hidream-checkpoint", type=str, help="HiDream checkpoint path")
    parser.add_argument("--hidream-config", type=str, help="HiDream config path")
    
    # GPT-4o configuration
    parser.add_argument("--action-library", type=str, help="Action library JSON (for GPT-4o)")
    
    # Device configuration
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    
    # Evaluation options
    parser.add_argument("--save-images", action="store_true", help="Save generated images")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions")
    parser.add_argument("--no-gpt-judge", action="store_true", help="Disable GPT judges")
    
    args = parser.parse_args()
    
    # Set device
    device = f"cuda:{args.gpu}" if args.gpu != "none" else "cuda"
    
    # Create evaluator
    evaluator = UnifiedModelEvaluator(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        data_path=args.data,
        output_dir=args.output,
        sample_ids_file=args.sample_ids_file,
        results_dir=args.results_dir,
        editor_type=args.editor_type,
        hidream_checkpoint=args.hidream_checkpoint,
        hidream_config=args.hidream_config,
        action_library=args.action_library,
        device=device,
        save_images=args.save_images,
        save_predictions=args.save_predictions,
        use_gpt_judge=not args.no_gpt_judge
    )
    
    # Run evaluation
    evaluator.run()


if __name__ == "__main__":
    main()

