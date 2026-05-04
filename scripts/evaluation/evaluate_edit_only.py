#!/usr/bin/env python3
"""
Edit-Only (E) Baseline Evaluation

This script evaluates a simple baseline that bypasses action planning:
- Takes edit instruction + original image
- Directly edits image using Qwen Edit model
- No planning step involved

Purpose: Validate the necessity of the action planning component.
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
from pathlib import Path
from typing import Dict, List, Any
import torch
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusers import DiffusionPipeline
from training.evaluation.metrics import MetricsCalculator
from training.evaluation.gpt_judge import GPT4oJudge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_edit_model(device: str = "cuda"):
    """Load Qwen-Image-Edit model."""
    print(f"Loading Qwen-Image-Edit model...")
    
    model = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    model.set_progress_bar_config(disable=True)
    print("✓ Qwen-Image-Edit model loaded")
    return model


def edit_image_directly(
    original_image: Image.Image,
    edit_instruction: str,
    edit_model,
    device: str = "cuda"
) -> Image.Image:
    """
    Direct image editing without planning using Qwen-Image-Edit.
    
    Args:
        original_image: PIL Image
        edit_instruction: Natural language edit instruction
        edit_model: Qwen-Image-Edit model
        device: Device to use
        
    Returns:
        Edited PIL Image
    """
    # Resize image to 768x768
    pil_image = original_image.resize((768, 768), Image.Resampling.LANCZOS)
    
    # Format prompt (simple wrapper like baseline)
    formatted_prompt = f"Editing Instruction: {edit_instruction}. Maintain high quality, original composition and style."
    
    # Direct call to Qwen-Image-Edit
    with torch.no_grad():
        output = edit_model(
            prompt=formatted_prompt,
            image=pil_image,
            height=768,
            width=768,
            num_inference_steps=28,
            guidance_scale=7.5,
        )
    
    return output.images[0]


def evaluate_edit_only(
    test_samples: List[str],
    results_base_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    checkpoint_num: int = None
):
    """
    Evaluate Edit-Only baseline using Qwen-Image-Edit.
    
    Args:
        test_samples: List of sample folder names
        results_base_dir: Base directory containing imageagent results
        output_dir: Output directory for results
        device: Device to use
        checkpoint_num: Optional checkpoint number for naming
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Edit-Only (E) Baseline Evaluation")
    print("="*70)
    print(f"Test samples: {len(test_samples)}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Load Qwen-Image-Edit model
    edit_model = load_edit_model(device)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(device=device)
    
    # Initialize GPT-4o Judge for image quality evaluation
    try:
        gpt_judge = GPT4oJudge()
        print("✓ GPT-4o Image Judge initialized")
    except Exception as e:
        print(f"⚠️  Could not initialize GPT-4o judge: {e}")
        gpt_judge = None
    
    # Process each sample
    results = []
    
    for idx, sample_name in enumerate(tqdm(test_samples, desc="Evaluating E baseline"), 1):
        try:
            sample_dir = results_base_dir / sample_name
            
            # Load original image
            original_path = sample_dir / "original.png"
            if not original_path.exists():
                print(f"⚠️  Skipping {sample_name}: No original image")
                continue
            
            original_image = Image.open(original_path).convert("RGB")
            
            # Load edit instruction
            prompt_path = sample_dir / "prompt.json"
            if not prompt_path.exists():
                print(f"⚠️  Skipping {sample_name}: No prompt file")
                continue
            
            with open(prompt_path, 'r') as f:
                prompt_data = json.load(f)
            
            # Extract edit instruction
            if 'edit_info' in prompt_data and 'text' in prompt_data['edit_info']:
                edit_instruction = prompt_data['edit_info']['text']
            elif 'edit' in prompt_data and 'text' in prompt_data['edit']:
                edit_instruction = prompt_data['edit']['text']
            else:
                print(f"⚠️  Skipping {sample_name}: No edit instruction")
                continue
            
            # Edit image directly (no planning)
            edited_image = edit_image_directly(
                original_image=original_image,
                edit_instruction=edit_instruction,
                edit_model=edit_model,
                device=device
            )
            
            # Save outputs (match baseline format with samples/ subdirectory)
            sample_output_dir = output_dir / "samples" / sample_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as predicted_edit.png (to match baseline naming)
            predicted_path = sample_output_dir / "predicted_edit.png"
            edited_image.save(predicted_path)
            
            # Save original.png (match baseline naming)
            original_output_path = sample_output_dir / "original.png"
            original_image.save(original_output_path)
            
            # Save edit prompt
            with open(sample_output_dir / "edit_prompt.txt", 'w') as f:
                f.write(edit_instruction)
            
            # Load ground truth edited image
            gt_edit_path = sample_dir / "edited.png"
            if gt_edit_path.exists():
                gt_edit = Image.open(gt_edit_path).convert("RGB")
                gt_edit.save(sample_output_dir / "ground_truth.png")
                
                # Create 3-way comparison (Original | Predicted | Ground Truth) matching baseline format
                size = (384, 384)
                top_margin = 40
                bottom_margin = 80
                
                # Resize images
                orig_resized = original_image.resize(size, Image.Resampling.LANCZOS)
                pred_resized = edited_image.resize(size, Image.Resampling.LANCZOS)
                gt_resized = gt_edit.resize(size, Image.Resampling.LANCZOS)
                
                # Create comparison with margins
                comparison = Image.new('RGB', (size[0] * 3, size[1] + top_margin + bottom_margin), color='white')
                
                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comparison)
                try:
                    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                except:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()
                
                # Paste images
                comparison.paste(orig_resized, (0, top_margin))
                comparison.paste(pred_resized, (size[0], top_margin))
                comparison.paste(gt_resized, (size[0] * 2, top_margin))
                
                # Add labels
                draw.text((size[0]//2 - 30, 5), "Original", fill="black", font=font_large)
                draw.text((size[0] + size[0]//2 - 40, 5), "Predicted", fill="black", font=font_large)
                draw.text((size[0]*2 + size[0]//2 - 55, 5), "Ground Truth", fill="black", font=font_large)
                
                # Add instruction
                instruction_text = edit_instruction
                if len(instruction_text) > 120:
                    instruction_text = instruction_text[:117] + "..."
                draw.text((10, size[1] + top_margin + 10), f"Instruction: {instruction_text}", fill="black", font=font_small)
                
                comparison.save(sample_output_dir / "comparison.png")
            
            # Calculate image metrics (if edited image exists in ground truth)
            edited_gt_path = sample_dir / "edited.png"
            image_metrics = {}
            
            if edited_gt_path.exists():
                edited_gt = Image.open(edited_gt_path).convert("RGB")
                # Compute image metrics (LPIPS, SSIM, PSNR, etc.)
                pred_tensor = torch.from_numpy(np.array(edited_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                gt_tensor = torch.from_numpy(np.array(edited_gt)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                pred_tensor = pred_tensor.to(metrics_calc.device)
                gt_tensor = gt_tensor.to(metrics_calc.device)
                
                image_metrics = {
                    'lpips': float(metrics_calc.compute_lpips(pred_tensor, gt_tensor)),
                    'ssim': float(metrics_calc.compute_ssim(pred_tensor, gt_tensor)),
                    'psnr': float(metrics_calc.compute_psnr(pred_tensor, gt_tensor))
                }
                
                # Add CLIP score (text-image alignment)
                try:
                    clip_score = metrics_calc.compute_clip_score(pred_tensor, [edit_instruction])
                    image_metrics['clip_score'] = float(clip_score)
                except Exception as e:
                    print(f"    ⚠️  CLIP computation failed: {e}")
                    image_metrics['clip_score'] = None
            
            # GPT-4o Image Quality Evaluation (Edit-Only: 3 dimensions only)
            if gpt_judge:
                try:
                    # Call GPT-4o to evaluate image quality
                    print(f"    🤖 Evaluating with GPT-4o Image Judge...")
                    gpt_scores = gpt_judge.judge_single_edit(
                        original_image=original_image,
                        generated_image=edited_image,
                        instruction=edit_instruction
                    )
                    
                    # Add GPT-4o scores to metrics (Edit-Only: only 3 dimensions)
                    # instruction_following, visual_quality, overall_image_score
                    image_metrics['gpt_judge_instruction_following'] = gpt_scores.get('instruction_following', 0.0)
                    image_metrics['gpt_judge_visual_quality'] = gpt_scores.get('visual_quality', 0.0)
                    image_metrics['gpt_judge_overall'] = gpt_scores.get('overall_image_score', 0.0)
                    image_metrics['gpt_judge_reasoning'] = gpt_scores.get('reasoning', '')
                    
                    print(f"    ✅ GPT-4o evaluation complete")
                    
                except Exception as e:
                    print(f"    ⚠️  GPT-4o evaluation failed for {sample_name}: {e}")
                    # Set default values on failure
                    image_metrics['gpt_judge_instruction_following'] = 0.0
                    image_metrics['gpt_judge_visual_quality'] = 0.0
                    image_metrics['gpt_judge_overall'] = 0.0
                    image_metrics['gpt_judge_reasoning'] = f"Error: {str(e)}"
            
            # Save image metrics
            if image_metrics:
                with open(sample_output_dir / "metrics.json", 'w') as f:
                    json.dump(image_metrics, f, indent=2)
            
            # Save metadata
            metadata = {
                "sample_id": sample_name,
                "edit_instruction": edit_instruction,
                "method": "edit_only",
                "planning_used": False
            }
            
            with open(sample_output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Store result
            result = {
                "sample_id": sample_name,
                "image_metrics": image_metrics
            }
            results.append(result)
            
            # Rate limiting: Add delay after every 10 samples
            if idx % 10 == 0 and idx < len(test_samples):
                print(f"\n⏸️  Rate limiting: Pausing 5 seconds after {idx} samples...")
                time.sleep(5)
            
        except Exception as e:
            print(f"❌ Error processing {sample_name}: {e}")
            continue
    
    # Save detailed results
    split = "test"  # Edit-only always uses test split
    detailed_path = output_dir / f"detailed_results_{split}.json"
    
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processed {len(results)} samples")
    print(f"✓ Results saved to: {output_dir}")
    
    # Calculate aggregates
    if results:
        aggregates = {}
        
        # Image metrics (objective + GPT-4o)
        for metric_name in ["lpips", "ssim", "psnr", "clip_score", 
                           "gpt_judge_instruction_following", "gpt_judge_visual_quality", "gpt_judge_overall"]:
            values = [r["image_metrics"].get(metric_name) for r in results 
                     if r.get("image_metrics") and metric_name in r["image_metrics"] 
                     and r["image_metrics"][metric_name] is not None]
            if values:
                aggregates[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Save aggregates
        summary_path = output_dir / f"evaluation_summary_{split}.json"
        summary = {
            "num_samples": len(results),
            "metrics": aggregates
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Edit-Only (E) Baseline")
    
    # Required arguments
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        required=True,
        help="File containing sample IDs to evaluate (one per line)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Base directory containing imageagent results"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Optional checkpoint number for naming"
    )
    
    args = parser.parse_args()
    
    # Load sample IDs
    with open(args.sample_ids_file, 'r') as f:
        test_samples = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(test_samples)} test samples from {args.sample_ids_file}")
    
    # Run evaluation
    evaluate_edit_only(
        test_samples=test_samples,
        results_base_dir=Path(args.results_dir),
        output_dir=Path(args.output),
        device=args.device,
        checkpoint_num=args.checkpoint
    )


if __name__ == "__main__":
    main()

