#!/usr/bin/env python3
"""
HiDream-E1 LoRA Checkpoint Evaluation Script

Comprehensive evaluation of trained checkpoints including:
- Quantitative metrics (LPIPS, SSIM, PSNR, CLIP)
- Sample generation with visualizations
- Comparison against ground truth
- HTML report generation

Usage:
    python scripts/evaluate_checkpoint.py \\
        --checkpoint checkpoints/hidream_e1_lora_theme_transform/best_model \\
        --output evaluation_results/best_model \\
        --num_samples 30
"""

# CRITICAL: Disable flash-attn BEFORE any imports
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["DIFFUSERS_ATTN_IMPLEMENTATION"] = "eager"

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import safetensors.torch

from training.data import ImageEditingDataset
from training.models import HiDreamE1LoRA
from training.evaluation.metrics import MetricsCalculator


class CheckpointEvaluator:
    """Evaluate trained HiDream-E1 LoRA checkpoints"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print("\n" + "="*70)
        print("CHECKPOINT EVALUATION")
        print("="*70 + "\n")
        
        # Load config
        print(f"📋 Loading configuration from {config_path}...")
        self.config = OmegaConf.load(config_path)
        
        # Load model
        print(f"🤖 Loading model from {self.checkpoint_path}...")
        self.model = self._load_checkpoint()
        
        # Initialize metrics calculator
        print(f"📊 Initializing metrics calculator...")
        self.metrics_calculator = MetricsCalculator(device=device)
        
        print(f"\n✅ Initialization complete!\n")
    
    def _load_checkpoint(self) -> HiDreamE1LoRA:
        """Load model with LoRA weights from checkpoint"""
        # Load base model
        model = HiDreamE1LoRA(self.config, device=self.device)
        
        # Try to load LoRA weights - check both possible filenames
        lora_weights_path = self.checkpoint_path / "adapter_model.safetensors"
        if not lora_weights_path.exists():
            lora_weights_path = self.checkpoint_path / "lora_weights.safetensors"
        
        if lora_weights_path.exists():
            print(f"   Loading LoRA weights from {lora_weights_path}")
            lora_state = safetensors.torch.load_file(str(lora_weights_path))
            
            # Load into transformer
            missing, unexpected = model.transformer.load_state_dict(lora_state, strict=False)
            print(f"   ✅ Loaded {len(lora_state)} LoRA parameters")
            if missing:
                print(f"   ⚠️  Missing keys: {len(missing)}")
            if unexpected:
                print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        else:
            print(f"   ⚠️  No LoRA weights found at checkpoint, using base model")
            print(f"   Checked: {self.checkpoint_path / 'adapter_model.safetensors'}")
            print(f"   Checked: {self.checkpoint_path / 'lora_weights.safetensors'}")
        
        # Set models to eval mode
        model.transformer.eval()
        model.vae.eval()
        for encoder in model.text_encoders.values():
            if encoder is not None:
                encoder.eval()
        
        return model
    
    @torch.no_grad()
    def generate_edit(
        self,
        source_image: torch.Tensor,
        instruction: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 5.0,
        image_guidance_scale: float = 4.0,
    ) -> torch.Tensor:
        """Generate edited image using the model
        
        Uses the HiDreamImageEditingPipeline following the official pattern from
        HiDream-E1 README.
        
        Args:
            source_image: Source image tensor [1, 3, H, W] in range [-1, 1]
            instruction: Text instruction for editing
            num_inference_steps: Number of denoising steps (default: 28)
            guidance_scale: Text guidance scale (default: 5.0)
            image_guidance_scale: Image conditioning scale (default: 4.0)
            
        Returns:
            Generated image tensor [1, 3, H, W] in range [-1, 1]
        """
        if self.model.pipeline is None:
            raise RuntimeError("Pipeline not available - cannot generate edits")
        
        # Convert tensor to PIL Image
        # Source tensor is [1, 3, H, W] in range [-1, 1]
        # Must convert to float32 first because numpy doesn't support bfloat16
        source_np = source_image[0].cpu().float().numpy()  # [3, H, W]
        source_np = (source_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        source_np = np.clip(source_np * 255, 0, 255).astype(np.uint8)
        source_np = np.transpose(source_np, (1, 2, 0))  # [H, W, 3]
        pil_image = Image.fromarray(source_np, mode='RGB')
        
        # CRITICAL: Resize to exactly 768×768 (official HiDream-E1 pattern)
        # Model has max_resolution=[96, 192] latent with max_seq=4608 patches
        # This supports 768×768 images (96×96 latent = 768 pixels)
        # See: https://huggingface.co/HiDream-ai/HiDream-E1-Full#quick-start
        pil_image = pil_image.resize((768, 768), Image.Resampling.LANCZOS)
        
        # Format instruction in official format:
        # "Editing Instruction: {instruction}. Target Image Description: {description}"
        # For now, use the instruction as-is (it should already be formatted)
        formatted_prompt = instruction
        
        # Generate with pipeline (following official example)
        result = self.model.pipeline(
            prompt=formatted_prompt,
            negative_prompt="low resolution, blur, distorted",
            image=pil_image,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(self.device.type).manual_seed(42),
        )
        
        # Convert back to tensor
        output_pil = result.images[0]
        output_np = np.array(output_pil).astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
        output_np = output_np * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        output_tensor = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return output_tensor.to(device=self.device, dtype=self.model.dtype)
    
    def evaluate_sample(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        instruction: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate a single sample
        
        Returns:
            (generated_image, metrics_dict)
        """
        # Generate edit
        generated = self.generate_edit(source_image, instruction)
        
        # Convert to float32 for metrics (bfloat16 not supported by all metrics)
        generated_float = generated.float()
        target_float = target_image.to(self.device).float()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            generated_float,
            target_float,
            [instruction]
        )
        
        return generated, metrics
    
    def evaluate_dataset(
        self,
        dataset: ImageEditingDataset,
        num_samples: int = None,
        save_samples: bool = True,
        output_dir: Path = None
    ) -> Dict:
        """Evaluate on entire dataset or subset
        
        Args:
            dataset: Dataset to evaluate
            num_samples: Number of samples to evaluate (None = all)
            save_samples: Whether to save generated samples
            output_dir: Directory to save results
            
        Returns:
            Dictionary with aggregated results
        """
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        print(f"\n{'='*70}")
        print(f"EVALUATING {num_samples} SAMPLES")
        print(f"{'='*70}\n")
        
        all_metrics = []
        samples_data = []
        
        # Create output directory
        if save_samples and output_dir:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate samples
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[idx]
            
            # Get data
            source_img = sample['source_image'].unsqueeze(0)  # Add batch dim
            target_img = sample['target_image'].unsqueeze(0)
            instruction = sample['instruction']
            
            # Generate and evaluate
            generated, metrics = self.evaluate_sample(
                source_img,
                target_img,
                instruction
            )
            
            all_metrics.append(metrics)
            
            # Save sample if requested
            if save_samples and output_dir:
                sample_dir = samples_dir / f"sample_{idx:03d}"
                sample_dir.mkdir(exist_ok=True)
                
                # Save images
                self._save_image(source_img[0], sample_dir / "source.png")
                self._save_image(target_img[0], sample_dir / "target.png")
                self._save_image(generated[0], sample_dir / "generated.png")
                
                # Save instruction
                with open(sample_dir / "instruction.txt", 'w') as f:
                    f.write(instruction)
                
                # Save metrics (convert numpy types to Python types for JSON)
                metrics_serializable = {k: float(v) for k, v in metrics.items()}
                with open(sample_dir / "metrics.json", 'w') as f:
                    json.dump(metrics_serializable, f, indent=2)
                
                # Create comparison image
                self._create_comparison(
                    source_img[0],
                    generated[0],
                    target_img[0],
                    instruction,
                    sample_dir / "comparison.png"
                )
                
                samples_data.append({
                    "idx": idx,
                    "instruction": instruction,
                    "metrics": metrics
                })
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        
        # Print results
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}\n")
        self._print_metrics(aggregated)
        
        # Save aggregated results
        if output_dir:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                """Recursively convert numpy types to Python types"""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            results = {
                "checkpoint": str(self.checkpoint_path),
                "num_samples": num_samples,
                "timestamp": datetime.now().isoformat(),
                "aggregated_metrics": convert_to_serializable(aggregated),
                "per_sample_metrics": convert_to_serializable(samples_data)
            }
            
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Results saved to {output_dir}")
        
        return aggregated
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across samples"""
        aggregated = {}
        
        # Get all metric names
        metric_names = set()
        for m in metrics_list:
            metric_names.update(m.keys())
        
        # Compute mean and std for each metric
        for name in metric_names:
            values = [m[name] for m in metrics_list if name in m]
            if values:
                aggregated[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return aggregated
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way"""
        print("Metric               Mean    Std     Min     Max")
        print("-" * 55)
        
        for name, stats in metrics.items():
            print(f"{name:15s}  {stats['mean']:7.4f} {stats['std']:7.4f} {stats['min']:7.4f} {stats['max']:7.4f}")
    
    def _save_image(self, tensor: torch.Tensor, path: Path):
        """Save tensor as image"""
        # Convert from [-1, 1] to [0, 255]
        # Must convert to float32 first because numpy doesn't support bfloat16
        img = ((tensor.cpu().float() + 1) / 2 * 255).clamp(0, 255)
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(img).save(path)
    
    def _create_comparison(
        self,
        source: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
        instruction: str,
        path: Path
    ):
        """Create side-by-side comparison image"""
        # Convert tensors to numpy arrays
        def tensor_to_np(t):
            # Must convert to float32 first because numpy doesn't support bfloat16
            img = ((t.cpu().float() + 1) / 2 * 255).clamp(0, 255)
            return img.permute(1, 2, 0).numpy().astype(np.uint8)
        
        source_np = tensor_to_np(source)
        generated_np = tensor_to_np(generated)
        target_np = tensor_to_np(target)
        
        # Create comparison image
        h, w = source_np.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = source_np
        comparison[:, w:2*w] = generated_np
        comparison[:, 2*w:] = target_np
        
        # Save
        img = Image.fromarray(comparison)
        
        # Add text labels if PIL supports it
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            # Use default font
            draw.text((10, 10), "Source", fill=(255, 255, 255))
            draw.text((w + 10, 10), "Generated", fill=(255, 255, 255))
            draw.text((2*w + 10, 10), "Target", fill=(255, 255, 255))
        except:
            pass
        
        img.save(path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HiDream-E1 LoRA checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation (10 samples, no visualizations)"
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        default=True,
        help="Save generated samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        default=None,
        help="Override data directory from config"
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.num_samples = 10
        args.save_samples = False
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = CheckpointEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Override data_dirs if provided
    if args.data_dirs:
        evaluator.config.data.data_dirs = [args.data_dirs]
        print(f"✓ Data dir overridden: {args.data_dirs}\n")
    
    # Load validation dataset
    print("📦 Loading validation dataset...")
    from training.data import prepare_datasets
    _, val_dataset = prepare_datasets(evaluator.config, verbose=False)
    print(f"   ✅ Loaded {len(val_dataset)} validation samples\n")
    
    # Run evaluation
    start_time = time.time()
    
    results = evaluator.evaluate_dataset(
        dataset=val_dataset,
        num_samples=args.num_samples,
        save_samples=args.save_samples,
        output_dir=output_dir
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ Evaluation complete in {elapsed:.1f} seconds")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


