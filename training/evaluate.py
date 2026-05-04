#!/usr/bin/env python3
"""
Evaluate Finetuned HiDream-E1 LoRA

Usage:
    python training/evaluate.py \
        --lora_path ./checkpoints/hidream_e1_lora_theme_transform/checkpoint-final \
        --data_dir ./imageagent_results_training_300 \
        --output_dir ./evaluation_results \
        --num_samples 10
"""

import argparse
import sys
from pathlib import Path
import json
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from training.models import load_model_with_lora
from training.data import ImageEditingDataset
from training.evaluation import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuned HiDream-E1 LoRA")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to finetuned LoRA weights"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"HIDREAM-E1 LORA EVALUATION")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"📝 Loading configuration...")
    config = OmegaConf.load(args.config)
    
    # Load dataset
    print(f"📦 Loading evaluation dataset from: {args.data_dir}")
    dataset = ImageEditingDataset(
        data_dirs=[args.data_dir],
        resolution=768,
        center_crop=True,
        random_flip=False,
        return_paths=True,
    )
    print(f"   Total samples: {len(dataset)}")
    
    # Determine number of samples to evaluate
    num_samples = args.num_samples if args.num_samples is not None else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"   Evaluating: {num_samples} samples")
    
    # Load model
    print(f"\n🤖 Loading model with LoRA...")
    model = load_model_with_lora(config, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load finetuned weights
    print(f"📂 Loading LoRA weights from: {args.lora_path}")
    model.load_lora_weights(args.lora_path)
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator(device=model.device)
    
    # Evaluate
    print(f"\n🔍 Running evaluation...")
    
    all_metrics = []
    
    for idx in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[idx]
        
        # Get source and target
        source_img = sample['source_image'].unsqueeze(0).to(model.device)
        target_img = sample['target_image'].unsqueeze(0).to(model.device)
        instruction = sample['instruction']
        sample_id = sample['sample_id']
        
        # TODO: Generate edited image with model
        # For now, we'll just compute metrics between source and target
        # as a placeholder
        
        # In actual implementation, you would do:
        # with torch.no_grad():
        #     edited_img = model.pipeline(
        #         prompt=instruction,
        #         image=source_img,
        #         ...
        #     ).images[0]
        
        # Placeholder: use target as "edited" for testing
        edited_img = target_img
        
        # Compute metrics
        sample_metrics = metrics_calculator.compute_all_metrics(
            edited_img,
            target_img,
            instructions=[instruction]
        )
        
        sample_metrics['sample_id'] = sample_id
        all_metrics.append(sample_metrics)
        
        # Save images if requested
        if args.save_images:
            sample_output_dir = output_dir / "images" / sample_id
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save source, target, edited
            Image.open(sample['source_path']).save(sample_output_dir / "source.png")
            Image.open(sample['target_path']).save(sample_output_dir / "target.png")
            # edited would be saved here in actual implementation
    
    # Compute average metrics
    print(f"\n📊 Computing statistics...")
    
    avg_metrics = {}
    metric_names = [k for k in all_metrics[0].keys() if k != 'sample_id']
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            avg_metrics[metric_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
            }
    
    # Save results
    results = {
        'lora_path': str(args.lora_path),
        'data_dir': str(args.data_dir),
        'num_samples': num_samples,
        'average_metrics': avg_metrics,
        'per_sample_metrics': all_metrics,
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📈 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nAverage Metrics:")
    for metric_name, stats in avg_metrics.items():
        print(f"  {metric_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Min:  {stats['min']:.4f}")
        print(f"    Max:  {stats['max']:.4f}")
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

