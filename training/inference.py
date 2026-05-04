#!/usr/bin/env python3
"""
Inference with Finetuned HiDream-E1 LoRA

Usage:
    python training/inference.py \
        --lora_path ./checkpoints/hidream_e1_lora_theme_transform/checkpoint-final \
        --source_image ./test_image.png \
        --instruction "Transform to cyberpunk style" \
        --output ./output.png
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from training.models import load_model_with_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with finetuned HiDream-E1 LoRA")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to finetuned LoRA weights"
    )
    parser.add_argument(
        "--source_image",
        type=str,
        required=True,
        help="Path to source image to edit"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Editing instruction"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output path for edited image"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/training_config.yaml",
        help="Path to training config (for model settings)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for diffusion"
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=4.0,
        help="Image guidance scale"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=28,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help="Resolution for inference"
    )
    
    return parser.parse_args()


def format_instruction(instruction: str, source_image: Image.Image = None) -> str:
    """Format instruction in HiDream-E1 format
    
    Args:
        instruction: User instruction
        source_image: Optional source image for context
        
    Returns:
        Formatted instruction
    """
    # Check if already in correct format
    if instruction.startswith("Editing Instruction:"):
        return instruction
    
    # Format as HiDream-E1 expects
    formatted = (
        f"Editing Instruction: {instruction}. "
        f"Target Image Description: An image where {instruction.lower()}, "
        f"maintaining the original composition and structure."
    )
    
    return formatted


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"HIDREAM-E1 LORA INFERENCE")
    print(f"{'='*60}\n")
    
    # Load config
    print(f"📝 Loading configuration...")
    config = OmegaConf.load(args.config)
    
    # Load model
    print(f"🤖 Loading base model...")
    model = load_model_with_lora(config, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load finetuned LoRA weights
    print(f"📂 Loading LoRA weights from: {args.lora_path}")
    model.load_lora_weights(args.lora_path)
    
    # Load source image
    print(f"🖼️  Loading source image: {args.source_image}")
    source_image = Image.open(args.source_image).convert("RGB")
    print(f"   Original size: {source_image.size}")
    
    # Resize to resolution
    source_image = source_image.resize((args.resolution, args.resolution))
    print(f"   Resized to: {args.resolution}x{args.resolution}")
    
    # Format instruction
    instruction = format_instruction(args.instruction, source_image)
    print(f"\n📝 Instruction:")
    print(f"   {instruction}")
    
    # Generate edited image
    print(f"\n🎨 Generating edited image...")
    print(f"   Guidance scale: {args.guidance_scale}")
    print(f"   Image guidance scale: {args.image_guidance_scale}")
    print(f"   Inference steps: {args.num_steps}")
    print(f"   Seed: {args.seed}")
    
    try:
        # Check if pipeline is available
        if model.pipeline is None:
            print(f"\n⚠️  Warning: HiDreamImageEditingPipeline not available")
            print(f"   Cannot generate edited image without the pipeline")
            print(f"\n💡 Make sure the HiDream-E1 pipeline is properly installed")
            return
        
        # Generate
        generator = torch.Generator(model.device).manual_seed(args.seed)
        
        result = model.pipeline(
            prompt=instruction,
            negative_prompt="low resolution, blur, distorted, low quality",
            image=source_image,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            num_inference_steps=args.num_steps,
            generator=generator,
        )
        
        edited_image = result.images[0]
        
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edited_image.save(output_path)
        
        print(f"\n✅ Edited image saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        print(f"\nThis is expected if the pipeline is not fully implemented yet.")
        print(f"The LoRA weights were loaded successfully.")
        raise
    
    print(f"\n{'='*60}")
    print(f"✅ INFERENCE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

