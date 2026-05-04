#!/usr/bin/env python3
"""
HiDream-E1 LoRA Finetuning - Main Training Script

Usage:
    python training/train.py --config training/config/training_config.yaml
    python training/train.py --config training/config/training_config.yaml --resume_from checkpoint-10
"""

# CRITICAL: Disable flash-attn BEFORE any imports
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["DIFFUSERS_ATTN_IMPLEMENTATION"] = "eager"

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from training.data import prepare_datasets
from training.models import load_model_with_lora
from training.training import HiDreamE1Trainer
from training.utils import set_seed, save_config


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "training.log")
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HiDream-E1 with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        default=None,
        help="Override data directories from config (can be comma-separated)"
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    print(f"\n{'='*60}")
    print(f"HIDREAM-E1 LORA FINETUNING")
    print(f"{'='*60}\n")
    
    print(f"📝 Loading configuration from: {args.config}")
    config = OmegaConf.load(args.config)
    
    # Override output dir if specified
    if args.output_dir:
        config.training.output_dir = args.output_dir
        print(f"✓ Output dir overridden: {args.output_dir}")
    
    # Override data dirs if specified
    if args.data_dirs:
        # Split by comma if multiple directories provided
        data_dirs = [d.strip() for d in args.data_dirs.split(',')]
        config.data.data_dirs = data_dirs
        print(f"✓ Data dirs overridden: {data_dirs}")
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting HiDream-E1 LoRA finetuning")
    
    # Save config to output directory
    save_config(config, output_dir / "training_config.yaml")
    
    # Set random seed
    set_seed(config.seed, config.get('deterministic', False))
    
    # Prepare datasets
    logger.info("\n📦 Preparing datasets...")
    try:
        train_dataset, val_dataset = prepare_datasets(config, verbose=True)
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        raise
    
    # Load model with LoRA
    logger.info("\n🤖 Loading model with LoRA...")
    try:
        model = load_model_with_lora(config, device=config.model.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("\nTroubleshooting tips:")
        logger.info("  1. Make sure you have access to meta-llama/Meta-Llama-3.1-8B-Instruct")
        logger.info("  2. Run: huggingface-cli login")
        logger.info("  3. Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        raise
    
    # Create trainer
    logger.info("\n🎓 Creating trainer...")
    trainer = HiDreamE1Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"\n📂 Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    logger.info("\n🚀 Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(epoch='interrupted')
        logger.info("✅ Checkpoint saved")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Final checkpoint: {output_dir}/checkpoint-final")
    logger.info(f"\nTo use the finetuned model:")
    logger.info(f"  python training/inference.py \\")
    logger.info(f"    --lora_path {output_dir}/checkpoint-final \\")
    logger.info(f"    --source_image your_image.png \\")
    logger.info(f"    --instruction 'Your editing instruction'")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()

