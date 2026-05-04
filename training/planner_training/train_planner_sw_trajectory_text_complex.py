"""
Train Action Planner - Standardized Weighted Text-Only Mode

Fine-tune Qwen3-VL using standardized trajectory-level weights (text-only, no images).
Samples weighted by trajectory z-scores (standardized rewards).
"""

import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.planner_training.planner_dataset_sw_trajectory_text_complex import create_dataloaders
from training.planner_training.weighted_trainer import WeightedLossTrainer


def load_config(config_path: str = None):
    """Load training configuration."""
    if config_path is None:
        config_path = SCRIPT_DIR / "configs_complex_8b" / "planner_config_sw_trajectory_text.yaml"
    
    config = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    return config


def initialize_model_and_processor(config):
    """Initialize Qwen3-VL model and processor."""
    print("\n🔧 Initializing model and processor...")
    print(f"  Base model: {config.model.base_model}")
    print(f"  Dtype: {config.model.dtype}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model.base_model,
        trust_remote_code=config.model.trust_remote_code
    )
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        config.model.base_model,
        torch_dtype=getattr(torch, config.model.dtype),
        trust_remote_code=config.model.trust_remote_code
    )
    
    print("✓ Model and processor loaded")
    
    return model, processor


def setup_lora(model, config):
    """Setup LoRA for efficient fine-tuning."""
    if not config.lora.enabled:
        print("\n⚠️  LoRA disabled, training full model")
        return model
    
    print("\n🔧 Setting up LoRA...")
    print(f"  Rank: {config.lora.r}")
    print(f"  Alpha: {config.lora.lora_alpha}")
    print(f"  Dropout: {config.lora.lora_dropout}")
    print(f"  Target modules: {', '.join(config.lora.target_modules)}")
    
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=list(config.lora.target_modules),  # Convert ListConfig to list
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable input gradients for gradient checkpointing compatibility
    model.enable_input_require_grads()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")
    
    return model


def setup_training_args(config):
    """Setup training arguments."""
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.data.batch_size,
        per_device_eval_batch_size=config.data.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy=config.training.get("evaluation_strategy", "steps"),
        save_strategy=config.training.get("save_strategy", "steps"),
        save_total_limit=config.output.save_total_limit,
        load_best_model_at_end=config.output.load_best_model_at_end,
        metric_for_best_model=config.evaluation.metric_for_best_model,
        greater_is_better=config.evaluation.greater_is_better,
        bf16=config.hardware.mixed_precision == "bf16",
        fp16=config.hardware.mixed_precision == "fp16",
        gradient_checkpointing=config.hardware.gradient_checkpointing,
        report_to=config.logging.report_to if config.logging.use_wandb else "none",
        run_name=config.logging.wandb_run_name if config.logging.use_wandb else None,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.data.num_workers,
        ddp_find_unused_parameters=False,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )
    
    return training_args


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Qwen3-VL Action Planner (Standardized Weighted Text-Only)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs_complex_8b/planner_config_sw_trajectory_text.yaml",
        help="Path to config file (default: planner_config_sw_trajectory_text.yaml)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override training data path from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isabs(args.config):
        config_path = SCRIPT_DIR / args.config
    config = load_config(config_path)
    
    # Apply overrides
    if args.data_path:
        config.data.training_data_path = args.data_path
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    print("\n" + "="*70)
    print("Standardized Weighted Action Planner Training (Text-Only)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Mode: Text-only (no images)")
    print(f"  Weighting: Standardized trajectory-level (z-scores)")
    print(f"  Training data: {config.data.training_data_path}")
    print(f"  Output dir: {config.output.output_dir}")
    print(f"  Batch size: {config.data.batch_size} per GPU")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.data.batch_size * config.training.gradient_accumulation_steps * torch.cuda.device_count()}")
    print()
    
    # Initialize model and processor
    model, processor = initialize_model_and_processor(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Create dataloaders
    print("\n📊 Creating standardized weighted dataloaders (text-only)...")
    train_loader, val_loader = create_dataloaders(
        data_path=config.data.training_data_path,
        processor=processor,
        batch_size=config.data.batch_size,
        train_val_split=config.data.train_val_split,
        num_workers=config.data.num_workers,
        max_length=config.training.max_length
    )
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Initialize WeightedLossTrainer (handles standardized weights including negative values)
    print("\n🚀 Initializing Standardized Weighted Loss Trainer...")
    print("  Note: Trainer handles both positive and negative z-score weights")
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        processing_class=processor
    )
    
    # Train model
    print("\n🔥 Starting training...")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print()
    
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    final_dir = Path(config.output.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    
    print(f"\n✓ Training complete! Model saved to: {final_dir}")
    print("\nStandardized Weighted Training Summary:")
    print(f"  Weighting: Trajectory-level z-scores (standardized)")
    print(f"  Above-average trajectories: Positive weights (emphasized)")
    print(f"  Below-average trajectories: Negative weights (de-emphasized)")
    print(f"  Weight distribution: Standard normal ~N(0,1)")


if __name__ == "__main__":
    main()


