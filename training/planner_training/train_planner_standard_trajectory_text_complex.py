"""
Train Action Planner - Text-Only Mode

Fine-tune Qwen3-VL using text prompts only (no images) for ultra-fast training.
"""

import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.planner_training.planner_dataset_text_only import create_dataloaders


def load_config(config_path: str = None):
    """Load training configuration."""
    if config_path is None:
        config_path = SCRIPT_DIR / "configs_complex_8b" / "planner_config_standard_trajectory_text.yaml"
    
    config = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    return config


def initialize_model_and_processor(config):
    """Initialize Qwen3-VL model and processor."""
    print(f"Loading model: {config.model.base_model}")
    
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
    
    print(f"✓ Model loaded: {config.model.base_model}")
    print(f"✓ Model dtype: {config.model.dtype}")
    
    # Freeze vision encoder (not used in text-only mode)
    if config.model.get("freeze_vision_encoder", True):
        print("Freezing vision encoder...")
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = False
        elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
            for param in model.model.visual.parameters():
                param.requires_grad = False
        print("✓ Vision encoder frozen (text-only mode)")
    
    return model, processor


def setup_lora(model, config):
    """Setup LoRA for efficient fine-tuning."""
    if not config.lora.enabled:
        print("LoRA disabled, training full model")
        return model
    
    print("Setting up LoRA...")
    
    # Convert target_modules from OmegaConf to list
    target_modules = list(config.lora.target_modules)
    
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for training (always prepare, even without 8bit)
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.hardware.gradient_checkpointing)
    
    model = get_peft_model(model, lora_config)
    
    # Enable input gradients if using gradient checkpointing
    if config.hardware.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ LoRA configured")
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
    parser = argparse.ArgumentParser(description="Train Qwen3-VL Action Planner (Text-Only)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs_complex_8b/planner_config_standard_trajectory_text.yaml",
        help="Path to config file (default: configs_complex_8b/planner_config_standard_trajectory_text.yaml)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override training data path from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Action Planner Training - Text-Only Mode")
    print("=" * 60)
    print()
    
    # Load configuration
    config_path = Path(args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isabs(args.config):
        config_path = SCRIPT_DIR / args.config
    config = load_config(config_path)
    
    # Override config with CLI arguments
    if args.data_path:
        config.data.training_data_path = args.data_path
        print(f"✓ Data path overridden: {args.data_path}")
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
        print(f"✓ Output dir overridden: {args.output_dir}")
    
    print("⚡ Text-only mode: Images will be ignored")
    print("⚡ Expected speedup: ~10× faster than vision-language training")
    
    # Multi-GPU setup
    num_gpus = config.hardware.get("num_gpus", 1)
    effective_batch = num_gpus * config.data.batch_size * config.training.gradient_accumulation_steps
    if num_gpus > 1:
        print(f"🎮 Multi-GPU Training: Using {num_gpus} GPUs")
        print(f"📊 Per-GPU batch: {config.data.batch_size}, Gradient accum: {config.training.gradient_accumulation_steps}")
        print(f"📊 Effective batch size: {num_gpus} × {config.data.batch_size} × {config.training.gradient_accumulation_steps} = {effective_batch}")
    else:
        print(f"🎮 Single GPU Training")
        print(f"📊 Effective batch size: {effective_batch}")
    
    # Initialize wandb if enabled (only on main process for DDP)
    if config.logging.use_wandb:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            import wandb
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_run_name,
                config=OmegaConf.to_container(config, resolve=True)
            )
            print("✓ Weights & Biases initialized (main process)")
        else:
            print(f"✓ WandB skipped on rank {local_rank}")
    
    # Initialize model and processor
    model, processor = initialize_model_and_processor(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Create dataloaders (text-only)
    print("\nCreating dataloaders (text-only mode)...")
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=train_loader.collate_fn,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 Starting Training (Text-Only)")
    print("=" * 60)
    print()
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 60)
    print("💾 Saving Final Model")
    print("=" * 60)
    
    final_output_dir = Path(config.output.output_dir) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(final_output_dir))
    processor.save_pretrained(str(final_output_dir))
    
    print(f"✓ Model saved to: {final_output_dir}")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("📊 Final Evaluation")
    print("=" * 60)
    
    eval_results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Training Complete!")
    print("\n⚠️  Note: Text-only model has no visual context.")
    print("   Performance may be lower than vision-language models.")


if __name__ == "__main__":
    main()

