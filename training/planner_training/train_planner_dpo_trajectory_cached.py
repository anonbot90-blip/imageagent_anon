"""
Train Action Planner - DPO (Direct Preference Optimization) Cached Embeddings Mode

Fine-tune Qwen3-VL using preference pairs with pre-computed vision embeddings.
Learns to prefer high-quality plans over low-quality plans (3× faster with cached embeddings).
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
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from copy import deepcopy

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.planner_training.planner_dataset_dpo_cached import create_dataloaders
from training.planner_training.dpo_trainer import DPOTrainer


def load_config(config_path: str = None):
    """Load training configuration."""
    if config_path is None:
        config_path = SCRIPT_DIR / "planner_config_dpo_trajectory_cached.yaml"
    
    config = OmegaConf.load(config_path)
    print(f"Loaded config from: {config_path}")
    return config


def initialize_model_and_processor(config, quantization_config=None, device_map=None):
    """Initialize Qwen3-VL model and processor."""
    print(f"Loading model: {config.model.base_model}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model.base_model,
        trust_remote_code=config.model.trust_remote_code
    )
    
    # Load model
    model_kwargs = {
        "torch_dtype": getattr(torch, config.model.dtype),
        "trust_remote_code": config.model.trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForImageTextToText.from_pretrained(
        config.model.base_model,
        **model_kwargs
    )
    
    print(f"✓ Model loaded: {config.model.base_model}")
    print(f"✓ Model dtype: {config.model.dtype}")
    
    # Freeze vision encoder (not needed with cached embeddings)
    if config.model.get("freeze_vision_encoder", True):
        print("Freezing vision encoder...")
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = False
        elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
            for param in model.model.visual.parameters():
                param.requires_grad = False
        print("✓ Vision encoder frozen")
    
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
    
    # Prepare model for training
    if config.hardware.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    
    # Enable input gradients for gradient checkpointing compatibility
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
    parser = argparse.ArgumentParser(description="Train Qwen3-VL Action Planner (Cached Embeddings)")
    parser.add_argument(
        "--config",
        type=str,
        default="planner_config_dpo_trajectory_cached.yaml",
        help="Path to config file (default: planner_config_dpo_trajectory_cached.yaml)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override training data path from config"
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=None,
        help="Override embeddings HDF5 path from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Action Planner Training - DPO Cached Embeddings Mode")
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
    
    if args.embeddings_path:
        config.data.embeddings_path = args.embeddings_path
        print(f"✓ Embeddings path overridden: {args.embeddings_path}")
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
        print(f"✓ Output dir overridden: {args.output_dir}")
    
    # Check if embeddings exist (resolve relative to PROJECT_ROOT)
    embeddings_path = Path(config.data.embeddings_path)
    if not embeddings_path.is_absolute():
        embeddings_path = PROJECT_ROOT / embeddings_path
    if not embeddings_path.exists():
        print(f"\n❌ Error: Embeddings file not found: {embeddings_path}")
        print("\nPlease run pre-computation first:")
        print(f"  python scripts/precompute_image_embeddings.py \\")
        print(f"    --data-path {config.data.training_data_path} \\")
        print(f"    --output-dir {embeddings_path.parent}")
        sys.exit(1)
    
    print(f"✓ Using cached embeddings: {embeddings_path}")
    
    # Resolve data path relative to PROJECT_ROOT
    data_path = Path(config.data.training_data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    
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
            try:
                wandb.init(
                    project=config.logging.wandb_project,
                    name=config.logging.wandb_run_name,
                    config=OmegaConf.to_container(config, resolve=True)
                )
                print("✓ Weights & Biases initialized (main process)")
            except Exception as e:
                print(f"⚠️  WandB initialization failed: {e}")
                print("   Continuing without WandB logging...")
        else:
            print(f"✓ WandB skipped on rank {local_rank}")
    
    # Initialize policy model and processor
    model, processor = initialize_model_and_processor(config)
    
    # Setup LoRA for policy model
    model = setup_lora(model, config)
    
    # Create reference model (frozen, int8) to save GPU memory
    print("\n🔧 Creating reference model (frozen, int8)...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"],
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    ref_model, _ = initialize_model_and_processor(
        config,
        quantization_config=bnb_config,
        device_map={"": device}
    )
    ref_model = setup_lora(ref_model, config)
    
    # Copy weights from policy model to reference model
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print(f"✓ Reference model created (int8), weights copied to {device}")
    
    # Create DPO dataloaders with cached embeddings
    print("\nCreating DPO dataloaders with cached embeddings...")
    train_loader, val_loader = create_dataloaders(
        data_path=str(data_path),
        processor=processor,
        embeddings_path=str(embeddings_path),
        batch_size=config.data.batch_size,
        train_val_split=config.data.train_val_split,
        num_workers=config.data.num_workers,
        max_length=config.training.max_length
    )
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Get DPO beta parameter
    dpo_beta = config.training.get("dpo_beta", 0.1)
    
    # Create DPO trainer (reference model on GPU with 8-bit quantization)
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=dpo_beta,
        ref_model_device=device,  # Keep reference model on GPU (8-bit quantized, uses less memory)
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=train_loader.collate_fn,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 Starting DPO Training (Cached Embeddings)")
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


if __name__ == "__main__":
    main()

