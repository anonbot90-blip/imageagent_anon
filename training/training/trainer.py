#!/usr/bin/env python3
"""
Main Trainer for HiDream-E1 LoRA Finetuning

Implements the training loop with distributed training support via Accelerate.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import logging
from typing import Optional, Dict
import json

from .loss import compute_loss_with_config, compute_diff_mask
from ..data.dataset import collate_fn
from ..evaluation.metrics import MetricsCalculator


class HiDreamE1Trainer:
    """Trainer for HiDream-E1 LoRA finetuning"""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        logger=None
    ):
        """
        Args:
            model: HiDreamE1LoRA model wrapper
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            logger: Optional logger
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup accelerator for distributed training
        self._setup_accelerator()
        
        # Create data loaders
        self._setup_dataloaders()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Prepare everything with accelerator
        self._prepare_training()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        self.logger.info("✅ Trainer initialized")
    
    def _setup_accelerator(self):
        """Setup Accelerate for distributed training"""
        project_config = ProjectConfiguration(
            project_dir=self.config.training.output_dir,
            logging_dir=str(Path(self.config.training.output_dir) / "logs"),
        )
        
        self.accelerator = Accelerator(
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with=self.config.logging.report_to if self.config.logging.report_to != "none" else None,
            project_config=project_config,
        )
        
        # Initialize tracking
        if self.accelerator.is_main_process and self.config.logging.report_to != "none":
            run_name = self.config.logging.get('wandb_run_name', 'hidream_e1_finetuning')
            self.accelerator.init_trackers(
                project_name=self.config.logging.get('wandb_project', 'hidream-e1'),
                config=dict(self.config),
                init_kwargs={
                    "wandb": {"name": run_name} if self.config.logging.use_wandb else {}
                }
            )
    
    def _setup_dataloaders(self):
        """Create data loaders"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Validate one at a time
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Get trainable parameters (only LoRA parameters)
        trainable_params = list(self.model.get_trainable_parameters())
        
        # Optimizer
        if self.config.training.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.config.training.learning_rate,
                    betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                    eps=self.config.training.adam_epsilon,
                    weight_decay=self.config.training.weight_decay,
                )
            except ImportError:
                self.logger.warning("bitsandbytes not available, using standard AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.training.learning_rate,
                    betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                    eps=self.config.training.adam_epsilon,
                    weight_decay=self.config.training.weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                eps=self.config.training.adam_epsilon,
                weight_decay=self.config.training.weight_decay,
            )
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        warmup_steps = self.config.training.lr_warmup_steps
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # Main scheduler
        if self.config.training.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.training.learning_rate * 0.1
            )
        else:
            main_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        
        # Combine warmup + main
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"LR Scheduler: {self.config.training.lr_scheduler}")
    
    def _prepare_training(self):
        """Prepare models and optimizers with accelerator"""
        (
            self.model.transformer,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model.transformer,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.lr_scheduler,
        )
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING TRAINING")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Val samples: {len(self.val_dataset)}")
        self.logger.info(f"Batch size: {self.config.training.train_batch_size}")
        self.logger.info(f"Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {self.config.training.train_batch_size * self.config.training.gradient_accumulation_steps}")
        self.logger.info(f"{'='*60}\n")
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            self.logger.info(f"\n📅 Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config.validation.validation_epochs == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.validation.save_model_epochs == 0:
                self.save_checkpoint(epoch=epoch)
        
        # Save final model
        self.save_checkpoint(epoch='final')
        
        # End tracking
        self.accelerator.end_training()
        
        self.logger.info(f"\n✅ Training complete!")
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.transformer.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader,
            disable=not self.accelerator.is_main_process,
            desc=f"Training Epoch {self.epoch + 1}"
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model.transformer):
                # Forward pass
                loss = self.train_step(batch)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Clip gradients
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.transformer.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                # Update weights
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # Track loss
            epoch_loss += loss.detach().item()
            
            # Log step
            if self.global_step % self.config.logging.log_every_n_steps == 0:
                self._log_training_step(loss.item(), step)
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})
        
        avg_loss = epoch_loss / len(self.train_loader)
        return {'train_loss': avg_loss}
    
    def train_step(self, batch) -> torch.Tensor:
        """Single training step"""
        source_img = batch['source_image']
        target_img = batch['target_image']
        instructions = batch['instruction']
        
        # Convert images to model dtype (bfloat16)
        source_img = source_img.to(dtype=self.model.dtype)
        target_img = target_img.to(dtype=self.model.dtype)
        
        # Encode images to latents
        with torch.no_grad():
            source_latents = self.model.vae.encode(source_img).latent_dist.sample()
            target_latents = self.model.vae.encode(target_img).latent_dist.sample()
            
            # Scale latents
            source_latents = source_latents * self.model.vae.config.scaling_factor
            target_latents = target_latents * self.model.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(target_latents)
        bsz = source_latents.shape[0]
        timesteps = torch.randint(
            0, self.config.diffusion.num_train_timesteps, (bsz,),
            device=source_latents.device
        ).long()
        
        # Add noise to target latents (flow matching)
        noisy_latents = self.model.scheduler.add_noise(target_latents, noise, timesteps)
        
        # Encode prompts using pipeline (if available)
        with torch.no_grad():
            if self.model.pipeline is not None:
                if not hasattr(self, '_first_encode_attempt'):
                    print(f"\n🔍 DEBUG: First encode attempt")
                    print(f"  - Pipeline type: {type(self.model.pipeline).__name__}")
                    print(f"  - Instructions: {instructions}")
                    print(f"  - Device: {noisy_latents.device}")
                    self._first_encode_attempt = True
                
                try:
                    # Use pipeline's encode_prompt for proper conditioning
                    encode_result = self.model.pipeline.encode_prompt(
                        prompt=instructions,
                        prompt_2=instructions,
                        prompt_3=instructions,
                        prompt_4=instructions,
                        device=noisy_latents.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    
                    # Unpack results - pipeline returns: (t5, neg_t5, llama, neg_llama, pooled, neg_pooled)
                    (
                        prompt_embeds_t5,
                        _,  # negative t5
                        prompt_embeds_llama3,
                        _,  # negative llama3
                        pooled_prompt_embeds,
                        _,  # negative pooled
                    ) = encode_result
                    
                    # Verify results
                    if pooled_prompt_embeds is None or prompt_embeds_t5 is None or prompt_embeds_llama3 is None:
                        if not hasattr(self, '_warned_about_none'):
                            print(f"\n❌ DEBUG: encode_prompt returned None values!")
                            print(f"  - pooled_prompt_embeds: {pooled_prompt_embeds}")
                            print(f"  - prompt_embeds_t5: {prompt_embeds_t5}")
                            print(f"  - prompt_embeds_llama3: {prompt_embeds_llama3}")
                            self._warned_about_none = True
                        raise ValueError("Pipeline encode_prompt returned None values")
                    
                    # Success!
                    if not hasattr(self, '_encoding_success'):
                        print(f"\n✅ Text encoding SUCCESS!")
                        print(f"  - pooled: {pooled_prompt_embeds.shape}, dtype: {pooled_prompt_embeds.dtype}")
                        print(f"  - t5: {prompt_embeds_t5.shape}, dtype: {prompt_embeds_t5.dtype}")
                        print(f"  - llama: {prompt_embeds_llama3.shape}, dtype: {prompt_embeds_llama3.dtype}")
                        print(f"  - Sample values (pooled): {pooled_prompt_embeds[0, :5]}")
                        self._encoding_success = True
                        
                except Exception as e:
                    # Fallback to zeros if encoding fails (only show warning once)
                    if not hasattr(self, '_warned_about_encoding'):
                        print(f"\n❌ Prompt encoding FAILED with exception:")
                        print(f"  Error: {e}")
                        import traceback
                        traceback.print_exc()
                        print("\n⚠️  Using zeros as fallback for all subsequent steps...")
                        print("  ⚠️  WARNING: Model will NOT learn instruction-specific transformations!")
                        self._warned_about_encoding = True
                    pooled_prompt_embeds = torch.zeros((bsz, 2048), dtype=self.model.dtype, device=noisy_latents.device)
                    prompt_embeds_t5 = torch.zeros((bsz, 256, 4096), dtype=self.model.dtype, device=noisy_latents.device)
                    prompt_embeds_llama3 = torch.zeros((32, bsz, 256, 4096), dtype=self.model.dtype, device=noisy_latents.device)
            else:
                # No pipeline available, use zeros (only show warning once)
                if not hasattr(self, '_warned_about_no_pipeline'):
                    print("\n❌ No pipeline available!")
                    print("  ⚠️  Using zeros for text conditioning...")
                    print("  ⚠️  WARNING: Model will NOT learn instruction-specific transformations!")
                    self._warned_about_no_pipeline = True
                pooled_prompt_embeds = torch.zeros((bsz, 2048), dtype=self.model.dtype, device=noisy_latents.device)
                prompt_embeds_t5 = torch.zeros((bsz, 256, 4096), dtype=self.model.dtype, device=noisy_latents.device)
                prompt_embeds_llama3 = torch.zeros((32, bsz, 256, 4096), dtype=self.model.dtype, device=noisy_latents.device)
        
        # Forward through transformer (with LoRA)
        model_pred = self.model.transformer(
            hidden_states=noisy_latents,  # 16 channels
            timesteps=timesteps.float() / self.config.diffusion.num_train_timesteps,  # Normalize to [0,1]
            pooled_embeds=pooled_prompt_embeds,
            encoder_hidden_states_t5=prompt_embeds_t5,
            encoder_hidden_states_llama3=prompt_embeds_llama3,
            return_dict=False,
        )[0]
        
        # Unpatchify the model output if needed
        # Model output is in patchified format: (B, C, num_patches, patch_dim)
        # Need to reshape to (B, num_patches, C * patch_dim) then unpatchify to (B, C, H, W)
        if model_pred.shape != target_latents.shape:
            # Reshape from (B, C, num_patches, patch_dim) to (B, num_patches, C * patch_dim)
            B, C, S, P = model_pred.shape
            model_pred = model_pred.permute(0, 2, 1, 3).reshape(B, S, C * P)
            # Prepare img_sizes as list of (patch_height, patch_width) tuples
            # Need to divide spatial dimensions by patch_size to get patch grid dimensions
            
            # Handle DDP: access .module if wrapped
            transformer = self.model.transformer.module if hasattr(self.model.transformer, 'module') else self.model.transformer
            patch_size = transformer.config.patch_size
            
            patch_h = target_latents.shape[2] // patch_size
            patch_w = target_latents.shape[3] // patch_size
            img_sizes = [(patch_h, patch_w)] * bsz
            model_pred = transformer.unpatchify(model_pred, img_sizes, is_training=False)
            # Convert back to model dtype (unpatchify returns float32)
            model_pred = model_pred.to(dtype=self.model.dtype)
        
        # Compute loss: predict the clean latents
        # Using weighted loss to focus on edited regions
        loss = compute_loss_with_config(
            model_pred,
            target_latents,
            source_latents,
            target_latents,
            timesteps,
            self.config
        )
        
        return loss
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation"""
        self.model.transformer.eval()
        self.logger.info("Running validation...")
        
        # For now, just return empty dict
        # TODO: Implement proper validation with image generation
        
        return {}
    
    def _log_training_step(self, loss: float, step: int):
        """Log training step metrics"""
        if self.accelerator.is_main_process:
            logs = {
                "train_loss": loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "epoch": self.epoch,
                "step": self.global_step,
            }
            self.accelerator.log(logs, step=self.global_step)
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch-level metrics"""
        if self.accelerator.is_main_process:
            all_metrics = {f"epoch_{k}": v for k, v in train_metrics.items()}
            all_metrics.update({f"epoch_{k}": v for k, v in val_metrics.items()})
            all_metrics['epoch'] = epoch
            
            self.accelerator.log(all_metrics, step=self.global_step)
            
            self.logger.info(f"Epoch {epoch + 1} metrics: {all_metrics}")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            output_dir = Path(self.config.training.output_dir) / f"checkpoint-{epoch}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA weights
            unwrapped_model = self.accelerator.unwrap_model(self.model.transformer)
            
            if hasattr(unwrapped_model, 'save_pretrained'):
                unwrapped_model.save_pretrained(output_dir)
            else:
                torch.save(
                    {k: v for k, v in unwrapped_model.state_dict().items() if 'lora' in k.lower()},
                    output_dir / "lora_weights.pt"
                )
            
            # Save training state
            # Convert OmegaConf to plain dict for JSON serialization
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            
            training_state = {
                'epoch': epoch,
                'global_step': self.global_step,
                'config': config_dict,
            }
            
            with open(output_dir / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2)
            
            self.logger.info(f"✅ Saved checkpoint to {output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load LoRA weights
        self.model.load_lora_weights(checkpoint_path)
        
        # Load training state
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.epoch = state.get('epoch', 0)
            self.global_step = state.get('global_step', 0)
            
            self.logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"   Resuming from epoch {self.epoch + 1}, step {self.global_step}")
        else:
            self.logger.warning(f"No training state found at {checkpoint_path}")

