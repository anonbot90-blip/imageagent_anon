"""
DPO (Direct Preference Optimization) Trainer

Custom Trainer class that implements DPO loss for preference learning.
Based on the DPO paper: https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional


class DPOTrainer(Trainer):
    """
    Custom Trainer that implements DPO (Direct Preference Optimization) loss.
    
    DPO learns to prefer chosen responses over rejected responses by optimizing:
    L_DPO = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    Where:
    - y_w: chosen (winner) response
    - y_l: rejected (loser) response
    - π_θ: policy model being trained
    - π_ref: reference model (frozen)
    - β: temperature parameter (controls strength of KL penalty)
    - σ: sigmoid function
    """
    
    def __init__(
        self,
        model,
        ref_model,
        beta: float = 0.1,
        ref_model_device: str = "cuda",
        **kwargs
    ):
        """
        Initialize DPO Trainer.
        
        Args:
            model: Policy model to train
            ref_model: Reference model (frozen, for KL penalty)
            beta: Temperature parameter for DPO loss (default: 0.1)
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(model=model, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.ref_model_device = ref_model_device
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Place reference model on desired device (default: GPU)
        self.ref_model = self.ref_model.to(ref_model_device)
        
        print(f"DPO Trainer initialized with β={beta}")
        print(f"Reference model on: {ref_model_device}")
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to handle DPO-specific input format during evaluation.
        For evaluation, we only compute loss on the chosen responses.
        """
        # Extract chosen inputs for evaluation
        eval_inputs = {
            "input_ids": inputs["input_ids_chosen"],
            "attention_mask": inputs["attention_mask_chosen"],
        }
        
        # NOTE: Do NOT add pixel_values for cached embeddings training
        # The model uses pre-computed embeddings, not raw pixel data
        # If pixel_values are passed, the model will try to process them through
        # the vision encoder, causing shape mismatch errors
        
        # Add labels (same as input_ids for language modeling)
        eval_inputs["labels"] = inputs["input_ids_chosen"].clone()
        
        # Call parent's prediction_step with standard inputs
        return super().prediction_step(model, eval_inputs, prediction_loss_only, ignore_keys)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute DPO loss for training, or standard loss for evaluation.
        
        MEMORY EFFICIENT: Reference model is temporarily moved to GPU, then back to CPU.
        
        Args:
            model: The policy model
            inputs: Dict with chosen and rejected inputs (training) or standard inputs (evaluation)
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        
        Returns:
            loss (and optionally outputs)
        """
        # Check if this is evaluation (standard format) or training (DPO format)
        if "input_ids_chosen" not in inputs:
            # Evaluation mode: use standard loss computation
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Training mode: compute DPO loss
        # Extract chosen and rejected inputs
        input_ids_chosen = inputs["input_ids_chosen"]
        attention_mask_chosen = inputs["attention_mask_chosen"]
        input_ids_rejected = inputs["input_ids_rejected"]
        attention_mask_rejected = inputs["attention_mask_rejected"]
        
        # Extract pixel_values if present (for vision models)
        # NOTE: For cached embeddings training, we should NOT pass raw pixel_values
        # The model should use cached embeddings instead
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        
        # Prepare inputs for chosen
        chosen_inputs = {
            "input_ids": input_ids_chosen,
            "attention_mask": attention_mask_chosen,
        }
        # Only pass pixel_values if NOT using cached embeddings
        # For cached embeddings, pixel_values should be removed by the dataset
        # but we explicitly skip it here as a safety measure
        if pixel_values is not None and "inputs_embeds" not in inputs:
            # This branch should NOT be taken for cached embeddings training
            pass  # Do not add pixel_values for cached training
        
        # Prepare inputs for rejected
        rejected_inputs = {
            "input_ids": input_ids_rejected,
            "attention_mask": attention_mask_rejected,
        }
        # Same as above - do not pass pixel_values for cached training
        if pixel_values is not None and "inputs_embeds" not in inputs:
            pass  # Do not add pixel_values for cached training
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Forward pass for chosen (policy model)
        chosen_outputs = model(**chosen_inputs)
        chosen_logits = chosen_outputs.logits
        
        # Forward pass for rejected (policy model)
        rejected_outputs = model(**rejected_inputs)
        rejected_logits = rejected_outputs.logits
        
        # Forward pass for reference model (both chosen and rejected)
        with torch.no_grad():
            # Move inputs if reference model is on a different device
            if self.ref_model_device != device:
                chosen_inputs_ref = {k: v.to(self.ref_model_device) for k, v in chosen_inputs.items()}
                rejected_inputs_ref = {k: v.to(self.ref_model_device) for k, v in rejected_inputs.items()}
            else:
                chosen_inputs_ref = chosen_inputs
                rejected_inputs_ref = rejected_inputs
            
            ref_chosen_outputs = self.ref_model(**chosen_inputs_ref)
            ref_chosen_logits = ref_chosen_outputs.logits.to(device)
            
            ref_rejected_outputs = self.ref_model(**rejected_inputs_ref)
            ref_rejected_logits = ref_rejected_outputs.logits.to(device)
        
        # Compute log probabilities
        chosen_log_probs = self._get_batch_log_probs(
            chosen_logits, input_ids_chosen, attention_mask_chosen
        )
        rejected_log_probs = self._get_batch_log_probs(
            rejected_logits, input_ids_rejected, attention_mask_rejected
        )
        ref_chosen_log_probs = self._get_batch_log_probs(
            ref_chosen_logits, input_ids_chosen, attention_mask_chosen
        )
        ref_rejected_log_probs = self._get_batch_log_probs(
            ref_rejected_logits, input_ids_rejected, attention_mask_rejected
        )
        
        # Compute DPO loss
        # π_θ(y_w|x) / π_ref(y_w|x) vs π_θ(y_l|x) / π_ref(y_l|x)
        policy_chosen_logratios = chosen_log_probs - ref_chosen_log_probs
        policy_rejected_logratios = rejected_log_probs - ref_rejected_log_probs
        
        # DPO loss: -log(σ(β * (log_ratio_chosen - log_ratio_rejected)))
        logits_diff = policy_chosen_logratios - policy_rejected_logratios
        loss = -F.logsigmoid(self.beta * logits_diff).mean()
        
        # Compute accuracy (how often chosen is preferred)
        with torch.no_grad():
            accuracy = (logits_diff > 0).float().mean()
        
        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "dpo_loss": loss.item(),
                "dpo_accuracy": accuracy.item(),
                "chosen_log_prob": chosen_log_probs.mean().item(),
                "rejected_log_prob": rejected_log_probs.mean().item(),
            })
        
        return (loss, chosen_outputs) if return_outputs else loss
    
    def _get_batch_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch.
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            log_probs: Average log probability per sample (batch_size,)
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_attention_mask = attention_mask[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual labels
        per_token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)
        
        # Mask out padding tokens
        per_token_log_probs = per_token_log_probs * shift_attention_mask
        
        # Average over sequence length (only non-padding tokens)
        sequence_lengths = shift_attention_mask.sum(dim=1)
        avg_log_probs = per_token_log_probs.sum(dim=1) / (sequence_lengths + 1e-8)
        
        return avg_log_probs


