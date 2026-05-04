"""
Weighted Loss Trainer

Custom Trainer class that implements TRUE sample-weighted loss for reward-weighted training.
Computes per-sample losses and applies individual weights correctly.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies sample weights to the loss.
    
    Each sample has a weight based on its reward score:
    - High-quality samples (score >= 4.5): weight = 2.0
    - Good samples (score >= 4.0): weight = 1.5  
    - Medium samples (score >= 3.5): weight = 1.0
    - Lower samples (score >= 3.0): weight = 0.5
    
    This implementation computes per-sample losses and applies weights individually:
    weighted_loss = sum(loss_i * weight_i) / sum(weight_i)
    
    This ensures high-quality samples have proportionally more influence on gradients.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Weighted Loss Trainer."""
        super().__init__(*args, **kwargs)
        print("WeightedLossTrainer initialized - using TRUE per-sample weighted loss")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss with per-sample weighting.
        
        Args:
            model: The model
            inputs: Dict with input_ids, attention_mask, labels, and sample_weight
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        
        Returns:
            loss (and optionally outputs)
        """
        # Extract sample weights and labels
        sample_weights = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")
        
        # Remove pixel_values if present (for cached embeddings training)
        # The model should not process raw images when using cached embeddings
        inputs.pop("pixel_values", None)
        inputs.pop("image_grid_thw", None)
        
        # Forward pass - get logits
        outputs = model(**inputs)
        logits = outputs.logits
        
        # If no weights provided, use standard loss
        if sample_weights is None:
            # Compute standard loss
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        
        # Ensure weights are on the same device
        sample_weights = sample_weights.to(logits.device)
        
        # Compute per-sample losses
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous()      # (batch_size, seq_len-1)
        
        # Compute loss per token (no reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            shift_labels.view(-1)                           # (batch_size * seq_len)
        )
        
        # Reshape to (batch_size, seq_len-1)
        per_token_loss = per_token_loss.view(shift_labels.size())
        
        # Create mask for valid tokens (ignore padding, typically -100)
        valid_token_mask = (shift_labels != -100).float()
        
        # Compute per-sample loss (average over valid tokens in each sample)
        per_sample_loss = (per_token_loss * valid_token_mask).sum(dim=1) / (valid_token_mask.sum(dim=1) + 1e-8)
        
        # Apply sample weights: weighted_loss = sum(loss_i * weight_i) / sum(weight_i)
        weighted_loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
