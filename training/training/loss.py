#!/usr/bin/env python3
"""
Loss functions for HiDream-E1 finetuning

Implements flow matching losses with optional spatial weighting for image editing.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def flow_matching_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> torch.Tensor:
    """Standard flow matching loss (MSE between model output and target)
    
    Args:
        model_output: Predicted flow from model [B, C, H, W]
        target: Target flow [B, C, H, W]
        timesteps: Optional timesteps [B] (for timestep-dependent weighting)
        reduction: Loss reduction method ('mean', 'sum', 'none')
        
    Returns:
        Loss value
    """
    # Compute MSE loss
    loss = F.mse_loss(model_output, target, reduction='none')
    
    # Optional: weight by timestep (noise level)
    if timesteps is not None:
        # Normalize timesteps to [0, 1]
        t_normalized = timesteps.float() / 1000.0  # Assuming 1000 timesteps
        # Reshape for broadcasting [B, 1, 1, 1]
        weight = t_normalized.view(-1, 1, 1, 1)
        loss = loss * weight
    
    # Reduce loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_diff_mask(
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    threshold: float = 0.1,
    blur_kernel_size: int = 3,
    blur_sigma: float = 1.0
) -> torch.Tensor:
    """Compute spatial difference mask between source and target latents
    
    This mask highlights regions that change between source and target,
    allowing the loss to focus on edited regions.
    
    Args:
        source_latents: Source image latents [B, C, H, W]
        target_latents: Target image latents [B, C, H, W]
        threshold: Threshold for considering a pixel as "changed"
        blur_kernel_size: Kernel size for gaussian blur (smoothing)
        blur_sigma: Sigma for gaussian blur
        
    Returns:
        Weight mask [B, 1, H, W] with values in [0, 1]
    """
    # Compute absolute difference
    diff = torch.abs(target_latents - source_latents)
    
    # Average across channels
    diff = diff.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Normalize to [0, 1]
    diff_min = diff.amin(dim=(-2, -1), keepdim=True)
    diff_max = diff.amax(dim=(-2, -1), keepdim=True)
    diff_normalized = (diff - diff_min) / (diff_max - diff_min + 1e-8)
    
    # Apply threshold
    mask = (diff_normalized > threshold).float()
    
    # Smooth with gaussian blur
    if blur_kernel_size > 0:
        # Simple box blur (can replace with gaussian if needed)
        kernel = torch.ones(1, 1, blur_kernel_size, blur_kernel_size).to(mask.device)
        kernel = kernel / (blur_kernel_size ** 2)
        mask = F.conv2d(mask, kernel, padding=blur_kernel_size // 2)
        mask = torch.clamp(mask, 0, 1)
    
    # Ensure mask is not all zeros (fallback to uniform weight)
    mask_sum = mask.sum(dim=(-2, -1), keepdim=True)
    mask = torch.where(mask_sum > 0, mask, torch.ones_like(mask))
    
    # Normalize mask to have mean of 1.0 (so it doesn't change loss scale)
    mask = mask / (mask.mean(dim=(-2, -1), keepdim=True) + 1e-8)
    
    return mask


def weighted_flow_matching_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    weight_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> torch.Tensor:
    """Flow matching loss with spatial weighting
    
    Applies higher weight to regions that change between source and target.
    This encourages the model to focus on learning the editing transformation.
    
    Args:
        model_output: Predicted flow from model [B, C, H, W]
        target: Target flow [B, C, H, W]
        weight_mask: Spatial weight mask [B, 1, H, W]
        timesteps: Optional timesteps [B]
        reduction: Loss reduction method
        
    Returns:
        Weighted loss value
    """
    # Compute base loss
    loss = F.mse_loss(model_output, target, reduction='none')
    
    # Apply spatial weight mask
    loss = loss * weight_mask
    
    # Optional: weight by timestep
    if timesteps is not None:
        t_normalized = timesteps.float() / 1000.0
        t_weight = t_normalized.view(-1, 1, 1, 1)
        loss = loss * t_weight
    
    # Reduce loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_loss_with_config(
    model_output: torch.Tensor,
    target: torch.Tensor,
    source_latents: Optional[torch.Tensor] = None,
    target_latents: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    config = None,
) -> torch.Tensor:
    """Compute loss according to training configuration
    
    Args:
        model_output: Model prediction
        target: Target
        source_latents: Source latents (for weighted loss)
        target_latents: Target latents (for weighted loss)
        timesteps: Timesteps
        config: Training config with loss settings
        
    Returns:
        Loss value
    """
    if config is None or not config.loss.use_weighted_loss:
        # Standard flow matching loss
        return flow_matching_loss(model_output, target, timesteps)
    else:
        # Weighted loss (focus on changed regions)
        if source_latents is None or target_latents is None:
            raise ValueError("source_latents and target_latents required for weighted loss")
        
        # Compute difference mask
        weight_mask = compute_diff_mask(
            source_latents,
            target_latents,
            threshold=config.loss.weight_threshold
        )
        
        return weighted_flow_matching_loss(
            model_output,
            target,
            weight_mask,
            timesteps
        )


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 2
    channels = 16
    height, width = 96, 96
    
    model_output = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    source_latents = torch.randn(batch_size, channels, height, width)
    target_latents = source_latents + torch.randn_like(source_latents) * 0.5
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Test standard loss
    loss1 = flow_matching_loss(model_output, target, timesteps)
    print(f"Standard loss: {loss1.item():.4f}")
    
    # Test weighted loss
    weight_mask = compute_diff_mask(source_latents, target_latents)
    print(f"Weight mask shape: {weight_mask.shape}, mean: {weight_mask.mean():.4f}")
    
    loss2 = weighted_flow_matching_loss(model_output, target, weight_mask, timesteps)
    print(f"Weighted loss: {loss2.item():.4f}")
    
    print("\n✅ Loss functions test complete!")

