#!/usr/bin/env python3
"""
General utility functions for training
"""

import random
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
import json


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    print(f"✅ Set random seed to {seed}")


def save_config(config, output_path: str):
    """Save configuration to file
    
    Args:
        config: Configuration object (OmegaConf or dict)
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, '__dict__'):
        config_dict = dict(config)
    else:
        config_dict = config
    
    # Save as YAML
    if output_path.suffix == '.yaml' or output_path.suffix == '.yml':
        OmegaConf.save(config, output_path)
    else:
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    print(f"✅ Saved config to {output_path}")


def load_config(config_path: str):
    """Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    if config_path.suffix in ['.yaml', '.yml']:
        config = OmegaConf.load(config_path)
    else:
        # Load JSON
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = OmegaConf.create(config_dict)
    
    print(f"✅ Loaded config from {config_path}")
    return config


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_parameters(model) -> dict:
    """Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_percent': 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_parameter_summary(model, name="Model"):
    """Print model parameter summary
    
    Args:
        model: PyTorch model
        name: Model name for display
    """
    params = count_parameters(model)
    
    print(f"\n{name} Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_percent']:.2f}%)")
    print(f"  Frozen: {params['frozen']:,}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test set_seed
    set_seed(42)
    
    # Test config save/load
    test_config = OmegaConf.create({
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
        },
        'model': {
            'name': 'test_model'
        }
    })
    
    save_config(test_config, '/tmp/test_config.yaml')
    loaded_config = load_config('/tmp/test_config.yaml')
    
    print(f"\nLoaded config: {loaded_config}")
    
    # Test time formatting
    print(f"\nTime formatting:")
    print(f"  30s -> {format_time(30)}")
    print(f"  150s -> {format_time(150)}")
    print(f"  7200s -> {format_time(7200)}")
    
    print("\n✅ Utils test complete!")

