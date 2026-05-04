#!/usr/bin/env python3
"""
Data preparation and validation utilities for HiDream-E1 finetuning
"""

import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import random_split
from .dataset import ImageEditingDataset


def validate_dataset(data_dirs: List[str], verbose: bool = True) -> Dict:
    """Validate dataset directories and files
    
    Args:
        data_dirs: List of data directories to validate
        verbose: Print detailed validation results
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_directories': len(data_dirs),
        'valid_directories': 0,
        'invalid_directories': [],
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': [],
        'missing_files': [],
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATING DATASET")
        print(f"{'='*60}\n")
    
    for data_dir_str in data_dirs:
        data_dir = Path(data_dir_str)
        
        if not data_dir.exists():
            stats['invalid_directories'].append(str(data_dir))
            if verbose:
                print(f"❌ Directory not found: {data_dir}")
            continue
        
        stats['valid_directories'] += 1
        
        # Check for image_* subdirectories
        image_dirs = sorted(data_dir.glob("image_*"))
        stats['total_samples'] += len(image_dirs)
        
        if verbose:
            print(f"📁 {data_dir.name}: Found {len(image_dirs)} samples")
        
        for image_dir in image_dirs:
            # Check required files
            required_files = {
                'original.png': image_dir / "original.png",
                'edited.png': image_dir / "edited.png",
                'prompt.json': image_dir / "prompt.json",
            }
            
            missing = []
            for file_name, file_path in required_files.items():
                if not file_path.exists():
                    missing.append(file_name)
            
            if missing:
                stats['invalid_samples'].append(image_dir.name)
                stats['missing_files'].append({
                    'sample': image_dir.name,
                    'missing': missing
                })
                if verbose:
                    print(f"  ⚠️  {image_dir.name}: Missing {', '.join(missing)}")
            else:
                stats['valid_samples'] += 1
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total directories: {stats['total_directories']}")
        print(f"Valid directories: {stats['valid_directories']}")
        print(f"Total samples found: {stats['total_samples']}")
        print(f"Valid samples: {stats['valid_samples']}")
        print(f"Invalid samples: {len(stats['invalid_samples'])}")
        
        if stats['invalid_samples']:
            print(f"\n⚠️  Warning: {len(stats['invalid_samples'])} samples have missing files")
        
        print(f"{'='*60}\n")
    
    return stats


def create_train_val_split(
    dataset: ImageEditingDataset,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[ImageEditingDataset, ImageEditingDataset]:
    """Split dataset into train and validation sets
    
    Args:
        dataset: Complete dataset
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_dataset, val_dataset)
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # Random split
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.manual_seed(seed) if seed is not None else None
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"  Total samples: {total_size}")
    print(f"  Train samples: {train_size} ({train_ratio*100:.1f}%)")
    print(f"  Val samples: {val_size} ({val_ratio*100:.1f}%)")
    print()
    
    return train_dataset, val_dataset


def analyze_dataset_statistics(dataset: ImageEditingDataset) -> Dict:
    """Generate detailed dataset statistics
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary with statistics
    """
    stats = dataset.get_statistics()
    
    # Calculate instruction statistics
    instruction_lengths = [inst['instruction_length'] for inst in stats['instructions']]
    
    stats['instruction_stats'] = {
        'min_length': min(instruction_lengths) if instruction_lengths else 0,
        'max_length': max(instruction_lengths) if instruction_lengths else 0,
        'mean_length': sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
    }
    
    return stats


def prepare_datasets(config, verbose: bool = True):
    """Prepare train and validation datasets from config
    
    Args:
        config: Training configuration object
        verbose: Print preparation details
        
    Returns:
        (train_dataset, val_dataset)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PREPARING DATASETS")
        print(f"{'='*60}\n")
    
    # Validate data directories
    data_dirs = config.data.data_dirs
    validation_stats = validate_dataset(data_dirs, verbose=verbose)
    
    if validation_stats['valid_samples'] == 0:
        raise ValueError("No valid samples found in data directories!")
    
    # Create full dataset
    full_dataset = ImageEditingDataset(
        data_dirs=data_dirs,
        resolution=config.data.resolution,
        center_crop=config.data.center_crop,
        random_flip=config.data.random_flip,
        return_paths=True,
    )
    
    # Split into train/val
    import torch
    train_dataset, val_dataset = create_train_val_split(
        full_dataset,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        seed=config.seed
    )
    
    # Analyze statistics
    if verbose:
        stats = analyze_dataset_statistics(full_dataset)
        print(f"📈 Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Resolution: {stats['resolution']}x{stats['resolution']}")
        print(f"  Instruction length (min/mean/max): "
              f"{stats['instruction_stats']['min_length']}/"
              f"{stats['instruction_stats']['mean_length']:.0f}/"
              f"{stats['instruction_stats']['max_length']}")
        print(f"\n{'='*60}\n")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test data preparation
    import sys
    import torch
    sys.path.append("../..")
    
    from omegaconf import OmegaConf
    
    # Load config
    config = OmegaConf.load("../config/training_config.yaml")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config, verbose=True)
    
    print(f"✅ Data preparation test complete!")
    print(f"   Train size: {len(train_dataset)}")
    print(f"   Val size: {len(val_dataset)}")

