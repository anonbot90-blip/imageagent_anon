#!/usr/bin/env python3
"""
Image Editing Dataset for HiDream-E1 Finetuning

Loads (source, target, instruction) triplets for instruction-based image editing training.
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Optional, Tuple


class ImageEditingDataset(Dataset):
    """Dataset for instruction-based image editing
    
    Expected directory structure:
        data_dir/
            image_1_name/
                original.png      # Source image
                edited.png        # Target image
                prompt.json       # Editing instruction
                analysis.json     # (optional) Image analysis
            image_2_name/
                ...
    """
    
    def __init__(
        self,
        data_dirs: List[str],
        resolution: int = 768,
        center_crop: bool = True,
        random_flip: bool = False,
        return_paths: bool = False,
    ):
        """
        Args:
            data_dirs: List of directories containing editing pairs
            resolution: Target resolution for images
            center_crop: Whether to center crop images
            random_flip: Whether to apply random horizontal flips
            return_paths: Whether to return file paths in output
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.return_paths = return_paths
        
        # Load all samples
        self.samples = self._load_samples()
        
        # Setup transforms
        self.transform = self._create_transforms()
        
        print(f"✅ Loaded {len(self.samples)} editing pairs from {len(self.data_dirs)} directories")
    
    def _load_samples(self) -> List[Dict]:
        """Scan directories and load all valid samples"""
        samples = []
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                print(f"⚠️  Warning: Directory does not exist: {data_dir}")
                continue
            
            # Find all image_* subdirectories
            image_dirs = sorted(data_dir.glob("image_*"))
            
            for image_dir in image_dirs:
                # Check required files
                original_path = image_dir / "original.png"
                edited_path = image_dir / "edited.png"
                prompt_path = image_dir / "prompt.json"
                
                if not all([original_path.exists(), edited_path.exists(), prompt_path.exists()]):
                    print(f"⚠️  Skipping {image_dir.name}: missing files")
                    continue
                
                # Load prompt data
                try:
                    with open(prompt_path, 'r') as f:
                        prompt_data = json.load(f)
                    
                    # Format instruction for HiDream-E1
                    instruction = self._format_instruction(prompt_data)
                    
                    samples.append({
                        'sample_id': image_dir.name,
                        'original_path': str(original_path),
                        'edited_path': str(edited_path),
                        'instruction': instruction,
                        'prompt_data': prompt_data,
                    })
                
                except Exception as e:
                    print(f"⚠️  Error loading {image_dir.name}: {e}")
                    continue
        
        return samples
    
    def _format_instruction(self, prompt_data: Dict) -> str:
        """Convert prompt.json to HiDream-E1 instruction format
        
        HiDream-E1 expects:
            "Editing Instruction: {instruction}. Target Image Description: {description}"
        """
        # Extract editing instruction
        if 'edit_info' in prompt_data and 'text' in prompt_data['edit_info']:
            edit_instruction = prompt_data['edit_info']['text']
        elif 'edit' in prompt_data and 'text' in prompt_data['edit']:
            edit_instruction = prompt_data['edit']['text']
        else:
            # Fallback: use main text as instruction
            edit_instruction = prompt_data.get('text', 'Transform the image')
        
        # Extract target description (optional, can be derived from instruction)
        if 'edit_info' in prompt_data and 'target_theme' in prompt_data['edit_info']:
            target_theme = prompt_data['edit_info']['target_theme']
            target_description = f"An image with {target_theme} theme"
        elif 'edit' in prompt_data and 'target_theme' in prompt_data['edit']:
            target_theme = prompt_data['edit']['target_theme']
            target_description = f"An image with {target_theme} theme"
        else:
            # Derive from instruction
            target_description = edit_instruction
        
        # Format in HiDream-E1 style
        formatted_instruction = (
            f"Editing Instruction: {edit_instruction}. "
            f"Target Image Description: {target_description}"
        )
        
        return formatted_instruction
    
    def _create_transforms(self):
        """Create image preprocessing transforms"""
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize(
            self.resolution, 
            interpolation=transforms.InterpolationMode.BILINEAR
        ))
        
        # Center crop
        if self.center_crop:
            transform_list.append(transforms.CenterCrop(self.resolution))
        
        # Random flip (disabled by default for editing tasks)
        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # To tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image"""
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Load images
        source_image = self._load_image(sample['original_path'])
        target_image = self._load_image(sample['edited_path'])
        
        # Apply same transform to both images
        # Note: Random transforms will use same seed for both
        seed = torch.randint(0, 2**32, (1,)).item()
        
        # Transform source
        torch.manual_seed(seed)
        source_tensor = self.transform(source_image)
        
        # Transform target with same seed
        torch.manual_seed(seed)
        target_tensor = self.transform(target_image)
        
        # Prepare output
        output = {
            'source_image': source_tensor,
            'target_image': target_tensor,
            'instruction': sample['instruction'],
            'sample_id': sample['sample_id'],
        }
        
        # Optionally include paths
        if self.return_paths:
            output['source_path'] = sample['original_path']
            output['target_path'] = sample['edited_path']
        
        return output
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading images"""
        return self.samples[idx]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.samples),
            'data_directories': len(self.data_dirs),
            'resolution': self.resolution,
            'instructions': [],
        }
        
        # Collect instruction info
        for sample in self.samples:
            stats['instructions'].append({
                'sample_id': sample['sample_id'],
                'instruction_length': len(sample['instruction']),
            })
        
        return stats


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """Custom collate function for batching
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with tensors and lists
    """
    # Stack image tensors
    source_images = torch.stack([item['source_image'] for item in batch])
    target_images = torch.stack([item['target_image'] for item in batch])
    
    # Keep instructions as list of strings
    instructions = [item['instruction'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    
    batched = {
        'source_image': source_images,
        'target_image': target_images,
        'instruction': instructions,
        'sample_id': sample_ids,
    }
    
    # Include paths if present
    if 'source_path' in batch[0]:
        batched['source_path'] = [item['source_path'] for item in batch]
        batched['target_path'] = [item['target_path'] for item in batch]
    
    return batched


if __name__ == "__main__":
    # Test dataset loading
    print("Testing ImageEditingDataset...")
    
    dataset = ImageEditingDataset(
        data_dirs=["./imageagent_results_training_300"],
        resolution=768,
        center_crop=True,
        random_flip=False,
        return_paths=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Test loading a sample
    print(f"\nTesting sample loading...")
    sample = dataset[0]
    print(f"Source image shape: {sample['source_image'].shape}")
    print(f"Target image shape: {sample['target_image'].shape}")
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Sample ID: {sample['sample_id']}")
    
    print("\n✅ Dataset test complete!")

