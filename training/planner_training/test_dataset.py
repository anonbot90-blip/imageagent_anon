"""
Test dataset to debug tensor shapes
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor

from training.planner_training.planner_dataset import PlannerDataset, PlannerDataCollator

# Load config
config_path = SCRIPT_DIR / "planner_config.yaml"
config = OmegaConf.load(config_path)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    config.model.base_model,
    trust_remote_code=True
)

print("Creating dataset...")
dataset = PlannerDataset(
    data_path=config.data.training_data_path,
    processor=processor,
    max_length=config.training.max_length
)

print(f"Dataset size: {len(dataset)}")

print("\n=== Testing single sample ===")
sample = dataset[0]
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"attention_mask shape: {sample['attention_mask'].shape}")
print(f"labels shape: {sample['labels'].shape}")

if 'pixel_values' in sample:
    print(f"pixel_values shape: {sample['pixel_values'].shape}")
if 'image_grid_thw' in sample:
    print(f"image_grid_thw shape: {sample['image_grid_thw'].shape}")

print("\n=== Testing collator with batch of 4 ===")
collator = PlannerDataCollator(processor)
batch = collator([dataset[i] for i in range(4)])

print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")

if 'pixel_values' in batch:
    print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
if 'image_grid_thw' in batch:
    print(f"Batch image_grid_thw shape: {batch['image_grid_thw'].shape}")

print("\n✅ Dataset test complete!")



