"""
HiDream-E1 LoRA Finetuning Pipeline

A complete training pipeline for finetuning HiDream-E1 on custom image editing datasets.
"""

__version__ = "1.0.0"

from .data import ImageEditingDataset, prepare_datasets
from .models import HiDreamE1LoRA, load_model_with_lora
from .training import HiDreamE1Trainer
from .evaluation import MetricsCalculator
from .utils import set_seed, save_config, load_config

__all__ = [
    'ImageEditingDataset',
    'prepare_datasets',
    'HiDreamE1LoRA',
    'load_model_with_lora',
    'HiDreamE1Trainer',
    'MetricsCalculator',
    'set_seed',
    'save_config',
    'load_config',
]

