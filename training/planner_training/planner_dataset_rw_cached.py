"""
Planner Training Dataset - Reward-Weighted Cached Embeddings Version

Dataset for fine-tuning Qwen3-VL using reward-weighted samples with cached embeddings.
Each sample has a weight based on its reward score, emphasizing high-quality examples.

Weight scheme:
- Score 4.5-5.0: weight = 2.0 (emphasize best)
- Score 4.0-4.5: weight = 1.5
- Score 3.5-4.0: weight = 1.0
- Score 3.0-3.5: weight = 0.5 (de-emphasize mediocre)
"""

import json
import h5py
import torch
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset


class PlannerDatasetRWCached(Dataset):
    """
    Reward-weighted dataset for action planner training using cached vision embeddings.
    
    Each sample contains:
    - Pre-computed vision embeddings (loaded from HDF5)
    - User prompt text
    - Target action plan (ground truth)
    - Sample weight (based on reward score)
    """
    
    def __init__(
        self,
        data_path: str,
        processor,
        embeddings_path: str,
        max_length: int = 2048,
        action_library_path: str = None
    ):
        """
        Initialize reward-weighted dataset with cached embeddings.
        
        Args:
            data_path: Path to planner_training_data.json (with reward_score field)
            processor: Qwen3-VL processor
            embeddings_path: Path to vision_embeddings.h5 file
            max_length: Maximum sequence length
            action_library_path: Path to action_library_v2.json
        """
        self.processor = processor
        self.max_length = max_length
        
        # Load training data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Handle both dict with "samples" key and direct list
        if isinstance(data, dict) and "samples" in data:
            self.data = data["samples"]
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError(f"Invalid data format in {data_path}")
        
        print(f"Loaded {len(self.data)} training samples (reward-weighted cached mode)")
        
        # Compute sample weights
        self._compute_weights()
        
        # Load action library
        if action_library_path is None:
            action_library_path = Path(__file__).parent.parent.parent / "actions" / "action_library_v2.json"
        
        with open(action_library_path, 'r') as f:
            self.action_library = json.load(f)
        
        # Open HDF5 file (keep it open for fast access)
        self.embeddings_path = embeddings_path
        self.h5_file = h5py.File(embeddings_path, 'r')
        
        # Load manifest
        manifest_path = Path(embeddings_path).parent / "embeddings_manifest.json"
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        print(f"Loaded embeddings from: {embeddings_path}")
        print(f"Available embeddings: {len(self.manifest)}")
        print("✓ Reward-weighted mode: Samples weighted by quality")
    
    def _compute_weights(self):
        """Compute sample weights based on reward scores."""
        weights = []
        
        for sample in self.data:
            score = sample.get("reward_score", 4.0)  # Default to 4.0 if missing
            
            # Weight scheme
            if score >= 4.5:
                weight = 2.0  # Emphasize best
            elif score >= 4.0:
                weight = 1.5
            elif score >= 3.5:
                weight = 1.0
            else:  # 3.0-3.5
                weight = 0.5  # De-emphasize mediocre
            
            weights.append(weight)
        
        self.weights = weights
        
        # Print weight distribution
        weight_counts = {0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0}
        for w in weights:
            weight_counts[w] += 1
        
        print(f"  Weight distribution:")
        print(f"    2.0 (score >= 4.5): {weight_counts[2.0]} samples")
        print(f"    1.5 (score >= 4.0): {weight_counts[1.5]} samples")
        print(f"    1.0 (score >= 3.5): {weight_counts[1.0]} samples")
        print(f"    0.5 (score >= 3.0): {weight_counts[0.5]} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training sample with cached embeddings and weight."""
        sample = self.data[idx]
        sample_id = sample["id"]
        
        # Load pre-computed vision embeddings from HDF5
        if sample_id not in self.h5_file:
            raise KeyError(f"Embeddings not found for sample: {sample_id}")
        
        vision_embeddings = torch.from_numpy(self.h5_file[sample_id][:])
        
        # Get prompt and target
        user_prompt = sample["user_prompt"]
        target_action_plan = sample["target_action_plan"]
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create conversation format for training
        target_json = json.dumps(target_action_plan, indent=2)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\nUser Request: {user_prompt}"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_json}
                ]
            }
        ]
        
        # Apply chat template (text only)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize text
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Remove batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Prepare labels
        labels = input_ids.clone()
        
        # Get sample weight
        weight = self.weights[idx]
        
        # Return inputs with cached embeddings and weight
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": vision_embeddings,  # Pre-computed embeddings
            "sample_weight": torch.tensor(weight, dtype=torch.float32)
        }
        
        return result
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for Chain-of-Thought action planning."""
        actions_summary = []
        for action in self.action_library['actions']:
            params = ', '.join(action['parameters'].keys())
            actions_summary.append(f"- {action['id']}: {action['description']} (params: {params})")
        
        actions_text = '\n'.join(actions_summary)
        
        system_prompt = f"""You are an expert THEME & STYLE transformation planner using Chain-of-Thought reasoning.

Your specialty: Planning BOLD, NOTICEABLE transformations that change:
- THEME: The scene's setting, atmosphere, mood, or environmental context (9 actions)
- STYLE: The artistic rendering or visual medium (1 action)

Available Actions (V2 - 10 atomic actions):
{actions_text}

Action Categories:
- THEME Actions (9): location_setting, architecture_style, time_period_era, time_of_day, season_cycle, weather_conditions, mood_lighting, color_grading, atmospheric_effects
- STYLE Actions (1): artistic_medium

Instructions:
STEP 1: ANALYZE & REASON (Theme/Style Focus)
- What do you see in the image? (key objects, scene, composition, style)
- What is the current THEME (setting, mood, atmosphere, time)?
- What is the current STYLE (photorealistic, painting, artistic)?
- What THEME or STYLE does the user want?
- What actions would create a BOLD, NOTICEABLE transformation?
- Prioritize dramatic changes over subtle adjustments

STEP 2: PLAN ACTIONS WITH REASONING
- Select 2-5 appropriate actions
- Focus on THEME transformations (scene/setting) and/or STYLE transformations (artistic rendering)
- Each action must contribute to a NOTICEABLE change
- For EACH action, provide "reasoning" explaining WHY it's needed
- Set clear priorities (1=highest, 5=lowest)

PER-ACTION REASONING REQUIREMENTS:
- Be SPECIFIC to the image
- Explain WHY this action creates theme/style transformation
- Keep concise (1-2 sentences, ~20-40 words)

Output Format (JSON):
{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
  "overall_instruction": "<high-level description>",
  "actions": [
    {{
      "action_id": "<action_id>",
      "reasoning": "Specific reasoning why THIS action creates theme/style transformation.",
      "priority": <1-5>,
      "parameters": {{
        "<param>": "<value>",
        ...
      }}
    }}
  ],
  "hidream_prompt": "style_transformation_mode [concise instruction, MAX 77 TOKENS]"
}}

CRITICAL: 
- "reasoning" MUST come FIRST
- Each action MUST have "reasoning"
- Focus on BOLD transformations
- Think before selecting actions."""
        
        return system_prompt
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def create_dataloaders(
    data_path: str,
    processor,
    embeddings_path: str,
    batch_size: int = 4,
    train_val_split: float = 0.9,
    num_workers: int = 4,
    max_length: int = 2048,
    action_library_path: str = None
):
    """
    Create train and validation dataloaders (reward-weighted cached).
    
    Args:
        data_path: Path to training data JSON
        processor: Qwen3-VL processor
        embeddings_path: Path to vision_embeddings.h5
        batch_size: Batch size per GPU
        train_val_split: Train/val split ratio
        num_workers: Number of data loading workers
        max_length: Maximum sequence length
        action_library_path: Path to action library
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = PlannerDatasetRWCached(
        data_path=data_path,
        processor=processor,
        embeddings_path=embeddings_path,
        max_length=max_length,
        action_library_path=action_library_path
    )
    
    # Split into train/val
    train_size = int(len(full_dataset) * train_val_split)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Collate function with sample weights
    def collate_fn(batch):
        """Collate batch with cached embeddings and sample weights."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        sample_weights = torch.stack([item["sample_weight"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "sample_weight": sample_weights
        }
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

