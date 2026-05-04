"""
Planner Training Dataset - Cached Embeddings Version

Dataset for fine-tuning Qwen3-VL to predict action plans using pre-computed
vision embeddings instead of loading images on-the-fly.

This provides 3× speedup by avoiding redundant vision encoding.
"""

import json
import h5py
import torch
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset


class PlannerDatasetCached(Dataset):
    """
    Dataset for action planner training using cached vision embeddings.
    
    Each sample contains:
    - Pre-computed vision embeddings (loaded from HDF5)
    - User prompt text
    - Target action plan (ground truth)
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
        Initialize dataset with cached embeddings.
        
        Args:
            data_path: Path to planner_training_data.json
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
        
        print(f"Loaded {len(self.data)} training samples")
        
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
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training sample with cached embeddings."""
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
        # Note: We can't use the image in messages since we're using cached embeddings
        # Instead, we'll process text only and inject embeddings later
        target_json = json.dumps(target_action_plan, indent=2)
        
        # Create text-only messages (embeddings will be injected separately)
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
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize text only
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
        
        # Return with cached vision embeddings
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_embeddings": vision_embeddings,  # Pre-computed embeddings
            "has_cached_embeddings": True  # Flag for model to use cached embeddings
        }
        
        return result
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for action planning."""
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
- Reference SPECIFIC image elements (objects, composition, current style)
- Explain WHY this action creates theme/style transformation
- Mention dependencies (e.g., "location_setting must come first")
- Keep concise (1-2 sentences, ~20-40 words)
- Avoid generic statements ("to improve" or "to enhance")

STEP 3: GENERATE EDIT PROMPT (hidream_prompt)
- Create a concise instruction for the image editor
- **MAXIMUM 77 TOKENS** (STRICT limit due to CLIP tokenizer)
- **MUST start with "style_transformation_mode"**
- Focus on BOLD, noticeable theme/style changes

Examples of GOOD transformations:
✅ THEME: "Transform urban street to tropical beach"
✅ STYLE: "Apply watercolor painting style"
✅ THEME: "Make it a moonlit night scene with eerie atmosphere"

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
- "reasoning" MUST come FIRST in the JSON
- Each action MUST have its own "reasoning" field
- Focus on BOLD, NOTICEABLE transformations
- Think through the problem before selecting actions."""
        
        return system_prompt
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def create_dataloaders(
    data_path: str,
    processor,
    embeddings_path: str,
    batch_size: int = 2,
    train_val_split: float = 0.9,
    num_workers: int = 4,
    max_length: int = 2048,
    action_library_path: str = None
):
    """
    Create train and validation dataloaders with cached embeddings.
    
    Args:
        data_path: Path to training data JSON
        processor: Qwen3-VL processor
        embeddings_path: Path to HDF5 embeddings file
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
    full_dataset = PlannerDatasetCached(
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
    
    # Custom collate function to handle cached embeddings
    def collate_fn(batch):
        """Collate batch with cached vision embeddings."""
        # Stack all tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        # Stack vision embeddings
        vision_embeddings = torch.stack([item["vision_embeddings"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_embeddings": vision_embeddings,
            "has_cached_embeddings": True
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

