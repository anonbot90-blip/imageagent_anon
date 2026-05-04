"""
Planner Training Dataset - DPO (Direct Preference Optimization) Cached Embeddings Version

Dataset for DPO training using preference pairs with cached vision embeddings.
Learns to prefer high-quality plans over low-quality plans (3× faster with cached embeddings).
"""

import json
import h5py
import torch
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset


class PlannerDatasetDPOTrajectoryCachedComplex(Dataset):
    """
    DPO dataset for action planner training using cached vision embeddings.
    
    Each sample contains:
    - Pre-computed vision embeddings (loaded from HDF5)
    - User prompt text
    - Chosen action plan (high quality)
    - Rejected action plan (low quality)
    - Score margin (chosen_score - rejected_score)
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
        Initialize DPO dataset with cached embeddings.
        
        Args:
            data_path: Path to planner_training_data_dpo.json (with preference pairs)
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
        
        print(f"Loaded {len(self.data)} DPO preference pairs (cached embeddings mode)")
        
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
        
        # Print statistics
        margins = [s["score_margin"] for s in self.data]
        print(f"  Score margins: {min(margins):.2f} - {max(margins):.2f} (avg: {sum(margins)/len(margins):.2f})")
        print("✓ DPO mode: Learning from preference pairs with cached embeddings")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get DPO training sample with cached embeddings."""
        sample = self.data[idx]
        
        # Extract sample IDs for embeddings
        # DPO pairs use the chosen sample's image (both plans are for same image)
        chosen_id = sample["metadata"]["chosen_id"]
        
        # Load pre-computed vision embeddings from HDF5
        if chosen_id not in self.h5_file:
            raise KeyError(f"Embeddings not found for sample: {chosen_id}")
        
        vision_embeddings = torch.from_numpy(self.h5_file[chosen_id][:])
        
        # Get prompt and both plans
        user_prompt = sample["user_prompt"]
        chosen_plan = sample["chosen_plan"]
        rejected_plan = sample["rejected_plan"]
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create conversation format for CHOSEN (high quality)
        chosen_json = json.dumps(chosen_plan, indent=2)
        
        chosen_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\nUser Request: {user_prompt}"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": chosen_json}
                ]
            }
        ]
        
        # Create conversation format for REJECTED (low quality)
        rejected_json = json.dumps(rejected_plan, indent=2)
        
        rejected_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\nUser Request: {user_prompt}"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": rejected_json}
                ]
            }
        ]
        
        # Apply chat template for both (text only)
        chosen_text = self.processor.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        rejected_text = self.processor.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize both
        chosen_inputs = self.processor(
            text=[chosen_text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        rejected_inputs = self.processor(
            text=[rejected_text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Remove batch dimension
        chosen_input_ids = chosen_inputs["input_ids"].squeeze(0)
        chosen_attention_mask = chosen_inputs["attention_mask"].squeeze(0)
        
        rejected_input_ids = rejected_inputs["input_ids"].squeeze(0)
        rejected_attention_mask = rejected_inputs["attention_mask"].squeeze(0)
        
        # Return DPO format with cached embeddings
        result = {
            "input_ids_chosen": chosen_input_ids,
            "attention_mask_chosen": chosen_attention_mask,
            "input_ids_rejected": rejected_input_ids,
            "attention_mask_rejected": rejected_attention_mask,
            "pixel_values": vision_embeddings,  # Same embeddings for both (same image)
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

STEP 2: PLAN ACTIONS WITH REASONING
- Select 2-5 appropriate actions
- Focus on THEME/STYLE transformations
- Each action must contribute to a NOTICEABLE change
- For EACH action, provide "reasoning" explaining WHY it's needed
- Set clear priorities (1=highest, 5=lowest)

PER-ACTION REASONING REQUIREMENTS:
- Be SPECIFIC to the image
- Explain WHY this action creates theme/style transformation
- Keep concise (1-2 sentences)

Output Format (JSON):
{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
  "overall_instruction": "<high-level description>",
  "actions": [
    {{
      "action_id": "<action_id>",
      "reasoning": "Specific reasoning for THIS theme/style transformation.",
      "priority": <1-5>,
      "parameters": {{
        "<param>": "<value>",
        ...
      }}
    }}
  ],
  "hidream_prompt": "style_transformation_mode [MAX 77 TOKENS]"
}}

CRITICAL: Each action MUST have "reasoning" field."""
        
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
    Create train and validation dataloaders (DPO cached).
    
    Args:
        data_path: Path to DPO training data JSON
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
    full_dataset = PlannerDatasetDPOCached(
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
    
    print(f"Train pairs: {len(train_dataset)}")
    print(f"Val pairs: {len(val_dataset)}")
    
    # Collate function for DPO pairs with cached embeddings
    def collate_fn(batch):
        """Collate batch of DPO pairs with cached embeddings."""
        input_ids_chosen = torch.stack([item["input_ids_chosen"] for item in batch])
        attention_mask_chosen = torch.stack([item["attention_mask_chosen"] for item in batch])
        input_ids_rejected = torch.stack([item["input_ids_rejected"] for item in batch])
        attention_mask_rejected = torch.stack([item["attention_mask_rejected"] for item in batch])
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        return {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
            "pixel_values": pixel_values,
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

