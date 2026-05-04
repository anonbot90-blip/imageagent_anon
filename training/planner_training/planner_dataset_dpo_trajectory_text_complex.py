"""
Planner Training Dataset - DPO (Direct Preference Optimization) Text-Only Version

Dataset for DPO training using preference pairs (chosen, rejected) for the same prompt.
Learns to prefer high-quality plans over low-quality plans.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset


class PlannerDatasetDPOTrajectoryTextComplex(Dataset):
    """
    DPO dataset for action planner training (text-only).
    
    Each sample contains:
    - User prompt text (no image)
    - Chosen action plan (high quality)
    - Rejected action plan (low quality)
    - Score margin (chosen_score - rejected_score)
    """
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048,
        action_library_path: str = None
    ):
        """
        Initialize DPO text-only dataset.
        
        Args:
            data_path: Path to planner_training_data_dpo.json (with preference pairs)
            processor: Qwen3-VL processor
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
        
        print(f"Loaded {len(self.data)} DPO preference pairs (text-only mode)")
        
        # Load action library
        if action_library_path is None:
            action_library_path = Path(__file__).parent.parent.parent / "actions" / "action_library_v2.json"
        
        with open(action_library_path, 'r') as f:
            self.action_library = json.load(f)
        
        # Print statistics
        margins = [s["score_margin"] for s in self.data]
        print(f"  Score margins: {min(margins):.2f} - {max(margins):.2f} (avg: {sum(margins)/len(margins):.2f})")
        print("✓ DPO mode: Learning from preference pairs")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get DPO training sample (chosen + rejected)."""
        sample = self.data[idx]
        
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
        
        # Apply chat template for both
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
        
        # Tokenize both (text only, no images)
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
        
        # Return DPO format (both chosen and rejected)
        result = {
            "input_ids_chosen": chosen_input_ids,
            "attention_mask_chosen": chosen_attention_mask,
            "input_ids_rejected": rejected_input_ids,
            "attention_mask_rejected": rejected_attention_mask,
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
- What is the current state based on the description?
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
  "reasoning": "First, I observe [current state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
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


def create_dataloaders(
    data_path: str,
    processor,
    batch_size: int = 4,
    train_val_split: float = 0.9,
    num_workers: int = 4,
    max_length: int = 2048,
    action_library_path: str = None
):
    """
    Create train and validation dataloaders (DPO text-only).
    
    Args:
        data_path: Path to DPO training data JSON
        processor: Qwen3-VL processor
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
    full_dataset = PlannerDatasetDPOText(
        data_path=data_path,
        processor=processor,
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
    
    # Collate function for DPO pairs
    def collate_fn(batch):
        """Collate batch of DPO pairs."""
        input_ids_chosen = torch.stack([item["input_ids_chosen"] for item in batch])
        attention_mask_chosen = torch.stack([item["attention_mask_chosen"] for item in batch])
        input_ids_rejected = torch.stack([item["input_ids_rejected"] for item in batch])
        attention_mask_rejected = torch.stack([item["attention_mask_rejected"] for item in batch])
        
        return {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
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

