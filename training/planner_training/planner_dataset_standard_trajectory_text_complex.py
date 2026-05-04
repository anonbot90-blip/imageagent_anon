"""
Planner Training Dataset - Standard Trajectory-Based (Text-Only)

Dataset for fine-tuning Qwen3-VL using trajectory-based train/test split.
Uses text-only inputs for faster training.

Trajectory-based: Train/test split at (image, style) level to prevent leakage.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset


class PlannerDatasetStandardTrajectoryTextComplex(Dataset):
    """
    Dataset for action planner training using text-only inputs.
    
    Each sample contains:
    - User prompt text (no image)
    - Target action plan (ground truth)
    """
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048,
        action_library_path: str = None
    ):
        """
        Initialize text-only dataset.
        
        Args:
            data_path: Path to planner_training_data.json
            processor: Qwen3-VL processor (used for text tokenization only)
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
        
        print(f"Loaded {len(self.data)} training samples (standard trajectory-based, text-only mode)")
        
        # Load action library
        if action_library_path is None:
            action_library_path = Path(__file__).parent.parent.parent / "actions" / "action_library_v2.json"
        
        with open(action_library_path, 'r') as f:
            self.action_library = json.load(f)
        
        print("✓ Text-only mode: Images will be ignored")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training sample (text-only)."""
        sample = self.data[idx]
        
        # Get prompt and target
        user_prompt = sample["user_prompt"]
        target_action_plan = sample["target_action_plan"]
        
        # Load analysis if available (for better CoT reasoning)
        analysis = sample.get("analysis")
        analysis_text = ""
        if analysis:
            # Create concise analysis summary (same format as inference)
            analysis_parts = []
            
            if "scene_type" in analysis:
                analysis_parts.append(f"Scene: {analysis['scene_type']}")
            
            if "objects" in analysis and analysis["objects"]:
                objects_list = [obj["name"] if isinstance(obj, dict) else obj for obj in analysis["objects"][:10]]
                analysis_parts.append(f"Objects: {', '.join(objects_list)}")
            
            if "colors" in analysis and analysis["colors"]:
                colors_list = [c["name"] if isinstance(c, dict) else c for c in analysis["colors"][:5]]
                analysis_parts.append(f"Colors: {', '.join(colors_list)}")
            
            if "style_attributes" in analysis:
                style_attrs = analysis["style_attributes"]
                style_parts = []
                if "artistic_style" in style_attrs:
                    style_parts.append(style_attrs["artistic_style"])
                if "mood" in style_attrs:
                    style_parts.append(style_attrs["mood"])
                if style_parts:
                    analysis_parts.append(f"Style: {', '.join(style_parts)}")
            
            if "composition" in analysis:
                comp = analysis["composition"]
                comp_parts = []
                if "layout" in comp:
                    comp_parts.append(f"layout: {comp['layout']}")
                if "focal_point" in comp:
                    comp_parts.append(f"focus: {comp['focal_point']}")
                if comp_parts:
                    analysis_parts.append(f"Composition: {', '.join(comp_parts)}")
            
            if analysis_parts:
                analysis_text = "\n\nIMAGE ANALYSIS:\n" + "\n".join(f"- {part}" for part in analysis_parts)
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create conversation format for training (text-only with analysis)
        target_json = json.dumps(target_action_plan, indent=2)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\nUser Request: {user_prompt}{analysis_text}"}
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
        
        # Tokenize text only (no images)
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
        
        # Return text-only inputs
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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
- Review the IMAGE ANALYSIS provided - this tells you what's in the image
- What is the current THEME (setting, mood, atmosphere, time)?
- What is the current STYLE (photorealistic, painting, artistic)?
- What THEME or STYLE does the user want?
- What actions would create a BOLD, NOTICEABLE transformation?
- Prioritize dramatic changes over subtle adjustments

STEP 2: PLAN ACTIONS WITH REASONING
- Select 2-5 appropriate actions from the library
- Focus on THEME transformations (scene/setting) and/or STYLE transformations (artistic rendering)
- Each action must contribute to a NOTICEABLE change
- For EACH action, provide "reasoning" explaining WHY it's needed
- Set clear priorities (1=highest, 5=lowest)

PER-ACTION REASONING REQUIREMENTS:
- Reference SPECIFIC image elements (objects, composition, current style from analysis)
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

Output Format (JSON):
{{
  "reasoning": "First, I observe [current state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
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
- Think through the problem before selecting actions."""
        
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
    Create train and validation dataloaders (text-only).
    
    Args:
        data_path: Path to training data JSON
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
    full_dataset = PlannerDatasetTextOnly(
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Standard collate function (no special handling needed for text-only)
    def collate_fn(batch):
        """Collate batch (text-only)."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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

