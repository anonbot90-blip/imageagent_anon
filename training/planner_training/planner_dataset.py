"""
Planner Training Dataset

Dataset for fine-tuning Qwen3-VL to predict action plans.
Format: (image, user_prompt) → action_plan.json
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from torch.utils.data import Dataset


class PlannerDataset(Dataset):
    """
    Dataset for action planner training.
    
    Each sample contains:
    - image: PIL Image
    - user_prompt: str
    - target_action_plan: Dict (ground truth)
    """
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 4096,
        action_library_path: str = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to planner_training_data.json
            processor: Qwen3-VL processor
            max_length: Maximum sequence length
            action_library_path: Path to action_library.json
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
            action_library_path = Path(__file__).parent.parent.parent / "actions" / "action_library_v2.json"  # V2: 10 atomic actions
        
        with open(action_library_path, 'r') as f:
            self.action_library = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training sample."""
        sample = self.data[idx]
        
        # Load image (support both "image_path" and "image" keys)
        image_path = sample.get("image_path", sample.get("image"))
        if not image_path:
            raise KeyError(f"Sample {idx} missing both 'image_path' and 'image' keys")
        image = Image.open(image_path).convert("RGB")
        
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
        
        # Create conversation format for training
        # Input: image + user_prompt + analysis (if available)
        # Output: target_action_plan (as JSON string)
        target_json = json.dumps(target_action_plan, indent=2)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
            add_generation_prompt=False  # We have the full conversation
        )
        
        # Tokenize - Don't add batch dimension here, collator will handle it
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Remove batch dimension (added by processor) - collator will re-add it properly
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Prepare labels (for causal LM training)
        labels = input_ids.clone()
        
        # Extract vision inputs without batch dimension
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Add vision inputs if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for Chain-of-Thought action planning."""
        actions_summary = []
        for action in self.action_library["actions"]:
            action_id = action["id"]
            description = action["description"]
            actions_summary.append(f"- {action_id}: {description}")
        
        actions_text = "\n".join(actions_summary)
        
        prompt = f"""You are an expert THEME & STYLE transformation planner using Chain-of-Thought reasoning.

Your specialty: Planning BOLD, NOTICEABLE transformations that change:
- THEME: The scene's setting, atmosphere, mood, or environmental context (9 actions)
- STYLE: The artistic rendering or visual medium (1 action)

Available Actions:
{actions_text}

Action Categories:
- THEME Actions (9): location_setting, architecture_style, time_period_era, time_of_day, season_cycle, weather_conditions, mood_lighting, color_grading, atmospheric_effects
- STYLE Actions (1): artistic_medium

Instructions:
STEP 1: ANALYZE & REASON (Theme/Style Focus)
- Review the IMAGE ANALYSIS (if available) - what's in the image
- What is the current THEME (setting, mood, atmosphere, time)?
- What is the current STYLE (photorealistic, painting, artistic)?
- What THEME or STYLE does the user want?
- What actions would create a BOLD, NOTICEABLE transformation?
- Prioritize dramatic changes over subtle adjustments

STEP 2: PLAN ACTIONS WITH REASONING
- Select 1-5 appropriate actions from the library
- Focus on THEME transformations (scene/setting) and/or STYLE transformations (artistic rendering)
- Each action must contribute to a NOTICEABLE change
- For EACH action, provide "reasoning" explaining WHY it's needed for THIS image
- Set clear priorities (1=highest, foundation changes first)

PER-ACTION REASONING REQUIREMENTS:
- Reference SPECIFIC image elements (objects, composition, current style)
- Explain WHY this action creates theme/style transformation
- Mention dependencies (e.g., "location_setting must come first to establish scene")
- Keep concise (1-2 sentences, ~20-40 words)
- Avoid generic statements ("to improve" or "to enhance")

STEP 3: GENERATE EDIT PROMPT (hidream_prompt)
- Create a concise instruction for the image editor
- **MAXIMUM 77 TOKENS** (STRICT limit due to CLIP tokenizer)
- **MUST start with "style_transformation_mode"**
- If changing STYLE: "style_transformation_mode Apply {artistic_style} with {characteristics}"
- If changing THEME only: "style_transformation_mode Transform to {theme}. Maintain photorealistic quality."
- Focus on BOLD, noticeable changes

Examples of GOOD theme/style transformations:
✅ THEME: "Transform urban street to tropical beach" (location + weather + mood)
✅ STYLE: "Apply watercolor painting style with soft edges" (artistic medium)
✅ THEME: "Make it a moonlit night scene with eerie atmosphere" (time + mood + lighting)

Examples of BAD (too subtle):
❌ "Adjust brightness slightly"
❌ "Tweak color saturation"

Respond ONLY with valid JSON in this EXACT format:

{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
  "overall_instruction": "Brief summary of the complete edit",
  "actions": [
    {{
      "action_id": "location_setting",
      "reasoning": "Specific reasoning why THIS action creates theme/style transformation for THIS image.",
      "priority": 1,
      "parameters": {{...}}
    }}
  ],
  "hidream_prompt": "style_transformation_mode [concise instruction, MAX 77 TOKENS]"
}}

CRITICAL: 
- "reasoning" MUST come FIRST in the JSON
- Each action MUST have its own "reasoning" field
- Focus on BOLD, NOTICEABLE transformations
- Think through the problem before selecting actions."""
        
        return prompt


class PlannerDataCollator:
    """Custom data collator for planner training."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch."""
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        
        # Add pixel_values if present - handle vision inputs carefully
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            pixel_values_list = [f["pixel_values"] for f in features]
            
            # Stack along batch dimension: [batch, num_patches, hidden_dim]
            batch["pixel_values"] = torch.stack(pixel_values_list, dim=0)
        
        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            grid_thw_list = [f["image_grid_thw"] for f in features]
            
            # Stack along batch dimension: [batch, 3]
            batch["image_grid_thw"] = torch.stack(grid_thw_list, dim=0)
        
        return batch


def create_dataloaders(config, processor):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader, random_split
    
    # Load full dataset
    dataset = PlannerDataset(
        data_path=config.data.training_data_path,
        processor=processor,
        max_length=config.training.max_length
    )
    
    # Split into train/val
    train_size = int(len(dataset) * config.data.train_val_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create data collator
    collator = PlannerDataCollator(processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader

