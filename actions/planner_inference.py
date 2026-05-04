"""
Planner Inference - Use Qwen3-VL to predict action plans

This module:
1. Takes user prompt + image
2. Uses Qwen3-VL (fine-tuned planner) to predict structured action plan
3. Returns action_plan.json with predicted actions
"""

import json
import torch
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Global lock to serialize model loading and prevent meta tensor conflicts during parallel init
_model_load_lock = threading.Lock()


class ActionPlanner:
    """
    Action planner using Qwen3-VL to predict structured action plans.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_library_path: Optional[str] = None
    ):
        """
        Initialize the action planner.
        
        Args:
            model_name: Base Qwen3-VL model
            lora_checkpoint: Path to fine-tuned LoRA checkpoint (if available)
            device: Device to run inference on
            action_library_path: Path to action_library.json
        """
        self.device = device
        self.model_name = model_name
        
        print(f"Loading planner model: {model_name}")
        
        # Serialize model loading to prevent meta tensor conflicts during parallel init
        with _model_load_lock:
            # Load model and processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # When a specific device is provided, don't use device_map
            # device_map with specific GPU indices can cause meta tensor issues
            # Load without device_map and move to device afterwards
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Move model to specific device after loading
            self.model = self.model.to(device)
            
            # Load LoRA if available
            if lora_checkpoint:
                self._load_lora(lora_checkpoint)
        
        self.model.eval()
        
        # Load action library
        if action_library_path is None:
            action_library_path = Path(__file__).parent / "action_library_v2.json"  # V2: 10 atomic actions
        
        with open(action_library_path, 'r') as f:
            self.action_library = json.load(f)
        
        print(f"Planner ready. Device: {device}")
    
    def _load_lora(self, checkpoint_path: str):
        """Load LoRA checkpoint for fine-tuned planner."""
        try:
            from peft import PeftModel
            print(f"Loading LoRA from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                is_trainable=False
            )
            print("✓ LoRA loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load LoRA: {e}")
            print("Using base model instead")
    
    @torch.no_grad()
    def predict_action_plan(
        self,
        image_path: str,
        user_prompt: str,
        analysis: Dict[str, Any] = None,
        max_actions: int = 5,
        temperature: float = 0.1,
        return_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Predict action plan for image editing.
        
        Args:
            image_path: Path to input image
            user_prompt: User's editing request
            analysis: Optional image analysis to provide context (recommended for better reasoning)
            max_actions: Maximum number of actions to predict (2-5)
            temperature: Sampling temperature for generation (default: 0.1 for deterministic, matches data generation)
            return_reasoning: Whether to return reasoning/explanation
        
        Returns:
            action_plan dictionary with structure:
            {
                "overall_instruction": str,
                "actions": [
                    {
                        "action_id": str,
                        "priority": int,
                        "parameters": {...}
                    }
                ],
                "reasoning": str (optional)
            }
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Create prompt for action planning
        system_prompt = self._create_planning_prompt(max_actions)
        
        # Format analysis if provided (same format as data generation)
        analysis_text = ""
        if analysis:
            # Create a concise, structured summary of the analysis
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
        
        # Prepare conversation with analysis
        user_text = f"{system_prompt}\n\nUser Request: {user_prompt}{analysis_text}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate (parameters aligned with data generation: temperature=0.1, max_tokens=1024, deterministic)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=False,
            top_p=0.9,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )
        
        # Decode
        generated_text = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # Parse JSON response
        action_plan = self._parse_action_plan(generated_text, user_prompt)
        
        return action_plan
    
    def _create_planning_prompt(self, max_actions: int) -> str:
        """Create system prompt for Chain-of-Thought action planning."""
        
        # Get available actions
        actions_summary = []
        for action in self.action_library["actions"]:
            action_id = action["id"]
            description = action["description"]
            params = ", ".join(action["parameters"].keys())
            actions_summary.append(f"- {action_id}: {description}\n  Parameters: {params}")
        
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
- Select 2-5 appropriate actions from the library
- Focus on THEME transformations (scene/setting) and/or STYLE transformations (artistic rendering)
- Each action must contribute to a NOTICEABLE change
- For EACH action, provide "reasoning" explaining WHY it's needed for THIS image
- Set clear priorities (1=highest, foundation changes first)
- **CRITICAL: Keep parameter 'description' SHORT (5-10 words max)**

PER-ACTION REASONING REQUIREMENTS:
- Reference SPECIFIC image elements (objects, composition, current style)
- Explain WHY this action creates theme/style transformation
- Mention dependencies (e.g., "location_setting must come first to establish scene")
- Keep concise (1-2 sentences, ~20-40 words)
- Avoid generic statements ("to improve" or "to enhance")

STEP 3: GENERATE EDIT PROMPT (hidream_prompt)
- Create a concise instruction for the image editor (Qwen-Image-Edit)
- **MAXIMUM 77 TOKENS** (STRICT limit due to CLIP tokenizer)
- **MUST start with "style_transformation_mode"**
- If changing artistic STYLE: "style_transformation_mode Apply {{artistic_style}} with {{characteristics}}"
- If changing THEME only: "style_transformation_mode Transform to {{theme}}. Maintain photorealistic quality."
- If THEME + STYLE: "style_transformation_mode Transform to {{theme}} AND apply {{artistic_style}}."
- Focus on BOLD, noticeable changes

Examples of GOOD theme/style transformations:
✅ THEME: "Transform urban street to tropical beach" (location + weather + mood)
✅ STYLE: "Apply watercolor painting style with soft edges" (artistic medium)
✅ THEME+STYLE: "Change to Victorian era AND render as oil painting" (era + style)
✅ THEME: "Make it a moonlit night scene with eerie atmosphere" (time + mood + lighting)

Examples of BAD (too subtle/minor):
❌ "Adjust brightness slightly" (not theme/style)
❌ "Tweak color saturation" (too granular)
❌ "Change shadow direction" (minor technical adjustment)

Examples of GOOD per-action reasoning:
✅ "The modern glass buildings define this as contemporary urban. Complete location transformation to beach is the foundation, as all subsequent tropical elements depend on removing city infrastructure first."
✅ "Current photorealistic rendering doesn't match the artistic goal. Watercolor medium adds the dreamlike quality requested, transforming the visual style entirely."

Examples of BAD reasoning (too generic):
❌ "Change the location to make it better."
❌ "Add artistic style for improvement."

Examples of GOOD hidream_prompts:
✅ "style_transformation_mode Transform urban city to tropical beach with palm trees, clear skies, and sunny atmosphere. Maintain photorealistic quality." (21 tokens)
✅ "style_transformation_mode Apply watercolor painting style with soft edges, flowing colors, and artistic brushstrokes." (17 tokens)
✅ "style_transformation_mode Change to Victorian era with period architecture AND render as oil painting with rich textures." (20 tokens)
✅ "style_transformation_mode Transform to moonlit night scene with dramatic shadows and mysterious atmosphere. Keep photorealistic." (17 tokens)

Respond ONLY with valid JSON in this EXACT format:

{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
  "overall_instruction": "Brief summary of the complete edit",
  "actions": [
    {{
      "action_id": "location_setting",
      "reasoning": "The modern glass buildings and wet asphalt define this as urban. A complete location swap is required as foundation, since all other tropical elements depend on removing the city infrastructure first.",
      "priority": 1,
      "parameters": {{
        "source_location": "urban_city",
        "target_location": "beach_coast",
        "replace_mode": "complete",
        "preserve_foreground": true,
        "description": "Transform city to beach"
      }}
    }},
    {{
      "action_id": "weather_conditions",
      "reasoning": "Clear tropical weather enhances the beach atmosphere and provides the bright, sunny lighting needed to match the user's 'tropical paradise' vision.",
      "priority": 2,
      "parameters": {{
        "weather_type": "clear",
        "intensity": "moderate",
        "visibility": "high",
        "description": "Add clear tropical weather"
      }}
    }}
  ],
  "hidream_prompt": "style_transformation_mode [your concise instruction here, MAX 77 TOKENS]"
}}

CRITICAL: 
- "reasoning" MUST come FIRST in the JSON
- Each action MUST have its own "reasoning" field
- Think through the problem before selecting actions."""
        
        return prompt
    
    def _parse_action_plan(self, generated_text: str, user_prompt: str) -> Dict[str, Any]:
        """Parse generated text into action plan JSON with CoT validation."""
        try:
            # Try to extract JSON from response
            start_idx = generated_text.find("{")
            end_idx = generated_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_text = generated_text[start_idx:end_idx]
            action_plan = json.loads(json_text)
            
            # Validate Chain-of-Thought structure
            if "reasoning" not in action_plan or not action_plan["reasoning"]:
                print("⚠️ Warning: Missing or empty reasoning field in CoT plan")
                action_plan["reasoning"] = "No reasoning provided by model"
            
            if "actions" not in action_plan:
                raise ValueError("Missing 'actions' field")
            
            # Ensure overall_instruction exists
            if "overall_instruction" not in action_plan:
                action_plan["overall_instruction"] = user_prompt
            
            return action_plan
        
        except Exception as e:
            print(f"⚠️ Failed to parse action plan: {e}")
            print(f"Generated text: {generated_text[:500]}")
            
            # Fallback: Create simple single-action plan
            return self._create_fallback_plan(user_prompt)
    
    def _create_fallback_plan(self, user_prompt: str) -> Dict[str, Any]:
        """Create fallback plan with CoT reasoning when parsing fails."""
        # Simple heuristic-based action selection
        prompt_lower = user_prompt.lower()
        
        if any(word in prompt_lower for word in ["beach", "ocean", "tropical", "vacation"]):
            action_id = "location_setting"
            params = {
                "source_location": "urban_city",
                "target_location": "beach_coast",
                "replace_mode": "complete",
                "preserve_foreground": True,
                "description": user_prompt
            }
            reasoning = "The user requests a beach/tropical environment. I will apply location_setting to convert the scene to a beach environment while preserving the original subjects."
        elif any(word in prompt_lower for word in ["night", "evening", "sunset", "dawn"]):
            action_id = "time_of_day"
            params = {
                "source_time": "midday",
                "target_time": "sunset",
                "lighting_direction": "natural",
                "shadow_length": "long",
                "sky_color": "warm golden",
                "description": user_prompt
            }
            reasoning = "The user wants to change the time of day. I will use time_of_day to adjust lighting and atmosphere to match the requested time period."
        elif any(word in prompt_lower for word in ["style", "painting", "art", "anime"]):
            action_id = "artistic_medium"
            params = {
                "source_medium": "photograph",
                "target_medium": "oil_painting",
                "texture_level": "moderate",
                "detail_level": "high",
                "brushstroke_visible": False,
                "description": user_prompt
            }
            reasoning = "The user requests an artistic style change. I will apply artistic_medium to convert the rendering style while maintaining the scene's composition and content."
        else:
            # Generic transformation using color_grading as safe fallback
            action_id = "color_grading"
            params = {
                "grading_style": "natural",
                "temperature_shift": "neutral",
                "saturation_level": "moderate",
                "contrast_level": "moderate",
                "intensity": "moderate",
                "description": user_prompt
            }
            reasoning = "Based on the user request, I will apply color grading to modify the visual aesthetic while preserving the scene structure and content."
        
        return {
            "reasoning": reasoning,
            "overall_instruction": user_prompt,
            "actions": [
                {
                    "action_id": action_id,
                    "priority": 1,
                    "parameters": params
                }
            ]
        }
    
    def validate_action_plan(self, action_plan: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate action plan structure and parameters.
        
        Returns:
            (is_valid, error_message)
        """
        if "actions" not in action_plan:
            return False, "Missing 'actions' field"
        
        actions = action_plan["actions"]
        if not isinstance(actions, list):
            return False, "'actions' must be a list"
        
        if len(actions) == 0:
            return False, "At least one action is required"
        
        if len(actions) > 5:
            return False, "Maximum 5 actions allowed"
        
        # Validate each action
        valid_action_ids = {action["id"] for action in self.action_library["actions"]}
        
        for i, action in enumerate(actions):
            if "action_id" not in action:
                return False, f"Action {i}: Missing 'action_id'"
            
            action_id = action["action_id"]
            if action_id not in valid_action_ids:
                return False, f"Action {i}: Invalid action_id '{action_id}'"
            
            if "parameters" not in action:
                return False, f"Action {i}: Missing 'parameters'"
        
        return True, "Valid"


def create_planner(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    lora_checkpoint: Optional[str] = None,
    device: str = "cuda"
) -> ActionPlanner:
    """
    Factory function to create action planner.
    
    Args:
        model_name: Base Qwen3-VL model name
        lora_checkpoint: Path to fine-tuned LoRA checkpoint
        device: Device for inference
    
    Returns:
        ActionPlanner instance
    """
    return ActionPlanner(
        model_name=model_name,
        lora_checkpoint=lora_checkpoint,
        device=device
    )


if __name__ == "__main__":
    # Test example
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python planner_inference.py <image_path> <prompt>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    prompt = sys.argv[2]
    
    # Create planner (base model, no fine-tuning yet)
    planner = create_planner()
    
    # Predict action plan
    print(f"\nPredicting action plan for: {prompt}")
    action_plan = planner.predict_action_plan(image_path, prompt)
    
    # Validate
    is_valid, msg = planner.validate_action_plan(action_plan)
    print(f"\nValidation: {msg}")
    
    # Print result
    print("\nAction Plan:")
    print(json.dumps(action_plan, indent=2))

