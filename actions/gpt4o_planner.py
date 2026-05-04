"""
GPT-4o Vision Planner - API-based Action Planning

This module provides a GPT-4o-based planner that matches the ActionPlanner interface.
It uses Azure OpenAI GPT-4o Vision to generate structured action plans for image editing.

Key Features:
- Compatible with ActionPlanner interface (drop-in replacement)
- Uses Azure OpenAI GPT-4o Vision API
- Returns action plans in identical JSON format
- No local model loading required (API-based)
"""

import os
import json
import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import requests


class GPT4oPlanner:
    """
    Action planner using GPT-4o Vision to predict structured action plans.
    
    This class matches the ActionPlanner interface for compatibility with
    existing evaluation scripts.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        lora_checkpoint: Optional[str] = None,  # Ignored (no LoRA for API)
        device: str = "cpu",  # Ignored (API-based)
        action_library_path: Optional[str] = None,
        azure_config: Optional[Dict] = None
    ):
        """
        Initialize GPT-4o planner with Azure credentials.
        
        Args:
            model_name: Model name (ignored, always uses GPT-4o)
            lora_checkpoint: LoRA checkpoint path (ignored, no LoRA for API)
            device: Device (ignored, API-based)
            action_library_path: Path to action_library.json
            azure_config: Dict with Azure OpenAI credentials (optional)
        """
        self.model_name = "gpt-4o"
        
        # Use provided Azure config or load from environment variables
        if azure_config is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
            
            if not azure_endpoint or not api_key:
                raise ValueError(
                    "Azure OpenAI credentials not found. Please set environment variables:\n"
                    "  - AZURE_OPENAI_ENDPOINT\n"
                    "  - AZURE_OPENAI_API_KEY\n"
                    "Or export the required environment variables: export the required environment variables"
                )
            
            azure_config = {
                "azure_endpoint": azure_endpoint,
                "api_key": api_key,
                "api_version": api_version,
                "model": model,
            }
        
        self.endpoint = azure_config["azure_endpoint"]
        self.api_key = azure_config["api_key"]
        self.api_version = azure_config["api_version"]
        self.model = azure_config["model"]
        
        # Construct full URL
        self.url = f"{self.endpoint}openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # Load action library
        if action_library_path is None:
            action_library_path = Path(__file__).parent / "action_library_v2.json"
        
        with open(action_library_path, 'r') as f:
            self.action_library = json.load(f)
        
        print(f"✅ GPT-4o Planner initialized (model: {self.model})")
    
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _create_planning_prompt(self, max_actions: int = 5) -> str:
        """
        Create the system prompt for action planning.
        
        Args:
            max_actions: Maximum number of actions to predict
            
        Returns:
            Formatted system prompt with action library
        """
        # Format action library for prompt
        actions_text = []
        for action in self.action_library["actions"]:
            # Note: action library uses 'id' not 'action_id'
            action_id = action.get('id', action.get('action_id', 'unknown'))
            action_name = action.get('name', action_id)
            action_desc_text = action.get('description', '')
            
            action_desc = f"- **{action_id}** ({action_name}): {action_desc_text}"
            params = action.get("parameters", {})
            if params:
                param_list = []
                for param_name, param_info in params.items():
                    if param_name != "description":  # Skip description field if it's a param
                        param_desc = param_info.get('description', '') if isinstance(param_info, dict) else ''
                        param_list.append(f"  - `{param_name}`: {param_desc}")
                if param_list:  # Only add if we have actual parameters
                    action_desc += "\n" + "\n".join(param_list)
            actions_text.append(action_desc)
        
        actions_library_text = "\n".join(actions_text)
        
        prompt = f"""You are an expert THEME & STYLE transformation planner using Chain-of-Thought reasoning.

Your specialty: Planning BOLD, NOTICEABLE transformations that change:
- THEME: The scene's setting, atmosphere, mood, or environmental context
- STYLE: The artistic rendering or visual medium

AVAILABLE ACTIONS:
{actions_library_text}

INSTRUCTIONS:
STEP 1: ANALYZE & REASON (Theme/Style Focus)
- Review the image - what's in the image
- What is the current THEME (setting, mood, atmosphere, time)?
- What is the current STYLE (photorealistic, painting, artistic)?
- What THEME or STYLE does the user want?
- What actions would create a BOLD, NOTICEABLE transformation?
- Prioritize dramatic changes over subtle adjustments

STEP 2: PLAN ACTIONS WITH REASONING
- Select 2-{max_actions} appropriate actions from the library
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

Examples of GOOD hidream_prompts:
✅ "style_transformation_mode Transform urban city to tropical beach with palm trees, clear skies, and sunny atmosphere. Maintain photorealistic quality."
✅ "style_transformation_mode Apply watercolor painting style with soft edges, flowing colors, and artistic brushstrokes."
✅ "style_transformation_mode Change to Victorian era with period architecture AND render as oil painting with rich textures."
✅ "style_transformation_mode Transform to moonlit night scene with dramatic shadows and mysterious atmosphere. Keep photorealistic."

Respond ONLY with valid JSON in this EXACT format:

{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification].",
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
    }}
  ],
  "hidream_prompt": "style_transformation_mode [your concise instruction here, MAX 77 TOKENS]"
}}

CRITICAL: 
- "reasoning" MUST come FIRST in the JSON
- Each action MUST have its own "reasoning" field
- MUST include "hidream_prompt" field for image editor
- Think through the problem before selecting actions
- Return ONLY valid JSON, no additional text
"""
        return prompt
    
    def predict_action_plan(
        self,
        image_path: str,
        user_prompt: str,
        analysis: Dict[str, Any] = None,
        max_actions: int = 5,
        temperature: float = 0.1,
        return_reasoning: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Predict action plan for image editing using GPT-4o Vision.
        
        Args:
            image_path: Path to input image
            user_prompt: User's editing request
            analysis: Optional image analysis to provide context
            max_actions: Maximum number of actions to predict (2-5)
            temperature: Sampling temperature for generation
            return_reasoning: Whether to return reasoning/explanation (ignored for now)
            max_retries: Maximum number of API retry attempts
        
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
                ]
            }
        """
        # Load and encode image
        image = Image.open(image_path).convert("RGB")
        image_base64 = self._encode_image_base64(image)
        
        # Create planning prompt
        system_prompt = self._create_planning_prompt(max_actions)
        
        # Format analysis if provided (same format as ActionPlanner)
        analysis_text = ""
        if analysis:
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
            
            if analysis_parts:
                analysis_text = "\n\nIMAGE ANALYSIS:\n" + "\n".join(f"- {part}" for part in analysis_parts)
        
        # Prepare user message
        user_text = f"{system_prompt}\n\nUser Request: {user_prompt}{analysis_text}"
        
        # Prepare API request payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_text
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": temperature
        }
        
        # Make API request with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON from response (handle markdown code blocks)
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # Parse action plan
                action_plan = json.loads(content)
                
                # Validate structure
                if "actions" not in action_plan:
                    raise ValueError("Response missing 'actions' field")
                
                if "overall_instruction" not in action_plan:
                    # Add overall_instruction if missing
                    action_plan["overall_instruction"] = user_prompt
                
                # Validate each action
                for action in action_plan["actions"]:
                    if "action_id" not in action:
                        raise ValueError("Action missing 'action_id' field")
                    if "priority" not in action:
                        raise ValueError("Action missing 'priority' field")
                    if "parameters" not in action:
                        action["parameters"] = {}
                
                return action_plan
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  API request attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ GPT-4o API request failed after {max_retries} attempts: {e}")
                    # Return empty plan on failure
                    return {
                        "overall_instruction": user_prompt,
                        "actions": [],
                        "error": str(e)
                    }
            
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Response parsing attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ Failed to parse GPT-4o response after {max_retries} attempts: {e}")
                    # Return empty plan on failure
                    return {
                        "overall_instruction": user_prompt,
                        "actions": [],
                        "error": f"Parse error: {str(e)}"
                    }
        
        # Should never reach here, but just in case
        return {
            "overall_instruction": user_prompt,
            "actions": [],
            "error": "Unknown error"
        }


if __name__ == "__main__":
    # Test GPT-4o Planner
    print("Testing GPT-4o Planner...")
    
    # Initialize planner
    planner = GPT4oPlanner()
    
    # Test with a simple image (you'll need to provide a valid image path)
    test_image_path = "./sample_image.png"  # update path
    
    if Path(test_image_path).exists():
        print(f"\nTesting with image: {test_image_path}")
        print("User prompt: Make the sky more vibrant and colorful")
        
        action_plan = planner.predict_action_plan(
            image_path=test_image_path,
            user_prompt="Make the sky more vibrant and colorful",
            max_actions=3
        )
        
        print("\n✅ Generated Action Plan:")
        print(json.dumps(action_plan, indent=2))
    else:
        print(f"\n⚠️  Test image not found: {test_image_path}")
        print("Skipping test run")




def build_theme_style_planning_prompt(action_library: dict, max_actions: int = 5) -> str:
    """Standalone version of GPT4oPlanner._create_planning_prompt for use by other planners."""
    actions_text = []
    for action in action_library["actions"]:
        action_id = action.get('id', action.get('action_id', 'unknown'))
        action_name = action.get('name', action_id)
        action_desc_text = action.get('description', '')
        action_desc = f"- **{action_id}** ({action_name}): {action_desc_text}"
        params = action.get("parameters", {})
        if params:
            param_list = []
            for param_name, param_info in params.items():
                if param_name != "description":
                    param_desc = param_info.get('description', '') if isinstance(param_info, dict) else ''
                    param_list.append(f"  - `{param_name}`: {param_desc}")
            if param_list:
                action_desc += "\n" + "\n".join(param_list)
        actions_text.append(action_desc)

    actions_library_text = "\n".join(actions_text)

    return f"""You are an expert THEME & STYLE transformation planner using Chain-of-Thought reasoning.

Your specialty: Planning BOLD, NOTICEABLE transformations that change:
- THEME: The scene's setting, atmosphere, mood, or environmental context
- STYLE: The artistic rendering or visual medium

AVAILABLE ACTIONS:
{actions_library_text}

INSTRUCTIONS:
STEP 1: ANALYZE & REASON (Theme/Style Focus)
- Review the image - what's in the image
- What is the current THEME (setting, mood, atmosphere, time)?
- What is the current STYLE (photorealistic, painting, artistic)?
- What THEME or STYLE does the user want?
- What actions would create a BOLD, NOTICEABLE transformation?
- Prioritize dramatic changes over subtle adjustments

STEP 2: PLAN ACTIONS WITH REASONING
- Select 2-{max_actions} appropriate actions from the library
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

Examples of GOOD hidream_prompts:
✅ "style_transformation_mode Transform urban city to tropical beach with palm trees, clear skies, and sunny atmosphere. Maintain photorealistic quality."
✅ "style_transformation_mode Apply watercolor painting style with soft edges, flowing colors, and artistic brushstrokes."
✅ "style_transformation_mode Change to Victorian era with period architecture AND render as oil painting with rich textures."
✅ "style_transformation_mode Transform to moonlit night scene with dramatic shadows and mysterious atmosphere. Keep photorealistic."

Respond ONLY with valid JSON in this EXACT format:

{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification].",
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
    }}
  ],
  "hidream_prompt": "style_transformation_mode [your concise instruction here, MAX 77 TOKENS]"
}}

CRITICAL: 
- "reasoning" MUST come FIRST in the JSON
- Each action MUST have its own "reasoning" field
- MUST include "hidream_prompt" field for image editor
- Think through the problem before selecting actions
- Return ONLY valid JSON, no additional text
"""
