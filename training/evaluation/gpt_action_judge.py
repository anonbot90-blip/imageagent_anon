#!/usr/bin/env python3
"""
GPT-4o Action Plan Judge for Planner Quality Evaluation

Uses Azure OpenAI GPT-4o to evaluate action plan quality independently:
- Relevance: Does the plan address the user's request?
- Completeness: Are all necessary edits covered?
- Efficiency: Is the plan unnecessarily complex?
- Correctness: Are the action parameters reasonable?

This provides an independent quality assessment separate from teacher-based metrics.
"""

import os
import json
import base64
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from PIL import Image
import requests


class GPT4oActionJudge:
    """Judge action plan quality using GPT-4o Vision"""
    
    def __init__(self, azure_config: Optional[Dict] = None):
        """
        Initialize GPT-4o action judge with Azure credentials.
        
        Args:
            azure_config: Dict with 'azure_endpoint', 'api_key', 'api_version', 'model'
                         If None, uses hardcoded credentials
        """
        # Use provided config or load from environment variables
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
        
        print(f"✅ GPT-4o Action Judge initialized (model: {self.model})")
    
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _format_action_plan(self, action_plan: Dict) -> str:
        """Format action plan as readable text"""
        lines = []
        lines.append(f"Overall Instruction: {action_plan.get('overall_instruction', 'N/A')}")
        lines.append(f"\nActions ({len(action_plan.get('actions', []))}):")
        
        for i, action in enumerate(action_plan.get('actions', []), 1):
            action_id = action.get('action_id', 'unknown')
            priority = action.get('priority', 'N/A')
            params = action.get('parameters', {})
            
            lines.append(f"  {i}. {action_id} (priority: {priority})")
            
            # Format key parameters
            if 'object_id' in params:
                lines.append(f"     - object: {params['object_id']}")
            if 'source_color' in params and 'target_color' in params:
                lines.append(f"     - color: {params['source_color']} → {params['target_color']}")
            if 'theme' in params:
                lines.append(f"     - theme: {params['theme']}")
            if 'style' in params:
                lines.append(f"     - style: {params['style']}")
            if 'brightness_adjustment' in params:
                lines.append(f"     - brightness: {params['brightness_adjustment']}")
            if 'contrast_adjustment' in params:
                lines.append(f"     - contrast: {params['contrast_adjustment']}")
        
        return "\n".join(lines)
    
    def _create_judge_prompt(
        self, 
        user_prompt: str, 
        action_plan: Dict,
        include_teacher: bool = False,
        teacher_plan: Optional[Dict] = None
    ) -> str:
        """Create the judging prompt"""
        
        formatted_plan = self._format_action_plan(action_plan)
        
        # Extract reasoning if present
        reasoning_text = action_plan.get('reasoning', 'No reasoning provided')
        reasoning_word_count = len(reasoning_text.split())
        
        base_prompt = f"""You are an expert image editing evaluator specializing in THEME & STYLE transformation assessment.

CONTEXT:
  - User Request: "{user_prompt}"
  - You will see the original image
  - Focus: Evaluate plan's ability to create BOLD, NOTICEABLE theme/style transformations
  
PREDICTED ACTION PLAN:
{formatted_plan}

REASONING PROVIDED WITH PLAN ({reasoning_word_count} words):
{reasoning_text}

EVALUATION TASK:
Evaluate the quality of this action plan on a 0-10 scale across 8 dimensions:

ACTION QUALITY (5 dimensions):

1. RELEVANCE (0-10): Does the plan address what the user asked for?
   - Do the actions match the user's intent?
   - Are the selected actions appropriate for the theme/style request?

2. THEME/STYLE FOCUS (0-10): Does the plan prioritize theme/style transformations?
   - Is the plan focused on THEME changes (setting, atmosphere, mood, environment)?
   - OR is it focused on STYLE changes (artistic rendering, visual medium)?
   - Are the transformations BOLD and NOTICEABLE (vs subtle/minor adjustments)?
   - Does it avoid overly granular or technical adjustments?

3. COMPLETENESS (0-10): Are all necessary edits covered?
   - Does the plan include all required changes for the theme/style transformation?
   - Are any critical steps missing?

4. EFFICIENCY (0-10): Is the plan well-structured?
   - Is it overly complex or redundant?
   - Are actions properly prioritized (foundation changes first)?
   - Could it be simplified without losing functionality?

5. CORRECTNESS (0-10): Are the action parameters reasonable?
   - Are object IDs, colors, styles, etc. specified correctly?
   - Would these parameters likely produce a BOLD, noticeable transformation?

REASONING QUALITY (3 dimensions):

6. REASONING CONCISENESS (0-10): Is reasoning appropriately detailed?
   - Optimal length: 40-70 words (focused and informative)
   - Too short (<30 words): likely superficial → score 3-5
   - Too long (>100 words): verbose/redundant → score 4-6
   - Just right (40-70 words): well-balanced → score 8-10

7. REASONING COMPLETENESS (0-10): Does it cover all key aspects?
   - Current state: What is in the image now?
   - Goal: What transformation is requested?
   - Approach: What strategy/sequence will be used?
   - Per-action reasoning: Why each specific action?
   - Missing 2+ aspects → score ≤5
   - Missing 1 aspect → score 6-7
   - Covers all aspects → score 8-10

8. REASONING SPECIFICITY (0-10): Does it mention specific image details?
   - Generic reasoning (no image details): "Add sunset effect" → score 3-5
   - Somewhat specific (some details): "Change sky color for sunset" → score 6-7
   - Highly specific (concrete details): "Transform blue midday sky to orange/pink sunset with warm lighting on buildings" → score 8-10
   - Check: Are colors, objects, regions, or visual elements explicitly mentioned?

Examples of GOOD theme/style plans:
✅ Urban street → tropical beach (location + weather + atmosphere)
✅ Apply watercolor painting style (artistic medium transformation)
✅ Daylight → moonlit night with eerie atmosphere (time + lighting + mood)

Examples of BAD plans (not theme/style focused):
❌ Adjust brightness/contrast slightly (too subtle)
❌ Crop and rotate image (not transformation)
❌ Change single object color (too granular)

"""

        if include_teacher and teacher_plan:
            formatted_teacher = self._format_action_plan(teacher_plan)
            base_prompt += f"""
REFERENCE PLAN (for context only, not ground truth):
{formatted_teacher}

Note: This reference plan was created by another model. It may or may not be optimal.
Judge the predicted plan on its own merits, not by how well it matches the reference.

"""

        base_prompt += """RESPOND ONLY with valid JSON (no markdown, no extra text):
{
  "relevance": <score 0-10>,
  "theme_style_focus": <score 0-10>,
  "completeness": <score 0-10>,
  "efficiency": <score 0-10>,
  "correctness": <score 0-10>,
  "reasoning_conciseness": <score 0-10>,
  "reasoning_completeness": <score 0-10>,
  "reasoning_specificity": <score 0-10>,
  "overall_action_quality": <average of 5 action dimensions>,
  "overall_reasoning_quality": <average of 3 reasoning dimensions>,
  "overall_score": <average of ALL 8 dimensions>,
  "explanation": "<brief 2-3 sentence explanation focusing on key strengths/weaknesses of both actions AND reasoning>"
}"""

        return base_prompt
    
    def judge_action_plan(
        self,
        original_image: Image.Image,
        user_prompt: str,
        predicted_plan: Dict,
        teacher_plan: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Judge a single action plan.
        
        Args:
            original_image: Original image before editing
            user_prompt: User's editing request
            predicted_plan: Predicted action plan to evaluate
            teacher_plan: Optional teacher/reference plan for context
            max_retries: Number of retries on failure
            
        Returns:
            Dict with scores and reasoning
        """
        # Encode image to base64
        image_b64 = self._encode_image_base64(original_image)
        
        # Create prompt
        include_teacher = teacher_plan is not None
        prompt = self._create_judge_prompt(user_prompt, predicted_plan, include_teacher, teacher_plan)
        
        # Prepare API request
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=45
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                scores = json.loads(content)
                
                # Validate scores
                required_keys = [
                    "relevance", "theme_style_focus", "completeness", "efficiency", "correctness",
                    "reasoning_conciseness", "reasoning_completeness", "reasoning_specificity",
                    "overall_action_quality", "overall_reasoning_quality", "overall_score", "explanation"
                ]
                if not all(k in scores for k in required_keys):
                    raise ValueError(f"Missing required keys in response: {scores.keys()}")
                
                # Ensure scores are numeric
                numeric_keys = [
                    "relevance", "theme_style_focus", "completeness", "efficiency", "correctness",
                    "reasoning_conciseness", "reasoning_completeness", "reasoning_specificity",
                    "overall_action_quality", "overall_reasoning_quality", "overall_score"
                ]
                for key in numeric_keys:
                    scores[key] = float(scores[key])
                
                return scores
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  GPT-4o judge attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ GPT-4o action judge failed after {max_retries} attempts: {e}")
                    # Return default scores on failure
                    return {
                        "relevance": 0.0,
                        "theme_style_focus": 0.0,
                        "completeness": 0.0,
                        "efficiency": 0.0,
                        "correctness": 0.0,
                        "reasoning_conciseness": 0.0,
                        "reasoning_completeness": 0.0,
                        "reasoning_specificity": 0.0,
                        "overall_action_quality": 0.0,
                        "overall_reasoning_quality": 0.0,
                        "overall_score": 0.0,
                        "explanation": f"Evaluation failed: {str(e)}",
                        "error": str(e)
                    }
    
    def judge_batch(
        self,
        samples: List[Tuple[Image.Image, str, Dict, Optional[Dict]]]
    ) -> List[Dict]:
        """
        Judge a batch of action plans.
        
        Args:
            samples: List of (original_image, user_prompt, predicted_plan, teacher_plan) tuples
            
        Returns:
            List of score dictionaries
        """
        results = []
        for i, (img, prompt, pred_plan, teacher_plan) in enumerate(samples, 1):
            print(f"  Judging action plan {i}/{len(samples)}...")
            scores = self.judge_action_plan(img, prompt, pred_plan, teacher_plan)
            results.append(scores)
            
            # Rate limiting (to avoid hitting API limits)
            if i < len(samples):
                time.sleep(0.5)
        
        return results


if __name__ == "__main__":
    # Test GPT-4o Action Judge
    print("Testing GPT-4o Action Judge...")
    
    judge = GPT4oActionJudge()
    
    # Create dummy test data
    original = Image.new('RGB', (256, 256), color='red')
    user_prompt = "Change the car to blue and add a sunset background"
    
    predicted_plan = {
        "overall_instruction": "Change car color and add sunset",
        "reasoning": "The image shows a red car in daylight. To fulfill the request, I will first change the car's color from red to blue using color transformation, then apply a sunset theme to the background and sky to create warm evening lighting with orange and pink tones.",
        "actions": [
            {
                "action_id": "change_color",
                "priority": 1,
                "parameters": {
                    "object_id": "car",
                    "source_color": "red",
                    "target_color": "blue"
                }
            },
            {
                "action_id": "apply_theme",
                "priority": 2,
                "parameters": {
                    "theme": "sunset",
                    "regions": ["background", "sky"]
                }
            }
        ]
    }
    
    teacher_plan = {
        "overall_instruction": "Modify car and background",
        "actions": [
            {
                "action_id": "change_color",
                "priority": 1,
                "parameters": {
                    "object_id": "car",
                    "source_color": "red",
                    "target_color": "blue"
                }
            }
        ]
    }
    
    print("\nEvaluating test action plan...")
    result = judge.judge_action_plan(original, user_prompt, predicted_plan, teacher_plan)
    
    print("\n✅ GPT-4o Action Judge Result:")
    print(f"  Relevance:             {result['relevance']}/10")
    print(f"  Theme/Style Focus:     {result['theme_style_focus']}/10")
    print(f"  Completeness:          {result['completeness']}/10")
    print(f"  Efficiency:            {result['efficiency']}/10")
    print(f"  Correctness:           {result['correctness']}/10")
    print(f"  Reasoning Conciseness: {result['reasoning_conciseness']}/10")
    print(f"  Reasoning Completeness:{result['reasoning_completeness']}/10")
    print(f"  Reasoning Specificity: {result['reasoning_specificity']}/10")
    print(f"  Overall Action Qual:   {result['overall_action_quality']}/10")
    print(f"  Overall Reasoning Qual:{result['overall_reasoning_quality']}/10")
    print(f"  Overall Score:         {result['overall_score']}/10")
    print(f"  Explanation:           {result['explanation']}")
    
    if 'error' not in result:
        print("\n✅ GPT-4o Action Judge test complete!")
    else:
        print(f"\n⚠️  Test completed with errors: {result['error']}")

