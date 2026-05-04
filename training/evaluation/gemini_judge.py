#!/usr/bin/env python3
"""
Gemini 2.5 Flash Judge — uses google.genai SDK.
Same prompts as GPT4oJudge for fair comparison.
Reads GOOGLE_API_KEY or GEMINI_API_KEY env vars.
"""

import os
import json
import time
from typing import Dict, Optional
from PIL import Image

from training.evaluation.judge_base import BaseJudge


def _get_client():
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY. Set the environment variable directly")
    client = genai.Client(api_key=api_key)
    return client


def _parse_json(text: str) -> dict:
    """Parse JSON from model response, stripping markdown code blocks."""
    text = text.strip()
    # Handle ```json\n...\n``` or ```\n...\n```
    if "```" in text:
        # Extract content between first ``` and last ```
        inner = text.split("```", 2)
        if len(inner) >= 3:
            block = inner[1]
            if block.startswith("json"):
                block = block[4:]
            text = block.strip()
    return json.loads(text)


def _call_gemini(client, model_id, prompt: str, images: list, max_retries=3):
    """Call Gemini with text + PIL images. Returns parsed JSON dict."""
    from google.genai import types

    parts = images + [prompt]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            return _parse_json(response.text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


class GeminiJudge(BaseJudge):
    """Image quality judge using Gemini 2.5 Flash."""

    def __init__(self):
        self.client = _get_client()
        self.model_id = os.getenv("GEMINI_PLANNER_MODEL", "gemini-2.5-flash")
        print(f"✅ GeminiJudge initialized (model: {self.model_id})")

    def judge_single_edit(self, original_image, generated_image, instruction, max_retries=3):
        prompt = f"""You are an expert image editing quality evaluator.

TASK: Evaluate how well the generated image follows the editing instruction.

INPUT:
  - Image 1: Original (before editing)
  - Image 2: Generated (after editing)
  - Instruction: "{instruction}"

EVALUATE on a 0-10 scale:
  1. instruction_following: Does the edit fulfill what the instruction requested?
  2. visual_quality: Is the result natural, realistic, and artifact-free?
  3. transformation_strength: How completely were the requested changes applied?
  4. coherence: Does the edited image maintain internal consistency (lighting, perspective, style)?
  5. semantic_accuracy: Are objects/scenes semantically correct for what was requested?
  6. technical_execution: Quality of specific edits (color grading, object placement, style transfer)?

RESPOND ONLY with valid JSON (no markdown, no extra text):
{{
  "instruction_following": <score 0-10>,
  "visual_quality": <score 0-10>,
  "transformation_strength": <score 0-10>,
  "coherence": <score 0-10>,
  "semantic_accuracy": <score 0-10>,
  "technical_execution": <score 0-10>,
  "overall_image_score": <average of the 6 scores>,
  "reasoning": "<brief 1-2 sentence explanation>"
}}"""

        try:
            return _call_gemini(self.client, self.model_id, prompt,
                                [original_image, generated_image], max_retries)
        except Exception as e:
            return {k: 0.0 for k in ["instruction_following","visual_quality","transformation_strength",
                                      "coherence","semantic_accuracy","technical_execution","overall_image_score"]} | {"reasoning": f"Error: {e}", "error": str(e)}

    def judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan=None, max_retries=3):
        return GeminiActionJudge._do_judge(self, original_image, user_prompt, predicted_plan, teacher_plan, max_retries)


class GeminiActionJudge(BaseJudge):
    """Action plan quality judge using Gemini 2.5 Flash."""

    def __init__(self):
        self.client = _get_client()
        self.model_id = os.getenv("GEMINI_PLANNER_MODEL", "gemini-2.5-flash")
        print(f"✅ GeminiActionJudge initialized (model: {self.model_id})")

    @staticmethod
    def _format_action_plan(action_plan: dict) -> str:
        lines = []
        lines.append(f"Overall Instruction: {action_plan.get('overall_instruction', 'N/A')}")
        lines.append(f"\nActions ({len(action_plan.get('actions', []))}):")
        for i, action in enumerate(action_plan.get('actions', []), 1):
            action_id = action.get('action_id', 'unknown')
            priority = action.get('priority', 'N/A')
            params = action.get('parameters', {})
            lines.append(f"  {i}. {action_id} (priority: {priority})")
            if 'object_id' in params:
                lines.append(f"     - object: {params['object_id']}")
            if 'source_color' in params and 'target_color' in params:
                lines.append(f"     - color: {params['source_color']} \u2192 {params['target_color']}")
            if 'theme' in params:
                lines.append(f"     - theme: {params['theme']}")
            if 'style' in params:
                lines.append(f"     - style: {params['style']}")
            if 'brightness_adjustment' in params:
                lines.append(f"     - brightness: {params['brightness_adjustment']}")
            if 'contrast_adjustment' in params:
                lines.append(f"     - contrast: {params['contrast_adjustment']}")
        return "\n".join(lines)

    @staticmethod
    def _do_judge(self_obj, original_image, user_prompt, predicted_plan, teacher_plan, max_retries):
        formatted_plan = GeminiActionJudge._format_action_plan(predicted_plan)
        reasoning_text = predicted_plan.get("reasoning", "No reasoning provided")
        reasoning_word_count = len(reasoning_text.split())

        prompt = f"""You are an expert image editing evaluator specializing in THEME & STYLE transformation assessment.

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
   - Too short (<30 words): likely superficial \u2192 score 3-5
   - Too long (>100 words): verbose/redundant \u2192 score 4-6
   - Just right (40-70 words): well-balanced \u2192 score 8-10

7. REASONING COMPLETENESS (0-10): Does it cover all key aspects?
   - Current state: What is in the image now?
   - Goal: What transformation is requested?
   - Approach: What strategy/sequence will be used?
   - Per-action reasoning: Why each specific action?
   - Missing 2+ aspects \u2192 score \u22645
   - Missing 1 aspect \u2192 score 6-7
   - Covers all aspects \u2192 score 8-10

8. REASONING SPECIFICITY (0-10): Does it mention specific image details?
   - Generic reasoning (no image details): "Add sunset effect" \u2192 score 3-5
   - Somewhat specific (some details): "Change sky color for sunset" \u2192 score 6-7
   - Highly specific (concrete details): "Transform blue midday sky to orange/pink sunset with warm lighting on buildings" \u2192 score 8-10
   - Check: Are colors, objects, regions, or visual elements explicitly mentioned?

Examples of GOOD theme/style plans:
\u2705 Urban street \u2192 tropical beach (location + weather + atmosphere)
\u2705 Apply watercolor painting style (artistic medium transformation)
\u2705 Daylight \u2192 moonlit night with eerie atmosphere (time + lighting + mood)

Examples of BAD plans (not theme/style focused):
\u274c Adjust brightness/contrast slightly (too subtle)
\u274c Crop and rotate image (not transformation)
\u274c Change single object color (too granular)

RESPOND ONLY with valid JSON (no markdown, no extra text):
{{
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
}}"""

        try:
            scores = _call_gemini(self_obj.client, self_obj.model_id, prompt,
                                  [original_image], max_retries)
            for k in ["relevance","theme_style_focus","completeness","efficiency","correctness",
                      "reasoning_conciseness","reasoning_completeness","reasoning_specificity",
                      "overall_action_quality","overall_reasoning_quality","overall_score"]:
                scores[k] = float(scores[k])
            return scores
        except Exception as e:
            return {k: 0.0 for k in ["relevance","theme_style_focus","completeness","efficiency","correctness",
                                      "reasoning_conciseness","reasoning_completeness","reasoning_specificity",
                                      "overall_action_quality","overall_reasoning_quality","overall_score"]} | {"explanation": f"Error: {e}", "error": str(e)}

    def judge_single_edit(self, original_image, generated_image, instruction, max_retries=3):
        return GeminiJudge().judge_single_edit(original_image, generated_image, instruction, max_retries)

    def judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan=None, max_retries=3):
        return self._do_judge(self, original_image, user_prompt, predicted_plan, teacher_plan, max_retries)
