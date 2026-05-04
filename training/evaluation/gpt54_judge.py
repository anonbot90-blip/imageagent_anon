#!/usr/bin/env python3
"""
GPT-5.4 Judge — uses Azure OpenAI GPT-5.4 endpoint.
Same prompts as GPT4oJudge for fair comparison.
Reads AZURE_OPENAI_GPT54_* env vars.
"""

import os
import json
import base64
import time
from io import BytesIO
from typing import Dict, Optional
from PIL import Image
import requests

from training.evaluation.judge_base import BaseJudge


class GPT54Judge(BaseJudge):
    """Image quality judge using GPT-5.4 via Azure OpenAI."""

    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_GPT54_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_GPT54_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_GPT54_API_VERSION", "2025-04-01-preview")
        model = os.getenv("AZURE_OPENAI_GPT54_MODEL", "gpt-5.4")

        if not endpoint or not api_key:
            raise ValueError(
                "GPT-5.4 credentials not found. Set:\n"
                "  AZURE_OPENAI_GPT54_ENDPOINT\n"
                "  AZURE_OPENAI_GPT54_API_KEY\n"
                "Or export the required environment variables"
            )

        self.url = f"{endpoint}openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.headers = {"Content-Type": "application/json", "api-key": api_key}
        self.model = model
        print(f"✅ GPT54Judge initialized (model: {model})")

    def _encode(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def judge_single_edit(self, original_image, generated_image, instruction, max_retries=3):
        orig_b64 = self._encode(original_image)
        gen_b64 = self._encode(generated_image)

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

        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
            ]}],
            "max_completion_tokens": 500,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(max_retries):
            try:
                r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
                r.raise_for_status()
                return json.loads(r.json()["choices"][0]["message"]["content"])
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {k: 0.0 for k in ["instruction_following","visual_quality","transformation_strength",
                                              "coherence","semantic_accuracy","technical_execution","overall_image_score"]} | {"reasoning": f"Error: {e}", "error": str(e)}

    def judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan=None, max_retries=3):
        return GPT54ActionJudge._shared_judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan, max_retries)


class GPT54ActionJudge(BaseJudge):
    """Action plan quality judge using GPT-5.4 via Azure OpenAI."""

    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_GPT54_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_GPT54_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_GPT54_API_VERSION", "2025-04-01-preview")
        model = os.getenv("AZURE_OPENAI_GPT54_MODEL", "gpt-5.4")

        if not endpoint or not api_key:
            raise ValueError("GPT-5.4 credentials not found. Set AZURE_OPENAI_GPT54_* vars or export the required environment variables")

        self.url = f"{endpoint}openai/deployments/{model}/chat/completions?api-version={api_version}"
        self.headers = {"Content-Type": "application/json", "api-key": api_key}
        self.model = model
        print(f"✅ GPT54ActionJudge initialized (model: {model})")

    def _encode(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _shared_judge_action_plan(self_obj, original_image, user_prompt, predicted_plan, teacher_plan, max_retries):
        image_b64 = self_obj._encode(original_image)

        lines = [f"Overall Instruction: {predicted_plan.get('overall_instruction', 'N/A')}",
                 f"\nActions ({len(predicted_plan.get('actions', []))}):" ]
        for i, a in enumerate(predicted_plan.get("actions", []), 1):
            lines.append(f"  {i}. {a.get('action_id','?')} (priority: {a.get('priority','?')})")
        formatted_plan = "\n".join(lines)
        reasoning_text = predicted_plan.get("reasoning", "No reasoning provided")

        prompt = f"""You are an expert image editing evaluator specializing in THEME & STYLE transformation assessment.

User Request: "{user_prompt}"
Predicted Plan:
{formatted_plan}

Reasoning ({len(reasoning_text.split())} words): {reasoning_text}

Score 0-10 on 8 dimensions:
ACTION: relevance, theme_style_focus, completeness, efficiency, correctness
REASONING: reasoning_conciseness, reasoning_completeness, reasoning_specificity

RESPOND ONLY with valid JSON:
{{
  "relevance": <0-10>,
  "theme_style_focus": <0-10>,
  "completeness": <0-10>,
  "efficiency": <0-10>,
  "correctness": <0-10>,
  "reasoning_conciseness": <0-10>,
  "reasoning_completeness": <0-10>,
  "reasoning_specificity": <0-10>,
  "overall_action_quality": <avg of 5 action dims>,
  "overall_reasoning_quality": <avg of 3 reasoning dims>,
  "overall_score": <avg of all 8>,
  "explanation": "<2-3 sentence summary>"
}}"""

        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]}],
            "max_completion_tokens": 600,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(max_retries):
            try:
                r = requests.post(self_obj.url, headers=self_obj.headers, json=payload, timeout=60)
                r.raise_for_status()
                scores = json.loads(r.json()["choices"][0]["message"]["content"])
                for k in ["relevance","theme_style_focus","completeness","efficiency","correctness",
                          "reasoning_conciseness","reasoning_completeness","reasoning_specificity",
                          "overall_action_quality","overall_reasoning_quality","overall_score"]:
                    scores[k] = float(scores[k])
                return scores
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {k: 0.0 for k in ["relevance","theme_style_focus","completeness","efficiency","correctness",
                                              "reasoning_conciseness","reasoning_completeness","reasoning_specificity",
                                              "overall_action_quality","overall_reasoning_quality","overall_score"]} | {"explanation": f"Error: {e}", "error": str(e)}

    def judge_single_edit(self, original_image, generated_image, instruction, max_retries=3):
        return GPT54Judge().__class__.judge_single_edit(GPT54Judge(), original_image, generated_image, instruction, max_retries)

    def judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan=None, max_retries=3):
        return self._shared_judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan, max_retries)
