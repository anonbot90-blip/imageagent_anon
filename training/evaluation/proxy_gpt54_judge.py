#!/usr/bin/env python3
"""
GPT-5.4 Judge — uses internal LLM proxy (OpenAI-compatible API).
Same prompts as GPT54Judge for fair comparison.
Reads LLM_PROXY_KEY and LLM_PROXY_ENDPOINT env vars.
"""

import os
import json
import base64
import time
from io import BytesIO
from PIL import Image

from training.evaluation.judge_base import BaseJudge


def _get_client():
    from openai import OpenAI
    key = os.getenv("LLM_PROXY_KEY")
    if not key:
        raise ValueError("LLM_PROXY_KEY not set. Add it to credentials.sh")
    endpoint = os.getenv("LLM_PROXY_ENDPOINT", "http://YOUR_LLM_PROXY_ENDPOINT:4000")
    return OpenAI(api_key="Bearer " + key, base_url=endpoint + "/v1")


def _encode(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _call_proxy(client, model, messages, max_tokens=600, max_retries=3) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
            if "```" in text:
                parts = text.split("```")
                if len(parts) >= 3:
                    block = parts[1]
                    if block.startswith("json"):
                        block = block[4:]
                    text = block.strip()
            return json.loads(text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


class ProxyGPT54Judge(BaseJudge):
    """Image quality judge using GPT-5.4 via LLM proxy."""

    def __init__(self):
        self.client = _get_client()
        self.model = os.getenv("LLM_PROXY_GPT54_MODEL", "gpt-5.4")
        print(f"✅ ProxyGPT54Judge initialized (model: {self.model})")

    def judge_single_edit(self, original_image, generated_image, instruction, max_retries=3):
        orig_b64 = _encode(original_image)
        gen_b64 = _encode(generated_image)

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

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}},
        ]}]

        try:
            return _call_proxy(self.client, self.model, messages, max_tokens=500, max_retries=max_retries)
        except Exception as e:
            return {k: 0.0 for k in ["instruction_following", "visual_quality", "transformation_strength",
                                      "coherence", "semantic_accuracy", "technical_execution", "overall_image_score"]} | \
                   {"reasoning": f"Error: {e}", "error": str(e)}

    def judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan=None, max_retries=3):
        return ProxyGPT54ActionJudge._do_judge(self, original_image, user_prompt, predicted_plan, max_retries)


class ProxyGPT54ActionJudge(BaseJudge):
    """Action plan quality judge using GPT-5.4 via LLM proxy."""

    def __init__(self):
        self.client = _get_client()
        self.model = os.getenv("LLM_PROXY_GPT54_MODEL", "gpt-5.4")
        print(f"✅ ProxyGPT54ActionJudge initialized (model: {self.model})")

    @staticmethod
    def _format_action_plan(action_plan: dict) -> str:
        lines = [f"Overall Instruction: {action_plan.get('overall_instruction', 'N/A')}",
                 f"\nActions ({len(action_plan.get('actions', []))}):"]
        for i, a in enumerate(action_plan.get("actions", []), 1):
            lines.append(f"  {i}. {a.get('action_id','?')} (priority: {a.get('priority','?')})")
        return "\n".join(lines)

    @staticmethod
    def _do_judge(self_obj, original_image, user_prompt, predicted_plan, max_retries):
        image_b64 = _encode(original_image)
        formatted_plan = ProxyGPT54ActionJudge._format_action_plan(predicted_plan)
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

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]}]

        try:
            scores = _call_proxy(self_obj.client, self_obj.model, messages, max_tokens=600, max_retries=max_retries)
            for k in ["relevance", "theme_style_focus", "completeness", "efficiency", "correctness",
                      "reasoning_conciseness", "reasoning_completeness", "reasoning_specificity",
                      "overall_action_quality", "overall_reasoning_quality", "overall_score"]:
                scores[k] = float(scores[k])
            return scores
        except Exception as e:
            return {k: 0.0 for k in ["relevance", "theme_style_focus", "completeness", "efficiency", "correctness",
                                      "reasoning_conciseness", "reasoning_completeness", "reasoning_specificity",
                                      "overall_action_quality", "overall_reasoning_quality", "overall_score"]} | \
                   {"explanation": f"Error: {e}", "error": str(e)}

    def judge_single_edit(self, original_image, generated_image, instruction, max_retries=3):
        return ProxyGPT54Judge().judge_single_edit(original_image, generated_image, instruction, max_retries)

    def judge_action_plan(self, original_image, user_prompt, predicted_plan, teacher_plan=None, max_retries=3):
        return self._do_judge(self, original_image, user_prompt, predicted_plan, max_retries)
