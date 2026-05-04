"""
Gemini vision planner — same JSON action-plan output as GPT4oPlanner.
Set GOOGLE_API_KEY or GEMINI_API_KEY (see credentials.sh).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from actions.gpt4o_planner import build_theme_style_planning_prompt


class GeminiPlanner:
    """Theme/style action planner using Google Gemini multimodal."""

    def __init__(
        self,
        action_library_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY (e.g. export the required environment variables)."
            )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("pip install google-generativeai") from e

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model_name or os.getenv("GEMINI_PLANNER_MODEL", "gemini-2.5-flash")
        self._model = genai.GenerativeModel(self.model_name)

        if action_library_path is None:
            action_library_path = Path(__file__).parent / "action_library_v2.json"
        with open(action_library_path, "r") as f:
            self.action_library = json.load(f)

        print(f"✅ Gemini planner initialized (model: {self.model_name})")

    def predict_action_plan(
        self,
        image_path: str,
        user_prompt: str,
        analysis: Dict[str, Any] = None,
        max_actions: int = 5,
        temperature: float = 0.1,
        return_reasoning: bool = False,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        system_prompt = build_theme_style_planning_prompt(self.action_library, max_actions=max_actions)

        analysis_text = ""
        if analysis:
            analysis_parts = []
            if "scene_type" in analysis:
                analysis_parts.append(f"Scene: {analysis['scene_type']}")
            if "objects" in analysis and analysis["objects"]:
                objects_list = [
                    obj["name"] if isinstance(obj, dict) else obj for obj in analysis["objects"][:10]
                ]
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
                analysis_text = "\n\nIMAGE ANALYSIS:\n" + "\n".join(f"- {p}" for p in analysis_parts)

        user_text = f"{system_prompt}\n\nUser Request: {user_prompt}{analysis_text}"

        for attempt in range(max_retries):
            try:
                try:
                    response = self._model.generate_content(
                        [user_text, image],
                        generation_config=self._genai.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=8192,
                            response_mime_type="application/json",
                        ),
                    )
                except Exception:
                    response = self._model.generate_content(
                        [user_text, image],
                        generation_config=self._genai.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=8192,
                        ),
                    )
                content = response.text.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                action_plan = json.loads(content)
                if "actions" not in action_plan:
                    raise ValueError("Response missing 'actions' field")
                if "overall_instruction" not in action_plan:
                    action_plan["overall_instruction"] = user_prompt
                for action in action_plan["actions"]:
                    if "action_id" not in action:
                        raise ValueError("Action missing 'action_id' field")
                    if "priority" not in action:
                        raise ValueError("Action missing 'priority' field")
                    if "parameters" not in action:
                        action["parameters"] = {}
                return action_plan

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Gemini planner attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2**attempt)
                else:
                    print(f"❌ Gemini planner failed after {max_retries} attempts: {e}")
                    return {
                        "overall_instruction": user_prompt,
                        "actions": [],
                        "error": str(e),
                    }

        return {"overall_instruction": user_prompt, "actions": [], "error": "Unknown error"}
