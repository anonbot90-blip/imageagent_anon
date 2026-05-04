"""
Claude vision planner on AWS Bedrock — same JSON action-plan output as GPT4oPlanner.
Requires AWS credentials with Bedrock invoke access (e.g. export AWS_* from CLAUDE_OPUS_* in credentials.sh).
"""

from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from PIL import Image

from actions.gpt4o_planner import build_theme_style_planning_prompt


class BedrockClaudePlanner:
    """Theme/style action planner using Claude on Bedrock (vision + JSON plan)."""

    def __init__(
        self,
        action_library_path: Optional[str] = None,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.model_id = model_id or os.getenv(
            "CLAUDE_OPUS_BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-6-v1"
        )
        self.region = region or os.getenv("AWS_DEFAULT_REGION") or os.getenv(
            "CLAUDE_OPUS_AWS_REGION", "us-east-2"
        )
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

        if action_library_path is None:
            action_library_path = Path(__file__).parent / "action_library_v2.json"
        with open(action_library_path, "r") as f:
            self.action_library = json.load(f)

        print(f"✅ Bedrock Claude planner initialized (model: {self.model_id}, region: {self.region})")

    def _encode_image_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
        image_base64 = self._encode_image_base64(image)
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

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        }

        for attempt in range(max_retries):
            try:
                response = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                )
                raw = json.loads(response["body"].read())
                content = raw["content"][0]["text"]
                content = content.strip()
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
                    print(f"⚠️  Bedrock Claude attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2**attempt)
                else:
                    print(f"❌ Bedrock Claude failed after {max_retries} attempts: {e}")
                    return {
                        "overall_instruction": user_prompt,
                        "actions": [],
                        "error": str(e),
                    }

        return {"overall_instruction": user_prompt, "actions": [], "error": "Unknown error"}
