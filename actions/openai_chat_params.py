"""Azure OpenAI chat payload helpers (no training imports)."""

from __future__ import annotations

import os
from typing import Any, Dict


def model_uses_max_completion_tokens(model: str) -> bool:
    """Azure OpenAI o-series / GPT-5+ often require max_completion_tokens instead of max_tokens."""
    m = (model or "").lower().strip()
    if os.getenv("AZURE_OPENAI_USE_MAX_COMPLETION_TOKENS", "").lower() in ("1", "true", "yes"):
        return True
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    return False


def apply_completion_token_params(payload: Dict[str, Any], model: str, max_tokens: int) -> None:
    """Set max_tokens or max_completion_tokens on a chat completions payload (mutates payload)."""
    payload.pop("max_tokens", None)
    payload.pop("max_completion_tokens", None)
    if model_uses_max_completion_tokens(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
