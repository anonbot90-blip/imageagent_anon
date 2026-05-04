#!/usr/bin/env python3
"""
Judge registry — single dispatch point for all judge implementations.

Usage:
    from training.evaluation.judge_registry import get_judge, get_action_judge

    judge = get_judge("gpt54")          # image quality judge
    action_judge = get_action_judge("gemini")   # action plan judge
"""

from training.evaluation.judge_base import BaseJudge

VALID_JUDGES = ["gpt4o", "gpt54", "claude_opus", "gemini", "proxy_claude", "proxy_gpt54", "proxy_gemini"]


def get_judge(judge_id: str) -> BaseJudge:
    """Return an image quality judge for the given judge_id."""
    if judge_id == "gpt4o":
        from training.evaluation.gpt_judge import GPT4oJudge
        return GPT4oJudge()
    elif judge_id == "gpt54":
        from training.evaluation.gpt54_judge import GPT54Judge
        return GPT54Judge()
    elif judge_id == "claude_opus":
        from training.evaluation.claude_judge import ClaudeJudge
        return ClaudeJudge()
    elif judge_id == "gemini":
        from training.evaluation.gemini_judge import GeminiJudge
        return GeminiJudge()
    elif judge_id == "proxy_claude":
        from training.evaluation.proxy_claude_judge import ProxyClaudeJudge
        return ProxyClaudeJudge()
    elif judge_id == "proxy_gpt54":
        from training.evaluation.proxy_gpt54_judge import ProxyGPT54Judge
        return ProxyGPT54Judge()
    elif judge_id == "proxy_gemini":
        from training.evaluation.proxy_gemini_judge import ProxyGeminiJudge
        return ProxyGeminiJudge()
    else:
        raise ValueError(f"Unknown judge_id: '{judge_id}'. Valid options: {VALID_JUDGES}")


def get_action_judge(judge_id: str) -> BaseJudge:
    """Return an action plan judge for the given judge_id."""
    if judge_id == "gpt4o":
        from training.evaluation.gpt_action_judge import GPT4oActionJudge
        return GPT4oActionJudge()
    elif judge_id == "gpt54":
        from training.evaluation.gpt54_judge import GPT54ActionJudge
        return GPT54ActionJudge()
    elif judge_id == "claude_opus":
        from training.evaluation.claude_judge import ClaudeActionJudge
        return ClaudeActionJudge()
    elif judge_id == "gemini":
        from training.evaluation.gemini_judge import GeminiActionJudge
        return GeminiActionJudge()
    elif judge_id == "proxy_claude":
        from training.evaluation.proxy_claude_judge import ProxyClaudeActionJudge
        return ProxyClaudeActionJudge()
    elif judge_id == "proxy_gpt54":
        from training.evaluation.proxy_gpt54_judge import ProxyGPT54ActionJudge
        return ProxyGPT54ActionJudge()
    elif judge_id == "proxy_gemini":
        from training.evaluation.proxy_gemini_judge import ProxyGeminiActionJudge
        return ProxyGeminiActionJudge()
    else:
        raise ValueError(f"Unknown judge_id: '{judge_id}'. Valid options: {VALID_JUDGES}")
