#!/usr/bin/env python3
"""
BaseJudge abstract base class for all judge implementations.

All judges must implement:
  - judge_single_edit: evaluate image edit quality
  - judge_action_plan: evaluate action plan quality
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from PIL import Image


class BaseJudge(ABC):
    """Abstract base class for image editing quality judges."""

    @abstractmethod
    def judge_single_edit(
        self,
        original_image: Image.Image,
        generated_image: Image.Image,
        instruction: str,
        max_retries: int = 3,
    ) -> Dict:
        """
        Evaluate image edit quality.

        Args:
            original_image: Original image before editing
            generated_image: Generated edited image
            instruction: The editing instruction
            max_retries: Number of retries on failure

        Returns:
            Dict with keys:
              instruction_following, visual_quality, transformation_strength,
              coherence, semantic_accuracy, technical_execution,
              overall_image_score, reasoning
            (and optionally "error" on failure)
        """
        ...

    @abstractmethod
    def judge_action_plan(
        self,
        original_image: Image.Image,
        user_prompt: str,
        predicted_plan: Dict,
        teacher_plan: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> Dict:
        """
        Evaluate action plan quality.

        Args:
            original_image: Original image before editing
            user_prompt: User's editing request
            predicted_plan: Predicted action plan to evaluate
            teacher_plan: Optional teacher/reference plan for context
            max_retries: Number of retries on failure

        Returns:
            Dict with keys:
              relevance, theme_style_focus, completeness, efficiency, correctness,
              reasoning_conciseness, reasoning_completeness, reasoning_specificity,
              overall_action_quality, overall_reasoning_quality, overall_score,
              explanation
            (and optionally "error" on failure)
        """
        ...
