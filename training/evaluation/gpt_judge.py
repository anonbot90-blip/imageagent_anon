#!/usr/bin/env python3
"""
GPT-4o Vision Judge for Image Editing Quality Evaluation

Uses Azure OpenAI GPT-4o to evaluate:
- How well the edit follows the instruction
- Transformation strength and completeness
- Semantic accuracy and coherence
- Technical execution quality
- Overall visual quality

NOTE: No ground truth comparison - models evaluated independently
"""

import os
import json
import base64
import time
from io import BytesIO
from typing import Dict, Tuple, Optional, List
from PIL import Image
import requests


class GPT4oJudge:
    """Judge image editing quality using GPT-4o Vision"""
    
    def __init__(self, azure_config: Optional[Dict] = None):
        """
        Initialize GPT-4o judge with Azure credentials.
        
        Args:
            azure_config: Dict with 'azure_endpoint', 'api_key', 'api_version', 'model'
                         If None, uses hardcoded credentials from config1.py
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
        
        print(f"✅ GPT-4o Judge initialized (model: {self.model})")
    
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _create_judge_prompt(self, instruction: str) -> str:
        """Create the judging prompt WITHOUT ground truth comparison"""
        return f"""You are an expert image editing quality evaluator.

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
    
    def judge_single_edit(
        self,
        original_image: Image.Image,
        generated_image: Image.Image,
        instruction: str,
        max_retries: int = 3
    ) -> Dict:
        """
        Judge a single image edit WITHOUT ground truth comparison.
        
        Args:
            original_image: Original image before editing
            generated_image: Generated edit
            instruction: Editing instruction
            max_retries: Number of retries on failure
            
        Returns:
            Dict with scores and reasoning
        """
        # Encode images to base64
        original_b64 = self._encode_image_base64(original_image)
        generated_b64 = self._encode_image_base64(generated_image)
        
        # Create prompt
        prompt = self._create_judge_prompt(instruction)
        
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
                                "url": f"data:image/png;base64,{original_b64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{generated_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
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
                    timeout=30
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                scores = json.loads(content)
                
                # Validate scores
                required_keys = [
                    "instruction_following", 
                    "visual_quality", 
                    "transformation_strength", 
                    "coherence", 
                    "semantic_accuracy", 
                    "technical_execution", 
                    "overall_image_score", 
                    "reasoning"
                ]
                if not all(k in scores for k in required_keys):
                    raise ValueError(f"Missing required keys in response: {scores.keys()}")
                
                return scores
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  GPT-4o judge attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ GPT-4o judge failed after {max_retries} attempts: {e}")
                    # Return default scores on failure
                    return {
                        "instruction_following": 0.0,
                        "visual_quality": 0.0,
                        "transformation_strength": 0.0,
                        "coherence": 0.0,
                        "semantic_accuracy": 0.0,
                        "technical_execution": 0.0,
                        "overall_image_score": 0.0,
                        "reasoning": f"Evaluation failed: {str(e)}",
                        "error": str(e)
                    }
    
    def judge_batch(
        self,
        samples: List[Tuple[Image.Image, Image.Image, str]]
    ) -> List[Dict]:
        """
        Judge a batch of edits WITHOUT ground truth.
        
        Args:
            samples: List of (original, generated, instruction) tuples
            
        Returns:
            List of score dictionaries
        """
        results = []
        for i, (orig, gen, inst) in enumerate(samples, 1):
            print(f"  Judging sample {i}/{len(samples)}...")
            scores = self.judge_single_edit(orig, gen, inst)
            results.append(scores)
        
        return results


if __name__ == "__main__":
    # Test GPT-4o judge
    print("Testing GPT-4o Judge...")
    
    judge = GPT4oJudge()
    
    # Create dummy test images
    original = Image.new('RGB', (256, 256), color='red')
    generated = Image.new('RGB', (256, 256), color='blue')
    
    instruction = "Change the color from red to green"
    
    print("\nEvaluating test image...")
    result = judge.judge_single_edit(original, generated, instruction)
    
    print("\n✅ GPT-4o Judge Result:")
    print(f"  Instruction Following: {result['instruction_following']}/10")
    print(f"  Visual Quality: {result['visual_quality']}/10")
    print(f"  Transformation Strength: {result['transformation_strength']}/10")
    print(f"  Coherence: {result['coherence']}/10")
    print(f"  Semantic Accuracy: {result['semantic_accuracy']}/10")
    print(f"  Technical Execution: {result['technical_execution']}/10")
    print(f"  Overall Score: {result['overall_image_score']}/10")
    print(f"  Reasoning: {result['reasoning']}")
    
    if 'error' not in result:
        print("\n✅ GPT-4o Judge test complete!")
    else:
        print(f"\n⚠️  Test completed with errors: {result['error']}")



