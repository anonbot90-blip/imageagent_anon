"""
Reward Model Evaluator for ImageAgent
Evaluates image transformations across multiple quality dimensions
Includes both objective metrics (LPIPS, SSIM, PSNR, CLIP) and VLM-based subjective evaluation
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
from datetime import datetime
import time

# Import the VLM analyzer (we'll reuse its infrastructure)
from src.image_analyzer_qwen3 import ImageAnalyzerQwen3


class RewardModelEvaluator:
    """
    Evaluates image transformation quality using:
    1. Objective metrics (LPIPS, SSIM, PSNR, CLIP)
    2. Vision-Language Model subjective evaluation (7 dimensions, 1-5 scale)
    
    VLM Dimensions:
    1. Action Plan Quality
    2. Plan Reasoning
    3. Final Image Quality
    4. Adherence to Plan
    5. Adherence to Prompt
    6. Overall Quality
    7. Reasoning Quality
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", existing_model=None):
        """
        Initialize the reward model evaluator
        
        Args:
            model_name: VLM model to use for evaluation
            existing_model: Optional pre-loaded ImageAnalyzerQwen3 instance to reuse
        """
        self.model_name = model_name
        
        # Reuse existing model if provided (memory efficient)
        if existing_model is not None:
            print(f"♻️  Reusing existing model for reward evaluation: {model_name}")
            self.model = existing_model
            self.owns_model = False
        else:
            print(f"🏆 Loading reward model: {model_name}")
            self.model = ImageAnalyzerQwen3(model_name=model_name)
            self.owns_model = True
        
        # Check if this is a "Thinking" model
        self.is_thinking = "Thinking" in model_name
        
        # Initialize objective metrics calculators
        self._init_objective_metrics()
        
        print("✅ Reward model evaluator initialized")
    
    def _init_objective_metrics(self):
        """Initialize models for computing objective metrics"""
        # LPIPS
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.model.device).eval()
            self.lpips_model = self.lpips_model.float()
            self.lpips_available = True
        except ImportError:
            print("⚠️  LPIPS not available (pip install lpips)")
            self.lpips_available = False
        
        # CLIP
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.model.device)
            self.clip_available = True
        except ImportError:
            print("⚠️  CLIP not available (pip install ftfy regex tqdm)")
            self.clip_available = False
    
    def evaluate_transformation(
        self,
        original_image_path: str,
        edited_image_path: str,
        user_prompt: str,
        action_plan: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an image transformation across multiple dimensions
        
        Args:
            original_image_path: Path to original image
            edited_image_path: Path to edited image
            user_prompt: User's transformation request
            action_plan: The action plan that was executed
            analysis: Optional scene analysis
            
        Returns:
            Dictionary with VLM scores, objective metrics, and metadata
        """
        print(f"🏆 Evaluating transformation: {os.path.basename(edited_image_path)}")
        
        start_time = time.time()
        
        # Load images
        original_image = Image.open(original_image_path)
        edited_image = Image.open(edited_image_path)
        
        # Step 1: Compute objective metrics
        print("  📊 Computing objective metrics...")
        objective_metrics = self._compute_objective_metrics(
            original_image, edited_image, user_prompt
        )
        
        # Display objective metrics
        self._print_objective_metrics(objective_metrics)
        
        # Step 2: Build evaluation prompt (including objective metrics)
        evaluation_prompt = self._build_evaluation_prompt(
            user_prompt, action_plan, analysis, objective_metrics
        )
        
        # Step 3: Prepare messages for the model (multi-image input)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ORIGINAL IMAGE:"},
                    {"type": "image", "image": original_image_path},
                    {"type": "text", "text": "\nEDITED IMAGE:"},
                    {"type": "image", "image": edited_image_path},
                    {"type": "text", "text": f"\n{evaluation_prompt}"}
                ]
            }
        ]
        
        # Process with model
        text = self.model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs = [original_image, edited_image]
        inputs = self.model.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate evaluation
        print("  🤔 Generating evaluation scores...")
        with torch.no_grad():
            generated_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.model.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle Thinking models)
            clean_response = response.strip()
            
            if self.is_thinking and not clean_response.startswith('{'):
                import re
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    clean_response = json_match.group(0)
                    print("  ✓ Extracted JSON from reasoning text")
            
            # Remove markdown code blocks if present
            clean_response = clean_response.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            scores = json.loads(clean_response)
            
            # Validate structure
            if 'scores' not in scores:
                raise ValueError("Response missing 'scores' key")
            
            evaluation_time = time.time() - start_time
            
            # Add metadata including objective metrics
            result = {
                "scores": scores.get('scores', scores),
                "objective_metrics": objective_metrics,
                "metadata": {
                    "reward_model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_time_seconds": round(evaluation_time, 2),
                    "original_image": os.path.basename(original_image_path),
                    "edited_image": os.path.basename(edited_image_path)
                }
            }
            
            print(f"  ✓ Evaluation complete ({evaluation_time:.1f}s)")
            self._print_scores(result['scores'])
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse JSON response: {e}")
            print(f"  Raw response: {response[:200]}...")
            
            # Return fallback scores
            return self._create_fallback_scores(response, time.time() - start_time)
        except Exception as e:
            print(f"  ⚠️  Error during evaluation: {e}")
            return self._create_fallback_scores(str(e), time.time() - start_time)
    
    def _build_evaluation_prompt(
        self,
        user_prompt: str,
        action_plan: Dict[str, Any],
        analysis: Optional[Dict[str, Any]],
        objective_metrics: Dict[str, Any]
    ) -> str:
        """Build the evaluation prompt for the reward model including objective metrics"""
        
        # Format action plan
        actions_str = json.dumps(action_plan.get('actions', []), indent=2)
        overall_instruction = action_plan.get('overall_instruction', 'N/A')
        reasoning = action_plan.get('reasoning', 'N/A')
        
        # Format analysis (optional)
        analysis_str = ""
        if analysis:
            analysis_str = f"""
SCENE ANALYSIS:
- Style: {analysis.get('basic_info', {}).get('style', 'N/A')}
- Setting: {analysis.get('basic_info', {}).get('setting', 'N/A')}
- Lighting: {analysis.get('technical_details', {}).get('lighting', {}).get('type', 'N/A')}
- Color Palette: {analysis.get('technical_details', {}).get('color_palette', {}).get('dominant_colors', [])}
"""
        
        # Format objective metrics
        lpips = objective_metrics.get('lpips', 'N/A')
        ssim = objective_metrics.get('ssim', 'N/A')
        psnr = objective_metrics.get('psnr', 'N/A')
        clip_score = objective_metrics.get('clip_score', 'N/A')
        
        # Format metrics with proper display
        lpips_str = f"{lpips:.3f}" if isinstance(lpips, float) else lpips
        ssim_str = f"{ssim:.3f}" if isinstance(ssim, float) else ssim
        psnr_str = f"{psnr:.1f} dB" if isinstance(psnr, float) else psnr
        clip_str = f"{clip_score:.3f}" if isinstance(clip_score, float) else clip_score
        
        objective_metrics_str = f"""
OBJECTIVE QUALITY METRICS:
- LPIPS (perceptual distance): {lpips_str} [lower is better, <0.2 excellent, >0.6 poor]
- SSIM (structural similarity): {ssim_str} [higher is better, >0.9 excellent, <0.5 poor]
- PSNR (pixel quality): {psnr_str} [higher is better, >30dB excellent, <20dB poor]
- CLIP Score (text-image alignment): {clip_str} [higher is better, >0.85 excellent, <0.65 poor]
"""
        
        prompt = f"""You are an expert image transformation evaluator. Evaluate this image transformation across 7 dimensions.

Consider BOTH the objective metrics provided AND your visual assessment of the images.

USER'S REQUEST:
"{user_prompt}"

ACTION PLAN EXECUTED:
Overall Instruction: {overall_instruction}

Reasoning:
{reasoning}

Actions:
{actions_str}
{analysis_str}
{objective_metrics_str}

EVALUATION CRITERIA (Score 1-5 for each):

1. **action_plan_quality** (1-5): Are the chosen actions appropriate for the transformation?
   - 5: Perfect action selection, highly relevant
   - 3: Adequate actions, some room for improvement
   - 1: Poor action choices, not suitable

2. **plan_reasoning** (1-5): Is the action sequence logical and well-justified?
   - 5: Excellent reasoning, optimal sequencing
   - 3: Reasonable logic, acceptable order
   - 1: Poor reasoning, illogical sequence

3. **final_image_quality** (1-5): Technical quality of the edited image
   - 5: Excellent quality, no artifacts, highly coherent
   - 3: Good quality, minor issues
   - 1: Poor quality, major artifacts or incoherence

4. **adherence_to_plan** (1-5): How well did the editing follow the action plan?
   - 5: Perfect execution of all planned actions
   - 3: Partial execution, some actions visible
   - 1: Plan not followed, actions not visible

5. **adherence_to_prompt** (1-5): Does the result match the user's request?
   - 5: Perfect match, exceeds expectations
   - 3: Acceptable match, captures main intent
   - 1: Poor match, misses user's request

6. **overall_quality** (1-5): Holistic assessment of the transformation
   - 5: Outstanding transformation, publication-ready
   - 3: Good transformation, acceptable result
   - 1: Poor transformation, needs major improvement

7. **reasoning_quality** (1-5): Quality of the planner's written reasoning and explanation
   
   Evaluate the written reasoning provided in the action plan:
   - Does it reference specific image elements (objects, style, composition)?
   - Are action choices clearly justified with causal explanations?
   - Is there logical flow and evidence of image analysis?
   - Is reasoning specific to this image (not generic)?
   - If per-action reasoning is present, evaluate each action's justification
   
   Scoring:
   - 5: Exceptional - Detailed, specific, grounded in image analysis, clear causal links
        Example: "The wet urban street must be replaced with beach sand first, as this 
        establishes the tropical foundation. Given the glass skyscrapers, a complete 
        location transformation is needed rather than partial overlay."
   - 4: Strong - Good detail, references image content, explains most choices clearly
        Example: "Transform urban setting to beach. Need to add palm trees and ocean 
        to create tropical atmosphere."
   - 3: Adequate - Basic reasoning, explains main actions but lacks specificity
        Example: "Change the scene to a beach with typical beach elements."
   - 2: Weak - Vague or generic, minimal explanation, doesn't reference image details
        Example: "Make it look like a beach scene."
   - 1: Poor - No meaningful reasoning, illogical, or missing entirely
        Example: "" (empty) or "Do various transformations."

Output ONLY valid JSON in this exact format:
{{
  "scores": {{
    "action_plan_quality": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }},
    "plan_reasoning": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }},
    "reasoning_quality": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }},
    "final_image_quality": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }},
    "adherence_to_plan": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }},
    "adherence_to_prompt": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }},
    "overall_quality": {{
      "score": <1-5>,
      "reasoning": "<brief explanation>"
    }}
  }}
}}

CRITICAL: Output ONLY the JSON object above. No explanations before or after."""
        
        return prompt
    
    def _compute_objective_metrics(
        self,
        original_image: Image.Image,
        edited_image: Image.Image,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Compute objective quality metrics
        
        Returns:
            Dictionary with LPIPS, SSIM, PSNR, CLIP scores
        """
        metrics = {}
        
        # Ensure images have the same dimensions (resize edited to match original)
        if original_image.size != edited_image.size:
            print(f"    ⚠️  Resizing edited image from {edited_image.size} to {original_image.size}")
            edited_image = edited_image.resize(original_image.size, Image.Resampling.LANCZOS)
        
        # Convert PIL images to tensors
        original_tensor = self._pil_to_tensor(original_image).unsqueeze(0).to(self.model.device)
        edited_tensor = self._pil_to_tensor(edited_image).unsqueeze(0).to(self.model.device)
        
        # 1. LPIPS (perceptual distance)
        if self.lpips_available:
            try:
                with torch.no_grad():
                    lpips_val = self.lpips_model(original_tensor.float(), edited_tensor.float())
                    metrics['lpips'] = float(lpips_val.mean().item())
            except Exception as e:
                print(f"    ⚠️  LPIPS computation failed: {e}")
                metrics['lpips'] = None
        else:
            metrics['lpips'] = None
        
        # 2. SSIM (structural similarity)
        try:
            from skimage.metrics import structural_similarity as ssim
            # Convert to numpy [0, 1]
            orig_np = ((original_tensor.cpu().float().numpy() + 1) / 2).transpose(0, 2, 3, 1)[0]
            edit_np = ((edited_tensor.cpu().float().numpy() + 1) / 2).transpose(0, 2, 3, 1)[0]
            ssim_val = ssim(orig_np, edit_np, multichannel=True, data_range=1.0, channel_axis=2)
            metrics['ssim'] = float(ssim_val)
        except Exception as e:
            print(f"    ⚠️  SSIM computation failed: {e}")
            metrics['ssim'] = None
        
        # 3. PSNR (peak signal-to-noise ratio)
        try:
            mse = torch.nn.functional.mse_loss(original_tensor, edited_tensor)
            psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)
            metrics['psnr'] = float(psnr.item())
        except Exception as e:
            print(f"    ⚠️  PSNR computation failed: {e}")
            metrics['psnr'] = None
        
        # 4. CLIP Score (text-image alignment)
        if self.clip_available:
            try:
                import clip
                # Preprocess edited image for CLIP
                clip_image = self.clip_preprocess(edited_image).unsqueeze(0).to(self.model.device)
                # Tokenize text
                text = clip.tokenize([user_prompt], truncate=True).to(self.model.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(clip_image)
                    text_features = self.clip_model.encode_text(text)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute cosine similarity
                    clip_score = (image_features @ text_features.T).squeeze().item()
                    metrics['clip_score'] = float(clip_score)
            except Exception as e:
                print(f"    ⚠️  CLIP computation failed: {e}")
                metrics['clip_score'] = None
        else:
            metrics['clip_score'] = None
        
        return metrics
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor in range [-1, 1]"""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        return transform(image)
    
    def _print_objective_metrics(self, metrics: Dict[str, Any]) -> None:
        """Pretty print objective metrics"""
        lpips = metrics.get('lpips')
        ssim = metrics.get('ssim')
        psnr = metrics.get('psnr')
        clip_score = metrics.get('clip_score')
        
        lpips_str = f"{lpips:.3f}" if lpips is not None else "N/A"
        ssim_str = f"{ssim:.3f}" if ssim is not None else "N/A"
        psnr_str = f"{psnr:.1f} dB" if psnr is not None else "N/A"
        clip_str = f"{clip_score:.3f}" if clip_score is not None else "N/A"
        
        print(f"    • LPIPS: {lpips_str}")
        print(f"    • SSIM: {ssim_str}")
        print(f"    • PSNR: {psnr_str}")
        print(f"    • CLIP: {clip_str}")
    
    def _print_scores(self, scores: Dict[str, Any]) -> None:
        """Pretty print the evaluation scores"""
        print("\n  📊 Evaluation Scores:")
        for dimension, data in scores.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                # Color code: 5=green, 4=cyan, 3=yellow, 2-1=red
                if score >= 5:
                    color = '\033[0;32m'  # Green
                elif score >= 4:
                    color = '\033[0;36m'  # Cyan
                elif score >= 3:
                    color = '\033[1;33m'  # Yellow
                else:
                    color = '\033[0;31m'  # Red
                reset = '\033[0m'
                
                print(f"    {color}• {dimension}: {score}/5{reset}")
    
    def _create_fallback_scores(self, error_msg: str, eval_time: float) -> Dict[str, Any]:
        """Create fallback scores when evaluation fails"""
        return {
            "scores": {
                "action_plan_quality": {"score": 0, "reasoning": "Evaluation failed"},
                "plan_reasoning": {"score": 0, "reasoning": "Evaluation failed"},
                "reasoning_quality": {"score": 0, "reasoning": "Evaluation failed"},
                "final_image_quality": {"score": 0, "reasoning": "Evaluation failed"},
                "adherence_to_plan": {"score": 0, "reasoning": "Evaluation failed"},
                "adherence_to_prompt": {"score": 0, "reasoning": "Evaluation failed"},
                "overall_quality": {"score": 0, "reasoning": "Evaluation failed"}
            },
            "metadata": {
                "reward_model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "evaluation_time_seconds": round(eval_time, 2),
                "error": error_msg
            }
        }
    
    def __del__(self):
        """Cleanup when evaluator is destroyed"""
        if self.owns_model and hasattr(self, 'model'):
            del self.model





