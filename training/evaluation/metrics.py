#!/usr/bin/env python3
"""
Evaluation Metrics for Image Editing

Implements LPIPS, SSIM, PSNR, CLIP scores, and GPT-4o judge for evaluating editing quality.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Union, Optional


class MetricsCalculator:
    """Calculate evaluation metrics for image editing"""
    
    def __init__(self, device="cuda", use_gpt_judge=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_gpt_judge = use_gpt_judge
        
        # Try to load LPIPS
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            # Ensure LPIPS uses float32
            self.lpips_model = self.lpips_model.float()
            self.lpips_available = True
        except ImportError:
            print("⚠️  LPIPS not available (pip install lpips)")
            self.lpips_available = False
        
        # Try to load CLIP
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_available = True
        except ImportError:
            print("⚠️  CLIP not available (pip install git+https://github.com/openai/CLIP.git)")
            self.clip_available = False
        
        # Initialize GPT-4o judge
        if self.use_gpt_judge:
            try:
                from .gpt_judge import GPT4oJudge
                self.gpt_judge = GPT4oJudge()
                self.gpt_judge_available = True
            except Exception as e:
                print(f"⚠️  GPT-4o Judge not available: {e}")
                self.gpt_judge_available = False
        else:
            self.gpt_judge_available = False
    
    def compute_lpips(self, pred_imgs: torch.Tensor, target_imgs: torch.Tensor) -> float:
        """Compute LPIPS perceptual distance
        
        Args:
            pred_imgs: Predicted images [B, 3, H, W] in range [-1, 1]
            target_imgs: Target images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Mean LPIPS distance
        """
        if not self.lpips_available:
            return 0.0
        
        with torch.no_grad():
            # Convert to float32 for LPIPS (it doesn't support bfloat16)
            pred_imgs_float = pred_imgs.float()
            target_imgs_float = target_imgs.float()
            lpips_values = self.lpips_model(pred_imgs_float, target_imgs_float)
            return lpips_values.mean().item()
    
    def compute_ssim(self, pred_imgs: torch.Tensor, target_imgs: torch.Tensor) -> float:
        """Compute SSIM (Structural Similarity Index)
        
        Args:
            pred_imgs: Predicted images [B, 3, H, W] in range [-1, 1]
            target_imgs: Target images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Mean SSIM score (higher is better)
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to numpy and [0, 1] range
        # Must convert to float32 first because numpy doesn't support bfloat16
        pred_np = ((pred_imgs.cpu().float().numpy() + 1) / 2).transpose(0, 2, 3, 1)
        target_np = ((target_imgs.cpu().float().numpy() + 1) / 2).transpose(0, 2, 3, 1)
        
        ssim_values = []
        for pred, target in zip(pred_np, target_np):
            ssim_val = ssim(
                target, pred,
                multichannel=True,
                data_range=1.0,
                channel_axis=2
            )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def compute_psnr(self, pred_imgs: torch.Tensor, target_imgs: torch.Tensor) -> float:
        """Compute PSNR (Peak Signal-to-Noise Ratio)
        
        Args:
            pred_imgs: Predicted images [B, 3, H, W] in range [-1, 1]
            target_imgs: Target images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Mean PSNR value (higher is better)
        """
        # Compute MSE
        mse = F.mse_loss(pred_imgs, target_imgs, reduction='none')
        mse = mse.mean(dim=[1, 2, 3])  # Mean per image
        
        # PSNR formula (for range [-1, 1], max value is 2)
        psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)
        
        return psnr.mean().item()
    
    def compute_clip_score(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        texts: List[str]
    ) -> float:
        """Compute CLIP alignment score between images and texts
        
        Args:
            images: Images (tensor or PIL Images)
            texts: Text prompts
            
        Returns:
            Mean CLIP similarity score
        """
        if not self.clip_available:
            return 0.0
        
        # Preprocess images
        if isinstance(images, torch.Tensor):
            # Convert from [-1, 1] to [0, 1] and to PIL
            images = ((images + 1) / 2).clamp(0, 1)
            images = [transforms.ToPILImage()(img) for img in images.cpu()]
        
        # Truncate texts to fit CLIP's 77 token limit
        # Use first ~50 words as a reasonable approximation (CLIP uses BPE, so this is conservative)
        truncated_texts = []
        for text in texts:
            words = text.split()
            if len(words) > 50:
                truncated_text = ' '.join(words[:50]) + '...'
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)
        
        # Process through CLIP
        with torch.no_grad():
            import clip
            
            # Preprocess images and text
            image_inputs = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
            text_inputs = clip.tokenize(truncated_texts, truncate=True).to(self.device)
            
            # Compute features
            image_features = self.clip_model.encode_image(image_inputs)
            text_features = self.clip_model.encode_text(text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features * text_features).sum(dim=-1)
            
            return similarity.mean().item()
    
    def compute_gpt_judge_score(
        self,
        pred_img: Image.Image,
        original_img: Image.Image,
        instruction: str
    ) -> Dict[str, float]:
        """
        Compute GPT-4o judge score for image editing quality.
        
        Args:
            pred_img: Predicted/generated image (PIL Image)
            original_img: Original image before editing (PIL Image)
            instruction: Editing instruction text
            
        Returns:
            Dict with judge scores and reasoning
        """
        if not self.gpt_judge_available:
            return {
                "gpt_judge_overall": 0.0,
                "gpt_judge_instruction_following": 0.0,
                "gpt_judge_visual_quality": 0.0,
                "gpt_judge_transformation_strength": 0.0,
                "gpt_judge_coherence": 0.0,
                "gpt_judge_semantic_accuracy": 0.0,
                "gpt_judge_technical_execution": 0.0,
                "gpt_judge_reasoning": "GPT judge not available"
            }
        
        try:
            result = self.gpt_judge.judge_single_edit(
                original_img,
                pred_img,
                instruction
            )
            
            return {
                "gpt_judge_overall": result.get("overall_image_score", 0.0),
                "gpt_judge_instruction_following": result.get("instruction_following", 0.0),
                "gpt_judge_visual_quality": result.get("visual_quality", 0.0),
                "gpt_judge_transformation_strength": result.get("transformation_strength", 0.0),
                "gpt_judge_coherence": result.get("coherence", 0.0),
                "gpt_judge_semantic_accuracy": result.get("semantic_accuracy", 0.0),
                "gpt_judge_technical_execution": result.get("technical_execution", 0.0),
                "gpt_judge_reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            print(f"⚠️  GPT judge error: {e}")
            return {
                "gpt_judge_overall": 0.0,
                "gpt_judge_instruction_following": 0.0,
                "gpt_judge_visual_quality": 0.0,
                "gpt_judge_transformation_strength": 0.0,
                "gpt_judge_coherence": 0.0,
                "gpt_judge_semantic_accuracy": 0.0,
                "gpt_judge_technical_execution": 0.0,
                "gpt_judge_reasoning": f"Error: {str(e)}"
            }
    
    def compute_all_metrics(
        self,
        pred_imgs: torch.Tensor,
        target_imgs: torch.Tensor,
        instructions: List[str] = None,
        original_imgs: Optional[List[Image.Image]] = None,
        pred_imgs_pil: Optional[List[Image.Image]] = None,
        target_imgs_pil: Optional[List[Image.Image]] = None
    ) -> Dict[str, float]:
        """Compute all available metrics
        
        Args:
            pred_imgs: Predicted images (tensors)
            target_imgs: Target images (tensors)
            instructions: Optional text instructions
            original_imgs: Optional list of original PIL images (for GPT judge)
            pred_imgs_pil: Optional list of predicted PIL images (for GPT judge)
            target_imgs_pil: Optional list of target PIL images (for GPT judge)
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        # Perceptual metrics
        if self.lpips_available:
            metrics['lpips'] = self.compute_lpips(pred_imgs, target_imgs)
        
        # Structural metrics
        try:
            metrics['ssim'] = self.compute_ssim(pred_imgs, target_imgs)
        except Exception as e:
            print(f"⚠️  SSIM computation failed: {e}")
        
        try:
            metrics['psnr'] = self.compute_psnr(pred_imgs, target_imgs)
        except Exception as e:
            print(f"⚠️  PSNR computation failed: {e}")
        
        # CLIP scores
        if self.clip_available and instructions is not None:
            metrics['clip_score'] = self.compute_clip_score(pred_imgs, instructions)
        
        # GPT-4o Judge (if images provided and enabled)
        if (self.gpt_judge_available and 
            original_imgs is not None and 
            pred_imgs_pil is not None and 
            instructions is not None):
            
            # Judge only the first sample (batch size should be 1 for evaluation)
            if len(pred_imgs_pil) > 0:
                instruction = instructions[0] if isinstance(instructions, list) else instructions
                gpt_scores = self.compute_gpt_judge_score(
                    pred_imgs_pil[0],
                    original_imgs[0],
                    instruction
                )
                metrics.update(gpt_scores)
        
        return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing MetricsCalculator...")
    
    calculator = MetricsCalculator()
    
    # Create dummy data
    pred_imgs = torch.randn(2, 3, 256, 256)
    target_imgs = pred_imgs + torch.randn_like(pred_imgs) * 0.1
    instructions = ["Test instruction 1", "Test instruction 2"]
    
    # Compute metrics
    metrics = calculator.compute_all_metrics(pred_imgs, target_imgs, instructions)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n✅ Metrics test complete!")

