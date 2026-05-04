#!/usr/bin/env python3
"""
Image Editor Module using HiDream-E1
Performs instruction-based image editing using natural language prompts
"""

import os
import sys
import torch
from PIL import Image
from typing import Dict, List, Any, Optional
import json

# Add HiDream paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'HiDream-I1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'HiDream-E1'))

try:
    from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast, LlamaForCausalLM
    
    # Try to import HiDream components
    try:
        from diffusers import HiDreamImagePipeline
        from hi_diffusers import HiDreamImagePipeline as CustomHiDreamImagePipeline
        from hi_diffusers import HiDreamImageTransformer2DModel
        from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
        HIDREAM_AVAILABLE = True
    except ImportError:
        print("HiDream custom components not available, using standard diffusers")
        HIDREAM_AVAILABLE = False
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure dependencies are installed")
    HIDREAM_AVAILABLE = False


class ImageEditor:
    """Image editor using HiDream models for instruction-based editing"""
    
    def __init__(self, model_path: str = None, use_hidream_e1: bool = True):
        """Initialize the image editing model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_loaded = False
        
        print(f"🔧 Initializing Image Editor...")
        print(f"📱 Using device: {self.device}")
        
        # Try HiDream-E1 first, but fallback to working alternatives
        if use_hidream_e1:
            print("🎯 Attempting HiDream-E1 for instruction-based editing...")
            self._load_hidream_e1(model_path)
            
            # If HiDream-E1 fails, try HiDream-I1 as img2img
            if not self.model_loaded:
                print("🔄 HiDream-E1 failed, trying HiDream-I1 for img2img editing...")
                self._load_hidream_i1_for_editing()
                
            # If both fail, load a standard img2img model
            if not self.model_loaded:
                print("🔄 Loading standard Stable Diffusion for img2img editing...")
                self._load_standard_img2img()
        else:
            self._load_hidream_i1_for_editing()
    
    def _load_hidream_i1_for_editing(self):
        """Load HiDream-I1 model for img2img editing"""
        try:
            print("🎨 Loading HiDream-I1 for image-to-image editing...")
            
            # Model configuration for editing (using dev model for balance)
            MODEL_PREFIX = "HiDream-ai"
            LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            model_path = f"{MODEL_PREFIX}/HiDream-I1-Dev"
            
            # Load components
            tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME,
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            # Load HiDream pipeline
            self.pipe = CustomHiDreamImagePipeline.from_pretrained(
                model_path,
                tokenizer_4=tokenizer_4,
                text_encoder_4=text_encoder_4,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            # Configure for editing
            self.guidance_scale = 7.5
            self.num_inference_steps = 28
            self.strength = 0.8  # For img2img editing
            
            self.model_loaded = True
            print("✅ HiDream-I1 loaded successfully for editing!")
            
        except Exception as e:
            print(f"❌ Error loading HiDream-I1: {e}")
            self._fallback_to_mock()
    
    def _load_hidream_e1(self, model_path: str = None):
        """Load HiDream-E1 model for editing using the official implementation"""
        try:
            print(f"🎨 Loading HiDream-E1 for instruction-based image editing...")
            
            # Method 1: Load from HuggingFace directly (recommended)
            try:
                print("🔧 Loading HiDream-E1 from HuggingFace...")
                
                # Load the tokenizer and text encoder as shown in the documentation
                from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
                
                tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
                text_encoder_4 = LlamaForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    output_hidden_states=True,
                    output_attentions=True,
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
                
                # Since HiDreamImageEditingPipeline is not available in current diffusers,
                # we'll use a practical approach with Stable Diffusion img2img for actual editing
                print("⚠️  HiDreamImageEditingPipeline not available in current diffusers version")
                print("🔄 Using Stable Diffusion img2img as HiDream-E1 alternative...")
                
                # Load a high-quality img2img pipeline for editing
                from diffusers import StableDiffusionImg2ImgPipeline
                
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.bfloat16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                
                # Store the Llama text encoder for advanced prompting
                self.text_encoder_4 = text_encoder_4
                self.tokenizer_4 = tokenizer_4
                
                print("✅ Loaded Stable Diffusion img2img as HiDream-E1 alternative!")
                self.model_loaded = True
                return
                    
            except Exception as e1:
                print(f"⚠️  HuggingFace loading failed: {e1}")
            
            # Method 2: Try local model if available
            if model_path and os.path.exists(model_path):
                try:
                    print(f"🔧 Trying local HiDream-E1 from: {model_path}")
                    
                    # Load tokenizer and text encoder
                    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
                    text_encoder_4 = LlamaForCausalLM.from_pretrained(
                        "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        output_hidden_states=True,
                        output_attentions=True,
                        torch_dtype=torch.bfloat16,
                    ).to(self.device)
                    
                    # Try to load local pipeline
                    self.pipe = DiffusionPipeline.from_pretrained(
                        model_path,
                        tokenizer_4=tokenizer_4,
                        text_encoder_4=text_encoder_4,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    ).to(self.device)
                    
                    print("✅ Loaded local HiDream-E1!")
                    self.model_loaded = True
                    return
                    
                except Exception as e2:
                    print(f"⚠️  Local loading failed: {e2}")
            
            # If all methods fail, raise exception to trigger fallback
            raise Exception("All HiDream-E1 loading methods failed")
            
        except Exception as e:
            print(f"❌ HiDream-E1 loading failed: {e}")
            print("🔄 Will try fallback methods...")
            # Don't call _fallback_to_mock here, let the main init handle fallbacks
    
    def _load_standard_img2img(self):
        """Load a standard Stable Diffusion img2img model as final fallback"""
        try:
            print("🎨 Loading standard Stable Diffusion img2img model...")
            
            # Use a reliable img2img model
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.model_loaded = True
            print("✅ Standard Stable Diffusion img2img loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading standard img2img: {e}")
            self._fallback_to_mock()
    
    def _fallback_to_mock(self):
        """Fallback to mock editing"""
        print("🔄 Falling back to enhanced mock editing...")
        self.pipe = None
        self.model_loaded = False
    
    def edit_image(self, 
                   image_path: str, 
                   edit_prompt: str, 
                   output_path: str = None,
                   strength: float = 0.8,
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 28) -> str:
        """
        Edit an image based on a text prompt
        
        Args:
            image_path: Path to the input image
            edit_prompt: Text description of the desired edit
            output_path: Path to save the edited image
            strength: Strength of the edit (0.0 to 1.0)
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            
        Returns:
            Tuple of (path to edited image, formatted prompt sent to model)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load the input image
        input_image = Image.open(image_path).convert("RGB")
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'edited_images')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_edited.png")
        
        print(f"🖼️  Editing image: {os.path.basename(image_path)}")
        print(f"📝 Edit prompt: {edit_prompt}")
        print(f"💾 Output: {output_path}")
        
        # Track the formatted prompt sent to the model
        formatted_prompt = None
        
        if self.model_loaded and self.pipe is not None:
            try:
                print("🎨 Performing actual image editing...")
                
                # Check if we're using HiDream-E1 editing pipeline
                if hasattr(self.pipe, '__class__') and 'HiDreamImageEditing' in str(self.pipe.__class__):
                    # Use HiDream-E1 for instruction-based editing
                    print("🎨 Using HiDream-E1 instruction-based editing...")
                    
                    # Format prompt for HiDream-E1 (simplified wrapper - no duplication)
                    # Instruction appears once, not twice
                    formatted_prompt = f"Editing Instruction: {edit_prompt}. Maintain high quality, original composition and style."
                    
                    # Prepare image as shown in documentation - resize to 768x768
                    test_image = input_image.resize((768, 768))
                    
                    edited_image = self.pipe(
                        prompt=formatted_prompt,
                        negative_prompt="low resolution, blur",
                        image=test_image,
                        guidance_scale=5.0,  # As recommended in docs
                        image_guidance_scale=4.0,  # HiDream-E1 specific parameter
                        num_inference_steps=28,  # As recommended in docs
                        generator=torch.Generator(self.device).manual_seed(3)  # Use seed 3 as in docs
                    ).images[0]
                    
                elif hasattr(self.pipe, '__class__') and 'HiDream' in str(self.pipe.__class__):
                    # Use HiDream-I1 for text-to-image generation (not editing)
                    print("🎨 Using HiDream-I1 for text-to-image generation (not true editing)...")
                    
                    # Format prompt for generation
                    formatted_prompt = f"A high-quality image where {edit_prompt}, maintaining photorealistic style"
                    
                    edited_image = self.pipe(
                        prompt=formatted_prompt,
                        negative_prompt="low resolution, blur, distorted",
                        guidance_scale=7.0,
                        num_inference_steps=30,
                        generator=torch.Generator(self.device).manual_seed(42)
                    ).images[0]
                    
                # DISABLED: Img2Img fallback (generic Stable Diffusion)
                # We require HiDream-E1 for consistent editing quality
                # elif hasattr(self.pipe, '__class__') and 'Img2Img' in str(self.pipe.__class__):
                #     # Use standard img2img pipeline (our fallback)
                #     print("🎨 Using Img2Img pipeline for editing...")
                #     
                #     # Create instruction-based prompt for img2img
                #     formatted_prompt = f"A high-quality image where {edit_prompt}. Professional, detailed, sharp focus."
                #     
                #     edited_image = self.pipe(
                #         prompt=formatted_prompt,
                #         image=input_image,
                #         strength=strength,
                #         guidance_scale=guidance_scale,
                #         num_inference_steps=num_inference_steps,
                #         generator=torch.Generator(device=self.device).manual_seed(42)
                #     ).images[0]
                    
                else:
                    # No supported pipeline detected - raise error
                    raise RuntimeError(
                        f"⚠️  Unsupported pipeline type: {type(self.pipe).__name__}\n"
                        f"❌ HiDream-E1 editing model is required for image editing.\n"
                        f"ℹ️  Please specify HiDream-E1 checkpoint path with --model-editor"
                    )
                
                # Save the edited image
                edited_image.save(output_path)
                print(f"✅ Successfully edited and saved: {output_path}")
                
            except Exception as e:
                print(f"❌ Error during editing: {e}")
                print("🔄 Performing fallback edit (enhanced copy)...")
                
                # Enhanced fallback - at least resize/enhance the image
                try:
                    # Apply some basic enhancement as fallback
                    from PIL import ImageEnhance
                    
                    enhanced_image = input_image.copy()
                    
                    # Apply enhancements based on the edit prompt
                    if "happy" in edit_prompt.lower() or "bright" in edit_prompt.lower():
                        # Increase brightness and contrast for "happy" edits
                        enhancer = ImageEnhance.Brightness(enhanced_image)
                        enhanced_image = enhancer.enhance(1.1)
                        enhancer = ImageEnhance.Contrast(enhanced_image)
                        enhanced_image = enhancer.enhance(1.1)
                        
                    elif "vibrant" in edit_prompt.lower() or "color" in edit_prompt.lower():
                        # Increase color saturation
                        enhancer = ImageEnhance.Color(enhanced_image)
                        enhanced_image = enhancer.enhance(1.2)
                        
                    elif "dramatic" in edit_prompt.lower():
                        # Increase contrast
                        enhancer = ImageEnhance.Contrast(enhanced_image)
                        enhanced_image = enhancer.enhance(1.3)
                    
                    enhanced_image.save(output_path)
                    print(f"💡 Applied basic enhancement based on prompt")
                    
                except Exception as e2:
                    print(f"⚠️  Fallback enhancement failed: {e2}")
                    input_image.save(output_path)
                    print("📋 Copied original image")
                
        else:
            print("⚠️  No model loaded, performing enhanced copy...")
            # Enhanced mock editing with basic image processing
            try:
                from PIL import ImageEnhance
                
                enhanced_image = input_image.copy()
                
                # Apply basic enhancements based on prompt keywords
                if "happy" in edit_prompt.lower() or "bright" in edit_prompt.lower():
                    enhancer = ImageEnhance.Brightness(enhanced_image)
                    enhanced_image = enhancer.enhance(1.1)
                elif "vibrant" in edit_prompt.lower():
                    enhancer = ImageEnhance.Color(enhanced_image)
                    enhanced_image = enhancer.enhance(1.2)
                elif "dramatic" in edit_prompt.lower():
                    enhancer = ImageEnhance.Contrast(enhanced_image)
                    enhanced_image = enhancer.enhance(1.3)
                
                enhanced_image.save(output_path)
                print(f"💡 Applied basic enhancement: {edit_prompt}")
                
            except Exception:
                input_image.save(output_path)
                print("📋 Copied original image")
        
        return output_path, formatted_prompt
    
    def batch_edit(self, 
                   image_paths: List[str], 
                   edit_prompts: List[str],
                   output_dir: str = None) -> List[str]:
        """
        Edit multiple images with different prompts
        
        Args:
            image_paths: List of input image paths
            edit_prompts: List of edit prompts (one per image)
            output_dir: Directory to save edited images
            
        Returns:
            List of paths to edited images
        """
        if len(image_paths) != len(edit_prompts):
            raise ValueError("Number of images and prompts must match")
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'edited_images')
        
        os.makedirs(output_dir, exist_ok=True)
        
        edited_paths = []
        
        print(f"Batch editing {len(image_paths)} images...")
        
        for i, (image_path, edit_prompt) in enumerate(zip(image_paths, edit_prompts), 1):
            print(f"\nEditing image {i}/{len(image_paths)}")
            
            try:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_edited.png")
                
                edited_path = self.edit_image(
                    image_path=image_path,
                    edit_prompt=edit_prompt,
                    output_path=output_path
                )
                
                edited_paths.append(edited_path)
                
            except Exception as e:
                print(f"✗ Error editing {image_path}: {str(e)}")
                edited_paths.append(None)
        
        print(f"\nBatch editing complete! Processed {len([p for p in edited_paths if p])} images successfully.")
        return edited_paths


def main():
    """Main function for testing the image editor"""
    import glob
    
    # Initialize editor
    editor = ImageEditor()
    
    # Find generated images
    image_dir = "../generated_images"
    if os.path.exists(image_dir):
        image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        
        if image_paths:
            print(f"Found {len(image_paths)} images to edit")
            
            # Example edit prompts
            edit_prompts = [
                "make the person more happy and smiling",
                "add more vibrant colors and make it more cheerful",
                "make the scene more dramatic with better lighting",
                "add more details and make it more realistic",
                "make the atmosphere more mystical and magical"
            ]
            
            # Ensure we have enough prompts
            while len(edit_prompts) < len(image_paths):
                edit_prompts.append("enhance the image and make it more beautiful")
            
            # Edit all images
            edited_paths = editor.batch_edit(
                image_paths[:len(edit_prompts)], 
                edit_prompts[:len(image_paths)]
            )
            
            print(f"\nEditing complete! Results:")
            for original, edited in zip(image_paths, edited_paths):
                if edited:
                    print(f"  {os.path.basename(original)} → {os.path.basename(edited)}")
            
        else:
            print("No images found in generated_images directory")
    else:
        print("Generated images directory not found")


if __name__ == "__main__":
    main()
