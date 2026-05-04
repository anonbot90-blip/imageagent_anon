#!/usr/bin/env python3
"""
Batch Image Generator for Complex Theme Dataset
Handles complex theme prompts with scene_description and transformation_request
"""

import os
import sys
import json
import argparse
import uuid
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add HiDream-I1 to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HIDREAM_PATH = PROJECT_ROOT / "HiDream-I1"
sys.path.insert(0, str(HIDREAM_PATH))

from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler


class BatchImageGeneratorComplex:
    """Batch image generator for complex theme prompts"""
    
    def __init__(self, prompts_file: str, output_base_dir: str, gpu_id: str = None):
        """
        Initialize the batch generator
        
        Args:
            prompts_file: Path to prompts.json (complex theme format)
            output_base_dir: Base directory for output
            gpu_id: GPU ID to use (from CUDA_VISIBLE_DEVICES)
        """
        self.prompts_file = prompts_file
        self.output_base_dir = Path(output_base_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
        print(f"🎨 Complex Theme Batch Image Generator initialized")
        print(f"📁 Output directory: {self.output_base_dir}")
        print(f"📱 Using device: {self.device}")
        if gpu_id:
            print(f"🎮 GPU ID: {gpu_id}")
        
        # Create base output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load prompts
        self.prompts = self._load_prompts()
        print(f"📝 Loaded {len(self.prompts)} complex theme prompts from {prompts_file}")
    
    def _load_prompts(self) -> List[Dict]:
        """
        Load prompts from JSON file (prompts_unified_complex format)
        
        Format from prompts_unified_complex.json:
        {
          "prompts": [
            {
              "id": 1,
              "text": "A modern living room...",  # Scene for generation
              "theme": "modern_living_room",
              "model": "fast",
              "resolution": "1024x1024",
              "edit_info": {
                "text": "Transform to Victorian AND add candlelit lighting",
                "expected_actions": ["architecture_style", "mood_lighting"]
              }
            }
          ]
        }
        """
        with open(self.prompts_file, 'r') as f:
            data = json.load(f)
        
        prompts = data['prompts']
        normalized_prompts = []
        
        print(f"🔄 Loading {len(prompts)} complex prompts (20-action system)...")
        
        for prompt in prompts:
            # Unified complex format: already has 'text' and 'edit_info'
            if 'text' in prompt and 'edit_info' in prompt:
                # Format is already correct for pipeline
                normalized = {
                    'id': prompt.get('id', 'unknown'),
                    'name': prompt.get('name', '')[:50],
                    'category': prompt.get('category', 'complex'),
                    'text': prompt['text'],  # Scene description for generation
                    'style': prompt.get('theme', f"image_{prompt.get('id', 'unknown')}"),
                    'model': prompt.get('model', 'dev'),
                    'resolution': prompt.get('resolution', '1024x1024'),
                    'edit_info': prompt['edit_info']  # Already contains transformation text & actions
                }
                
                normalized_prompts.append(normalized)
            else:
                # Fallback: log missing fields
                print(f"⚠️  Warning: Prompt {prompt.get('id')} missing 'text' or 'edit_info', skipping...")
                continue
        
        print(f"✅ Loaded {len(normalized_prompts)} prompts with 20-action system")
        return normalized_prompts
    
    def _load_model(self, model_type: str = "dev"):
        """
        Load HiDream-I1 model once (following official inference.py structure)
        
        Args:
            model_type: 'fast', 'dev', or 'full'
        """
        if self.pipe is not None:
            print("⚠️  Model already loaded, skipping...")
            return
        
        print(f"🔧 Loading HiDream-I1-{model_type.capitalize()} model...")
        
        MODEL_PREFIX = "HiDream-ai"
        LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        # Model configurations (from official inference.py)
        MODEL_CONFIGS = {
            "dev": {
                "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
                "guidance_scale": 0.0,
                "num_inference_steps": 28,
                "shift": 6.0,
                "scheduler": FlashFlowMatchEulerDiscreteScheduler
            },
            "full": {
                "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
                "guidance_scale": 5.0,
                "num_inference_steps": 50,
                "shift": 3.0,
                "scheduler": FlowUniPCMultistepScheduler
            },
            "fast": {
                "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
                "guidance_scale": 0.0,
                "num_inference_steps": 16,
                "shift": 3.0,
                "scheduler": FlashFlowMatchEulerDiscreteScheduler
            }
        }
        
        config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["dev"])
        model_path = config["path"]
        
        try:
            # Load scheduler
            print("  Loading scheduler...")
            scheduler = config["scheduler"](
                num_train_timesteps=1000,
                shift=config["shift"],
                use_dynamic_shifting=False
            )
            
            # Load Llama tokenizer and text encoder
            print("  Loading Llama tokenizer and text encoder...")
            tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
                LLAMA_MODEL_NAME,
                use_fast=False
            )
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME,
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            # Load transformer separately (critical step!)
            print(f"  Loading transformer from {model_path}...")
            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
            # Load HiDream pipeline
            print(f"  Loading HiDream-I1 pipeline...")
            self.pipe = HiDreamImagePipeline.from_pretrained(
                model_path,
                scheduler=scheduler,
                tokenizer_4=tokenizer_4,
                text_encoder_4=text_encoder_4,
                torch_dtype=torch.bfloat16
            ).to(self.device, torch.bfloat16)
            
            # Assign transformer to pipeline (critical!)
            self.pipe.transformer = transformer
            
            # Store inference parameters
            self.inference_params = {
                "steps": config["num_inference_steps"],
                "guidance_scale": config["guidance_scale"]
            }
            
            print(f"✅ Model loaded successfully!")
            print(f"   Inference steps: {self.inference_params['steps']}")
            print(f"   Guidance scale: {self.inference_params['guidance_scale']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _unload_model(self):
        """Unload model to free GPU memory"""
        if self.pipe is not None:
            print("🗑️  Unloading model to free GPU memory...")
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Model unloaded")
    
    def _generate_single_image(self, prompt: Dict, output_folder: Path, image_index: int = 0) -> bool:
        """
        Generate a single image
        
        Args:
            prompt: Prompt dictionary with text, style, etc.
            output_folder: Folder to save the image
            image_index: Index of the image in the batch (for unique seeding)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output folder
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Paths
            image_path = output_folder / "original.png"
            prompt_path = output_folder / "prompt.json"
            
            # Generate image
            print(f"  Generating: {prompt['text'][:60]}...")
            
            # Parse resolution (can be string "1024x1024" or list [1024, 1024])
            resolution = prompt.get('resolution', [1024, 1024])
            if isinstance(resolution, str):
                # Parse "1024x1024" to [1024, 1024]
                width, height = map(int, resolution.split('x'))
                resolution = [width, height]
            
            # Generate with proper parameters
            # Use image_index for unique but reproducible seeding
            seed = 42 + image_index  # Each image gets a unique seed
            result = self.pipe(
                prompt=prompt['text'],
                height=resolution[1],
                width=resolution[0],
                num_inference_steps=self.inference_params['steps'],
                guidance_scale=self.inference_params['guidance_scale'],
                generator=torch.Generator(device=self.device).manual_seed(seed)
            )
            
            # Extract image from result
            if hasattr(result, 'images'):
                image = result.images[0]
            elif isinstance(result, tuple):
                image = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                image = result
            
            # Save image
            image.save(image_path)
            print(f"  ✅ Saved: {image_path}")
            
            # Save prompt metadata
            with open(prompt_path, 'w') as f:
                json.dump(prompt, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error generating image: {e}")
            return False
    
    def generate_batch(self, num_images: int = 5) -> str:
        """
        Generate multiple images in batch
        
        Args:
            num_images: Number of images to generate
            
        Returns:
            Path to manifest.json
        """
        print("=" * 60)
        print(f"🚀 Starting batch generation of {num_images} complex theme images")
        print("=" * 60)
        
        # Select prompts with cycling logic
        # If num_images > available prompts, cycle through prompts
        num_available_prompts = len(self.prompts)
        
        if num_images <= num_available_prompts:
            # Use first N prompts
            prompts_to_use = self.prompts[:num_images]
        else:
            # Cycle through prompts to reach num_images
            prompts_to_use = []
            cycles_needed = (num_images // num_available_prompts) + 1
            print(f"📦 Cycling through {num_available_prompts} prompts {cycles_needed} times to generate {num_images} images")
            
            for cycle in range(cycles_needed):
                for prompt in self.prompts:
                    if len(prompts_to_use) >= num_images:
                        break
                    # Create a copy of the prompt with updated ID to avoid conflicts
                    prompt_copy = prompt.copy()
                    # Keep original ID but add cycle suffix for folder naming
                    prompt_copy['_cycle'] = cycle
                    prompt_copy['_original_id'] = prompt['id']
                    prompts_to_use.append(prompt_copy)
                if len(prompts_to_use) >= num_images:
                    break
            
            print(f"✓ Generated {len(prompts_to_use)} prompts by cycling")
        
        # Determine which model to use (use the most common one)
        model_types = [p.get('model', 'dev') for p in prompts_to_use]
        primary_model = max(set(model_types), key=model_types.count)
        
        # Load model once
        self._load_model(primary_model)
        
        # Generate images
        manifest_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "num_images": num_images,
            "prompts_file": str(self.prompts_file),
            "model_used": primary_model,
            "dataset_type": "complex_theme",
            "images": []
        }
        
        success_count = 0
        fail_count = 0
        
        for i, prompt in enumerate(prompts_to_use, 1):
            print(f"\n[{i}/{num_images}] Processing: {prompt['style']}")
            
            # Create unique folder name with UUID prefix
            base_id = prompt.get('_original_id', prompt['id'])
            cycle = prompt.get('_cycle', 0)
            unique_id = uuid.uuid4().hex[:8]  # 8-char UUID for uniqueness
            
            # Format: image_{uuid}_{id}_{style} (no cycle suffix needed, UUID handles uniqueness)
            folder_name = f"image_{unique_id}_{base_id}_{prompt['style']}"
            
            output_folder = self.output_base_dir / folder_name
            
            # Generate image (pass i-1 as image_index since i starts at 1)
            success = self._generate_single_image(prompt, output_folder, image_index=i-1)
            
            if success:
                success_count += 1
                status = "success"
            else:
                fail_count += 1
                status = "failed"
            
            # Add to manifest
            manifest_data["images"].append({
                "uuid": unique_id,
                "id": base_id,
                "original_id": prompt['id'],
                "cycle": cycle,
                "style": prompt['style'],
                "folder": folder_name,
                "original": f"{folder_name}/original.png",
                "prompt_file": f"{folder_name}/prompt.json",
                "status": status,
                "complexity_level": prompt.get('complex_theme_data', {}).get('complexity_level')
            })
        
        # Unload model
        self._unload_model()
        
        # Save manifest
        manifest_path = self.output_base_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        print("\n" + "=" * 60)
        print("🎉 Batch generation complete!")
        print(f"✅ Success: {success_count}/{num_images}")
        print(f"❌ Failed: {fail_count}/{num_images}")
        print(f"📄 Manifest: {manifest_path}")
        print("=" * 60)
        
        return str(manifest_path)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Batch Image Generator for Complex Theme Dataset')
    parser.add_argument('--prompts', '-p', required=True, help='Path to complex theme prompts.json file')
    parser.add_argument('--num-images', '-n', type=int, default=5, help='Number of images to generate (default: 5)')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--gpu', help='GPU ID to use (overrides CUDA_VISIBLE_DEVICES)')
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Generate output directory name if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"generation_output_complex_{timestamp}"
    
    try:
        # Initialize generator
        generator = BatchImageGeneratorComplex(
            prompts_file=args.prompts,
            output_base_dir=args.output_dir,
            gpu_id=os.environ.get('CUDA_VISIBLE_DEVICES')
        )
        
        # Generate images
        manifest_path = generator.generate_batch(num_images=args.num_images)
        
        print(f"\n✅ Generation completed successfully!")
        print(f"📂 Output directory: {args.output_dir}")
        print(f"📄 Manifest file: {manifest_path}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

