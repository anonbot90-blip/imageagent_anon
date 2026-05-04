#!/usr/bin/env python3
"""
Complete ImageAgent Pipeline
Integrates HiDream-I1 (generation), Qwen-VL (analysis), and image editors (editing)
Supports various VLM models for action plan generation
"""

import os
import sys
import json
import glob
import argparse
import shutil
import subprocess
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import our modules
from image_analyzer import ImageAnalyzer
from image_editor import ImageEditor
from batch_image_generator import BatchImageGenerator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ImageAgentPipeline:
    """Complete pipeline for image generation, analysis, and editing"""
    
    def __init__(self, model_analyzer=None, model_planner=None, model_editor=None, hidream_checkpoint=None, model_reward=None, action_library_path=None):
        """Initialize the complete pipeline
        
        Args:
            model_analyzer: Model for image analysis (e.g., "Qwen/Qwen3-VL-8B-Instruct")
            model_planner: Model for action planning (e.g., "Qwen/Qwen3-VL-8B-Instruct")
            model_editor: Image editor to use - "qwen" or "hidream" (REQUIRED)
            hidream_checkpoint: HiDream-E1 checkpoint path (required if model_editor="hidream")
            model_reward: Model for reward evaluation (e.g., "Qwen/Qwen3-VL-8B-Instruct")
            action_library_path: Path to action library JSON (defaults to action_library_v2.json)
        """
        print("🚀 Initializing ImageAgent Pipeline...")
        
        # Validate model_editor (MANDATORY)
        if model_editor is None:
            raise ValueError("--model-editor is REQUIRED! Use 'qwen' or 'hidream'")
        
        if model_editor.lower() not in ["qwen", "hidream"]:
            raise ValueError(f"Invalid --model-editor: '{model_editor}'. Must be 'qwen' or 'hidream'")
        
        # If hidream, checkpoint is required
        if model_editor.lower() == "hidream" and hidream_checkpoint is None:
            raise ValueError("--hidream-checkpoint is REQUIRED when using --model-editor hidream")
        
        # Don't load models yet - lazy loading to save GPU memory
        self.model_analyzer = model_analyzer
        self.model_planner = model_planner
        self.model_editor = model_editor.lower()
        self.hidream_checkpoint = hidream_checkpoint
        self.model_reward = model_reward
        self.action_library_path = action_library_path
        self.analyzer = None
        self.planner = None
        self.editor = None
        self.generator = None
        self.reward_evaluator = None
        
        # Setup directories
        self.setup_directories()
        
        print("✅ ImageAgent Pipeline initialized successfully!")
        if model_analyzer:
            print(f"   🔍 Analyzer Model: {model_analyzer}")
        if model_planner:
            print(f"   🧠 Planner Model: {model_planner}")
        if model_reward:
            print(f"   🏆 Reward Model: {model_reward}")
    
    def _get_analyzer(self):
        """Lazy load analyzer when needed"""
        if self.analyzer is None:
            print(f"📊 Loading image analyzer ({self.model_analyzer})...")
            from image_analyzer_qwen3 import ImageAnalyzerQwen3
            self.analyzer = ImageAnalyzerQwen3(
                model_name=self.model_analyzer,
                action_library_path=self.action_library_path
            )
        return self.analyzer
    
    def _get_planner(self):
        """Lazy load planner when needed"""
        if self.planner is None:
            # If planner is same as analyzer, reuse the same model
            if self.model_planner == self.model_analyzer and self.analyzer is not None:
                print(f"📊 Reusing analyzer as planner ({self.model_planner})")
                self.planner = self.analyzer
            else:
                print(f"📊 Loading action planner ({self.model_planner})...")
                from image_analyzer_qwen3 import ImageAnalyzerQwen3
                self.planner = ImageAnalyzerQwen3(
                    model_name=self.model_planner,
                    action_library_path=self.action_library_path
                )
        return self.planner
    
    def _get_reward_evaluator(self):
        """Lazy load reward evaluator when needed"""
        if self.reward_evaluator is None:
            from reward_model_evaluator import RewardModelEvaluator
            
            # Check if we can reuse an existing model (memory efficient)
            existing_model = None
            if self.model_reward == self.model_analyzer and self.analyzer is not None:
                existing_model = self.analyzer
            elif self.model_reward == self.model_planner and self.planner is not None:
                existing_model = self.planner
            
            self.reward_evaluator = RewardModelEvaluator(
                model_name=self.model_reward,
                existing_model=existing_model
            )
        return self.reward_evaluator
    
    def _get_editor(self):
        """Lazy load editor when needed"""
        if self.editor is None:
            if self.model_editor == "qwen":
                print(f"✏️  Loading Qwen-Image-Edit (20B parameters)...")
                print(f"   Model: Qwen/Qwen-Image-Edit")
                
                from diffusers import DiffusionPipeline
                import torch
                
                self.editor = DiffusionPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit",
                    torch_dtype=torch.bfloat16,
                ).to("cuda")
                
                self.editor.set_progress_bar_config(disable=True)
                print(f"   ✅ Qwen-Image-Edit loaded successfully")
                
            elif self.model_editor == "hidream":
                print(f"✏️  Loading HiDream-E1 (17B parameters)...")
                print(f"   Checkpoint: {self.hidream_checkpoint}")
                
                # CRITICAL: Set environment variables BEFORE imports
                import os
                os.environ["ATTN_BACKEND"] = "xformers"
                os.environ["DIFFUSERS_ATTN_IMPLEMENTATION"] = "eager"
                
                # Import required modules
                from omegaconf import OmegaConf
                from pathlib import Path
                import safetensors.torch
                from training.models import HiDreamE1LoRA
                
                # Load config
                config_path = Path(__file__).parent.parent / "training" / "config" / "training_config.yaml"
                config = OmegaConf.load(config_path)
                print(f"   Config loaded from: {config_path}")
                
                # Create HiDreamE1LoRA instance (loads BASE model)
                print(f"   Loading base HiDream-E1 model...")
                self.editor = HiDreamE1LoRA(config, device="cuda")
                print(f"   ✅ Base HiDream-E1 loaded successfully")
                
                # Load LoRA weights if checkpoint path provided and exists
                checkpoint_path = Path(self.hidream_checkpoint)
                if checkpoint_path.exists():
                    lora_weights_path = checkpoint_path / "adapter_model.safetensors"
                    if not lora_weights_path.exists():
                        lora_weights_path = checkpoint_path / "lora_weights.safetensors"
                    
                    if lora_weights_path.exists():
                        print(f"   Loading LoRA weights from: {lora_weights_path}")
                        lora_state = safetensors.torch.load_file(str(lora_weights_path))
                        missing, unexpected = self.editor.transformer.load_state_dict(lora_state, strict=False)
                        print(f"   ✅ Loaded {len(lora_state)} LoRA parameters")
                    else:
                        print(f"   ℹ️  No LoRA weights found - using base model")
                else:
                    print(f"   ℹ️  Checkpoint path doesn't exist - using base model")
                
                # Set to eval mode (CRITICAL for inference)
                self.editor.transformer.eval()
                print(f"   ✅ Model set to eval mode")
            else:
                print("⚠️  No HiDream-E1 model specified - image editing will fail!")
                print("ℹ️  Provide --model-editor path to enable editing")
                # Don't create ImageEditor - it will fail anyway
                raise RuntimeError("HiDream-E1 model is required for image editing")
        return self.editor
    
    def _unload_models(self):
        """Unload models to free GPU memory"""
        if self.analyzer is not None:
            print("🗑️  Unloading analyzer...")
            del self.analyzer
            self.analyzer = None
        
        if self.planner is not None and self.planner != self.analyzer:
            print("🗑️  Unloading planner...")
            del self.planner
            self.planner = None
        
        if self.editor is not None:
            print("🗑️  Unloading editor...")
            del self.editor
            self.editor = None
        
        # Clear GPU cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ GPU memory cleared")
    
    def setup_directories(self):
        """Create necessary directories"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        self.dirs = {
            'generated': os.path.join(base_dir, 'generated_images'),
            'analyzed': os.path.join(base_dir, 'analyzed_images'),
            'edited': os.path.join(base_dir, 'edited_images'),
            'results': os.path.join(base_dir, 'pipeline_results')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_images_with_structure(self, 
                                      num_images: int,
                                      output_dir: str,
                                      prompts_file: str = None) -> Dict:
        """
        Generate images using batch generator with folder structure
        
        Args:
            num_images: Number of images to generate
            output_dir: Base output directory
            prompts_file: Path to prompts.json
            
        Returns:
            Manifest dictionary with image data
        """
        print(f"🎨 Generating {num_images} new images...")
        
        # Default prompts file
        if prompts_file is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            prompts_file = os.path.join(base_dir, 'config', 'prompts.json')
        
        # Initialize batch generator
        generator = BatchImageGenerator(
            prompts_file=prompts_file,
            output_base_dir=output_dir,
            gpu_id=os.environ.get('CUDA_VISIBLE_DEVICES')
        )
        
        # Generate images
        manifest_path = generator.generate_batch(num_images=num_images)
        
        # Load and return manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        print(f"✅ Generated {len(manifest['images'])} images with folder structure")
        return manifest
    
    def prepare_existing_images(self, 
                                num_images: int,
                                output_dir: str) -> Dict:
        """
        Copy existing images into new folder structure
        
        Args:
            num_images: Number of images to prepare
            output_dir: Base output directory
            
        Returns:
            Manifest dictionary with image data
        """
        print(f"📸 Preparing {num_images} existing images...")
        
        # Find existing images
        existing_images = glob.glob(os.path.join(self.dirs['generated'], "*.png"))[:num_images]
        
        if not existing_images:
            raise ValueError(f"No existing images found in {self.dirs['generated']}")
        
        print(f"   Found {len(existing_images)} existing images")
        
        # Create manifest structure
        manifest = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "num_images": len(existing_images),
            "source": "existing_images",
            "images": []
        }
        
        # Copy each image to folder structure
        for i, img_path in enumerate(existing_images, 1):
            img_name = os.path.basename(img_path)
            # Extract style from filename if available (e.g., image_1_photorealistic.png)
            parts = os.path.splitext(img_name)[0].split('_')
            style = parts[-1] if len(parts) > 1 else f"image{i}"
            
            # Create folder
            folder_name = f"image_{i}_{style}"
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Copy image as original.png
            dest_path = os.path.join(folder_path, "original.png")
            shutil.copy2(img_path, dest_path)
            
            # Create basic prompt.json
            prompt_data = {
                "id": i,
                "style": style,
                "source": img_name,
                "text": f"Existing image: {img_name}"
            }
            with open(os.path.join(folder_path, "prompt.json"), 'w') as f:
                json.dump(prompt_data, f, indent=2)
            
            # Add to manifest
            manifest["images"].append({
                "id": i,
                "style": style,
                "folder": folder_name,
                "original": f"{folder_name}/original.png",
                "prompt_file": f"{folder_name}/prompt.json",
                "status": "copied"
            })
            
            print(f"   ✓ Prepared: {folder_name}")
        
        # Save manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Prepared {len(existing_images)} images in folder structure")
        return manifest
    
    def analyze_images_in_folders(self, 
                                  image_data_list: List[Dict],
                                  output_dir: str,
                                  user_prompt: str = None) -> None:
        """
        Analyze images in folder structure
        
        Args:
            image_data_list: List of image data from manifest
            output_dir: Base output directory
            user_prompt: User's transformation request (required for 30B action planning)
        """
        print(f"🔍 Analyzing {len(image_data_list)} images...")
        print(f"   📊 Analyzer: {self.model_analyzer}")
        print(f"   🧠 Planner: {self.model_planner}")
        print(f"   📝 DEBUG: user_prompt = {repr(user_prompt)}")  # DEBUG
        
        if not user_prompt:
            print("⚠️  Warning: user_prompt required for action planning")
        
        for i, img_data in enumerate(image_data_list, 1):
            folder_path = os.path.join(output_dir, img_data['folder'])
            original_path = os.path.join(output_dir, img_data['original'])
            analysis_path = os.path.join(folder_path, "analysis.json")
            action_plan_path = os.path.join(folder_path, "action_plan.json")
            prompt_path = os.path.join(folder_path, "prompt.json")
            
            print(f"\nAnalyzing image {i}/{len(image_data_list)}: {img_data['folder']}")
            
            # Load prompt data if available
            if os.path.exists(prompt_path) and 'prompt_data' not in img_data:
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        img_data['prompt_data'] = json.load(f)
                except Exception as e:
                    print(f"  ⚠️  Could not load prompt.json: {e}")
            
            try:
                # Step 1: Analyze image
                analyzer = self._get_analyzer()
                print(f"  📊 Generating analysis...")
                analysis = analyzer.analyze_image(original_path, img_data.get('id'))
                
                # Save analysis
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)
                img_data['analysis'] = f"{img_data['folder']}/analysis.json"
                print(f"  ✓ Analysis saved: analysis.json")
                
                # Step 2: Generate action plan (if user_prompt provided)
                print(f"  🔧 DEBUG: Checking if user_prompt exists... user_prompt={repr(user_prompt)}")  # DEBUG
                if user_prompt:
                    print(f"  🔧 DEBUG: Inside if user_prompt block!")  # DEBUG
                    # Get per-image edit prompt (for paired transformations)
                    current_user_prompt = user_prompt
                    if 'prompt_data' in img_data:
                        print(f"  🔧 DEBUG: prompt_data found in img_data")  # DEBUG
                        # Check for 'edit' or 'edit_info' in prompt_data
                        if 'edit' in img_data['prompt_data']:
                            current_user_prompt = img_data['prompt_data']['edit'].get('text', user_prompt)
                        elif 'edit_info' in img_data['prompt_data']:
                            current_user_prompt = img_data['prompt_data']['edit_info'].get('text', user_prompt)
                            print(f"  🔧 DEBUG: Using edit_info.text = {current_user_prompt[:50]}...")  # DEBUG
                        
                        if current_user_prompt != user_prompt:
                            print(f"  📝 Using paired edit prompt: {current_user_prompt[:80]}...")
                    
                    print(f"  🔧 DEBUG: About to load planner...")  # DEBUG
                    planner = self._get_planner()
                    print(f"  🧠 Generating action plan...")
                    action_plan = planner.generate_action_plan(
                        original_path,
                        current_user_prompt,
                        analysis
                    )
                    
                    # Save action plan
                    with open(action_plan_path, 'w', encoding='utf-8') as f:
                        json.dump(action_plan, f, indent=2, ensure_ascii=False)
                    img_data['action_plan'] = f"{img_data['folder']}/action_plan.json"
                    print(f"  ✓ Action plan saved: action_plan.json")
                
            except Exception as e:
                print(f"✗ Error analyzing {img_data['folder']}: {str(e)}")
                img_data['analysis_error'] = str(e)
        
        print(f"\n✅ Analysis completed for all images")
    
    def edit_images_in_folders(self,
                              image_data_list: List[Dict],
                              output_dir: str,
                              edit_prompt: str,
                              paired_edit_prompts: List[Dict] = None) -> None:
        """
        Edit images in folder structure
        
        Args:
            image_data_list: List of image data from manifest
            output_dir: Base output directory
            edit_prompt: The editing instruction (single prompt for all images)
            paired_edit_prompts: List of edit prompt dicts (one per image) for style transformations
        """
        print(f"✏️  Editing {len(image_data_list)} images using HiDream-E1...")
        
        if paired_edit_prompts:
            print(f"   🎨 Using paired style transformation prompts")
        
        for i, img_data in enumerate(image_data_list, 1):
            folder_path = os.path.join(output_dir, img_data['folder'])
            original_path = os.path.join(output_dir, img_data['original'])
            edited_path = os.path.join(folder_path, "edited.png")
            analysis_path = os.path.join(folder_path, "analysis.json")
            
            print(f"\nEditing image {i}/{len(image_data_list)}: {img_data['folder']}")
            
            try:
                # Determine which edit prompt to use
                if paired_edit_prompts and i <= len(paired_edit_prompts):
                    # Use corresponding edit prompt from paired list
                    edit_data = paired_edit_prompts[i-1]
                    current_edit_prompt = edit_data.get('text', edit_prompt)
                    print(f"  🎨 Style transformation: {edit_data.get('name', 'Custom')}")
                    print(f"     {edit_data.get('target_style', '')} ({edit_data.get('transformation_type', '')})")
                else:
                    # Use default single edit prompt for all
                    current_edit_prompt = edit_prompt
                
                # Check if action_plan.json exists (30B mode)
                action_plan_path = os.path.join(folder_path, "action_plan.json")
                
                if os.path.exists(action_plan_path):
                    # Use action-to-NL conversion (same as evaluation pipeline)
                    print(f"  🎯 Using action plan → NL conversion")
                    
                    # Load action plan
                    with open(action_plan_path, 'r', encoding='utf-8') as f:
                        action_plan = json.load(f)
                    
                    # Load analysis
                    analysis = None
                    if os.path.exists(analysis_path):
                        with open(analysis_path, 'r', encoding='utf-8') as f:
                            analysis = json.load(f)
                    
                    # Use model-generated hidream_prompt
                    if "hidream_prompt" in action_plan and action_plan["hidream_prompt"]:
                        enhanced_prompt = action_plan["hidream_prompt"]
                        print(f"  ✓ Using model-generated hidream_prompt")
                    else:
                        # Fallback: use overall_instruction or basic prompt
                        enhanced_prompt = action_plan.get("overall_instruction", "style_transformation_mode")
                        print(f"  ⚠ hidream_prompt not found, using overall_instruction fallback")
                    
                    # Save instruction for reference
                    instruction_path = os.path.join(folder_path, "instruction.txt")
                    with open(instruction_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced_prompt)
                    
                    print(f"  ✓ Instruction generated from action plan")
                    
                else:
                    # Standard context enhancement (old method)
                    enhanced_prompt = current_edit_prompt
                    context_parts = []
                    
                    if os.path.exists(analysis_path):
                        with open(analysis_path, 'r', encoding='utf-8') as f:
                            analysis = json.load(f)
                    
                    # Extract comprehensive context from analysis
                    
                    # 1. Scene & Style context
                    if 'basic_info' in analysis:
                        basic = analysis['basic_info']
                        if 'overall_style' in basic:
                            context_parts.append(f"SCENE: {basic['overall_style']}")
                        if 'mood' in basic:
                            context_parts.append(f"MOOD: {basic['mood']}")
                    
                    # 2. Style analysis
                    if 'style_analysis' in analysis:
                        style = analysis['style_analysis']
                        if 'art_style' in style:
                            context_parts.append(f"STYLE: {style['art_style']}")
                        if 'color_palette' in style:
                            context_parts.append(f"COLORS: {style['color_palette']}")
                    
                    # 3. Composition & lighting
                    if 'composition' in analysis:
                        comp = analysis['composition']
                        if 'focal_point' in comp:
                            context_parts.append(f"FOCAL_POINT: {comp['focal_point']}")
                        if 'lighting' in comp:
                            context_parts.append(f"LIGHTING: {comp['lighting']}")
                    
                    # 4. Objects to avoid overlapping
                    if 'objects' in analysis and len(analysis['objects']) > 0:
                        main_objects = []
                        for obj in analysis['objects'][:3]:  # Top 3 objects
                            if 'name' in obj and 'location' in obj:
                                main_objects.append(f"{obj['name']} at {obj['location']}")
                        if main_objects:
                            context_parts.append(f"EXISTING_OBJECTS: {', '.join(main_objects)}")
                    
                    # 5. Editing suggestions
                    if 'editing_suggestions' in analysis:
                        suggestions = analysis['editing_suggestions'][:2]
                        if suggestions:
                            context_parts.append(f"SUGGESTIONS: {', '.join(suggestions)}")
                    
                    # Build enhanced prompt with context
                    if context_parts:
                        context_str = " | ".join(context_parts)
                        enhanced_prompt = f"{current_edit_prompt} [{context_str}]"
                
                print(f"  📝 Edit prompt: {enhanced_prompt[:120]}...")
                
                # Edit image (lazy load editor)
                editor = self._get_editor()
                
                # Load image and resize to 768x768 (CRITICAL for HiDream-E1)
                from PIL import Image
                import time
                input_image = Image.open(original_path).convert("RGB")
                input_image = input_image.resize((768, 768), Image.Resampling.LANCZOS)
                
                # Format prompt (matching evaluation)
                formatted_prompt = f"Editing Instruction: {enhanced_prompt}. Maintain high quality, original composition and style."
                
                print(f"  🎨 Generating edited image with {self.model_editor.upper()}...")
                start_time = time.time()
                
                # Generate edited image (API differs by editor type)
                if self.model_editor == "qwen":
                    output = editor(
                        prompt=formatted_prompt,
                        image=input_image,
                        height=768,
                        width=768,
                        num_inference_steps=28,
                        guidance_scale=7.5,
                    )
                elif self.model_editor == "hidream":
                    output = editor.pipeline(
                        prompt=formatted_prompt,
                        image=input_image,
                        height=768,
                        width=768,
                        num_inference_steps=28,
                        guidance_scale=3.5,
                        image_guidance_scale=4.0,
                    )
                
                edited_image = output.images[0]
                gen_time = time.time() - start_time
                
                # Save edited image
                edited_image.save(edited_path)
                result_path = edited_path
                hidream_prompt = formatted_prompt
                
                print(f"  ✅ Image edited in {gen_time:.2f}s")

                
                # Save edit prompt if it was generated
                if hidream_prompt:
                    edit_prompt_path = os.path.join(folder_path, "edit_prompt.txt")
                    with open(edit_prompt_path, 'w', encoding='utf-8') as f:
                        f.write(hidream_prompt)
                    print(f"  ✓ Edit prompt saved")
                
                # Update image_data with edited path
                img_data['edited'] = f"{img_data['folder']}/edited.png"
                
                print(f"  ✓ Successfully edited: {img_data['folder']}")
                
            except Exception as e:
                print(f"  ✗ Error editing {img_data['folder']}: {str(e)}")
                img_data['edit_error'] = str(e)
        
        print(f"\n✅ Editing completed for all images")
    
    def evaluate_images_in_folders(self,
                                   image_data_list: List[Dict],
                                   output_dir: str,
                                   edit_prompt: str) -> None:
        """
        Evaluate edited images using reward model
        
        Args:
            image_data_list: List of image data from manifest
            output_dir: Base output directory
            edit_prompt: User's transformation request
        """
        if not self.model_reward:
            print("⚠️  Skipping evaluation: No reward model specified")
            return
        
        print(f"\n🏆 Evaluating {len(image_data_list)} transformations...")
        
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, img_data in enumerate(image_data_list, 1):
            folder_path = os.path.join(output_dir, img_data['folder'])
            original_path = os.path.join(output_dir, img_data['original'])
            edited_path = os.path.join(folder_path, "edited.png")
            action_plan_path = os.path.join(folder_path, "action_plan.json")
            analysis_path = os.path.join(folder_path, "analysis.json")
            reward_scores_path = os.path.join(folder_path, "reward_scores.json")
            
            print(f"\nEvaluating {i}/{len(image_data_list)}: {img_data['folder']}")
            
            # Skip if edited image doesn't exist
            if not os.path.exists(edited_path):
                print(f"  ⚠️  Skipping: edited.png not found")
                failed_evaluations += 1
                continue
            
            # Skip if action plan doesn't exist
            if not os.path.exists(action_plan_path):
                print(f"  ⚠️  Skipping: action_plan.json not found")
                failed_evaluations += 1
                continue
            
            try:
                # Load action plan
                with open(action_plan_path, 'r', encoding='utf-8') as f:
                    action_plan = json.load(f)
                
                # Load analysis (optional)
                analysis = None
                if os.path.exists(analysis_path):
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                
                # Get user prompt from paired prompts or use default
                user_prompt = edit_prompt
                if 'prompt_data' in img_data:
                    # Check for 'edit' or 'edit_info' in prompt_data
                    if 'edit' in img_data['prompt_data']:
                        user_prompt = img_data['prompt_data']['edit'].get('text', edit_prompt)
                    elif 'edit_info' in img_data['prompt_data']:
                        user_prompt = img_data['prompt_data']['edit_info'].get('text', edit_prompt)
                
                # Lazy load reward evaluator
                evaluator = self._get_reward_evaluator()
                
                # Evaluate transformation
                reward_scores = evaluator.evaluate_transformation(
                    original_image_path=original_path,
                    edited_image_path=edited_path,
                    user_prompt=user_prompt,
                    action_plan=action_plan,
                    analysis=analysis
                )
                
                # Save reward scores
                with open(reward_scores_path, 'w', encoding='utf-8') as f:
                    json.dump(reward_scores, f, indent=2, ensure_ascii=False)
                
                # Update image_data
                img_data['reward_scores'] = f"{img_data['folder']}/reward_scores.json"
                
                print(f"  ✓ Evaluation saved: reward_scores.json")
                successful_evaluations += 1
                
            except Exception as e:
                print(f"  ✗ Error evaluating {img_data['folder']}: {str(e)}")
                img_data['evaluation_error'] = str(e)
                failed_evaluations += 1
        
        print(f"\n✅ Evaluation completed")
        print(f"   📊 Successful: {successful_evaluations}/{len(image_data_list)}")
        if failed_evaluations > 0:
            print(f"   ⚠️  Failed: {failed_evaluations}/{len(image_data_list)}")
    
    def save_pipeline_summary(self,
                             output_dir: str,
                             manifest: Dict,
                             edit_prompt: str) -> str:
        """
        Save pipeline summary
        
        Args:
            output_dir: Base output directory
            manifest: Manifest with image data
            edit_prompt: Edit prompt used
            
        Returns:
            Path to summary file
        """
        summary = {
            "timestamp": manifest.get('timestamp'),
            "edit_prompt": edit_prompt,
            "num_images": manifest.get('num_images'),
            "source": manifest.get('source', 'generated'),
            "output_directory": output_dir,
            "images": manifest['images']
        }
        
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary_path
    
    def run_complete_pipeline(self, 
                            edit_prompt: str,
                            num_images: int = 5,
                            generate_new: bool = False,
                            output_dir: str = None,
                            prompts_file: str = None,
                            use_paired_edits: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline: Generate → Analyze → Edit
        
        Args:
            edit_prompt: The editing instruction (e.g., "make the person more happy")
            num_images: Number of images to process
            generate_new: Whether to generate new images or use existing ones
            output_dir: Custom output directory name
            prompts_file: Custom prompts file for generation
            use_paired_edits: Whether to use paired edit prompts from same JSON file
            
        Returns:
            Dictionary containing all pipeline results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        if output_dir is None:
            output_dir = f"imageagent_results_{timestamp}"
        
        # Make it absolute if relative
        if not os.path.isabs(output_dir):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            output_dir = os.path.join(base_dir, output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print(f"🤖 IMAGEAGENT PIPELINE STARTED - {timestamp}")
        print("=" * 60)
        print(f"✏️  IMAGE EDITOR: {self.model_editor.upper()} ({'Qwen-Image-Edit 20B' if self.model_editor == 'qwen' else 'HiDream-E1 17B'})")
        print(f"📝 Edit Mode: {'Paired Style Transformations' if use_paired_edits else 'Single Edit Prompt'}")
        if not use_paired_edits:
            print(f"📝 Edit Prompt: '{edit_prompt}'")
        print(f"🖼️  Processing: {num_images} images")
        print(f"🔄 Generate New: {generate_new}")
        print(f"📁 Output Directory: {output_dir}")
        print()
        
        # Load paired edit prompts if requested
        paired_edit_prompts = None
        if use_paired_edits:
            if prompts_file and os.path.exists(prompts_file):
                print(f"📖 Loading paired edit prompts from: {prompts_file}")
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    prompts_data = json.load(f)
                    paired_edit_prompts = [p.get('edit', {}) for p in prompts_data.get('prompts', [])]
                print(f"   ✓ Loaded {len(paired_edit_prompts)} paired edit prompts")
            else:
                print(f"⚠️  Warning: use_paired_edits=True but no valid prompts_file provided")
                print(f"   Falling back to single edit prompt")
        
        # Step 1: Generate or prepare images with folder structure
        if generate_new:
            manifest = self.generate_images_with_structure(
                num_images=num_images,
                output_dir=output_dir,
                prompts_file=prompts_file
            )
        else:
            manifest = self.prepare_existing_images(
                num_images=num_images,
                output_dir=output_dir
            )
        
        # Get image data list from manifest
        image_data_list = manifest['images']
        
        # Clear GPU memory after generation (generator already unloads itself)
        print("\n🧹 Clearing GPU memory after generation...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 2: Analyze images in folders (pass edit_prompt for 30B action planning)
        self.analyze_images_in_folders(image_data_list, output_dir, user_prompt=edit_prompt)
        
        # Clear GPU memory after analysis, before editing
        print("\n🧹 Clearing GPU memory before editing...")
        self._unload_models()  # Unload analyzer
        
        # Step 3: Edit images in folders
        self.edit_images_in_folders(image_data_list, output_dir, edit_prompt, paired_edit_prompts)
        
        # Step 4: Evaluate transformations (if reward model specified)
        if self.model_reward:
            # Clear GPU memory before evaluation
            print("\n🧹 Clearing GPU memory before evaluation...")
            self._unload_models()  # Unload editor
            
            self.evaluate_images_in_folders(image_data_list, output_dir, edit_prompt)
        
        # Step 5: Save pipeline summary
        summary_path = self.save_pipeline_summary(output_dir, manifest, edit_prompt)
        
        print()
        print("=" * 60)
        print("🎉 IMAGEAGENT PIPELINE COMPLETED!")
        print("=" * 60)
        
        # Count successes
        analyzed_count = len([img for img in image_data_list if 'analysis' in img])
        edited_count = len([img for img in image_data_list if 'edited' in img])
        evaluated_count = len([img for img in image_data_list if 'reward_scores' in img])
        
        print(f"📊 Analysis: {analyzed_count}/{len(image_data_list)} successful")
        print(f"✏️  Editing: {edited_count}/{len(image_data_list)} successful")
        if self.model_reward:
            print(f"🏆 Evaluation: {evaluated_count}/{len(image_data_list)} successful")
        print(f"💾 Results: {output_dir}")
        print(f"📄 Summary: {summary_path}")
        print()
        
        return {
            'output_dir': output_dir,
            'manifest': manifest,
            'edit_prompt': edit_prompt,
            'summary_file': summary_path
        }


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='ImageAgent Pipeline - Generate, Analyze, and Edit Images')
    parser.add_argument('edit_prompt', help='The editing instruction (e.g., "make the person more happy")')
    parser.add_argument('--num-images', '-n', type=int, default=5, help='Number of images to process (default: 5)')
    parser.add_argument('--generate-new', '-g', action='store_true', help='Generate new images instead of using existing ones')
    parser.add_argument('--output-dir', '-o', help='Output directory for results')
    parser.add_argument('--prompts-file', '-pf', help='Custom prompts file for generation')
    parser.add_argument('--style-prompts', '-s', action='store_true',
                        help='Use paired style transformation prompts from prompts_style.json')
    parser.add_argument('--model-analyzer', '-ma', type=str, required=True,
                        help='Model for image analysis (e.g., Qwen/Qwen3-VL-8B-Instruct)')
    parser.add_argument('--model-planner', '-mp', type=str, required=True,
                        help='Model for action planning (e.g., Qwen/Qwen3-VL-8B-Instruct)')
    parser.add_argument('--model-editor', '-me', type=str, required=True, choices=['qwen', 'hidream'],
                        help='Image editor to use: "qwen" (Qwen-Image-Edit) or "hidream" (HiDream-E1) - REQUIRED')
    parser.add_argument('--hidream-checkpoint', type=str, default=None,
                        help='HiDream-E1 checkpoint path (REQUIRED if --model-editor is "hidream")')
    parser.add_argument('--reward-model', '-rm', type=str, default=None,
                        help='Model for reward evaluation (e.g., Qwen/Qwen3-VL-8B-Instruct). Optional.')
    parser.add_argument('--action-library', '-al', type=str, default=None,
                        help='Path to action library JSON (defaults to action_library_v2.json)')
    
    args = parser.parse_args()
    
    try:
        # Determine prompts file
        prompts_file = args.prompts_file
        if args.style_prompts and not prompts_file:
            # Default to prompts_style.json when -s flag is used
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompts_file = os.path.join(base_dir, 'config', 'prompts_style.json')
        
        # Initialize pipeline
        pipeline = ImageAgentPipeline(
            model_analyzer=args.model_analyzer,
            model_planner=args.model_planner,
            model_editor=args.model_editor,
            hidream_checkpoint=args.hidream_checkpoint,
            model_reward=args.reward_model,
            action_library_path=args.action_library
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            edit_prompt=args.edit_prompt,
            num_images=args.num_images,
            generate_new=args.generate_new,
            output_dir=args.output_dir,
            prompts_file=prompts_file,
            use_paired_edits=args.style_prompts
        )
        
        print("✅ Pipeline completed successfully!")
        print(f"📄 Results directory: {results.get('output_dir', 'N/A')}")
        print(f"📄 Summary file: {results.get('summary_file', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
