#!/usr/bin/env python3
"""
Image Analysis Module using Qwen3-VL (Next Generation)
Analyzes generated images and creates JSON representations describing object locations and content

This is an experimental implementation using Qwen3-VL models (4B, 8B, or 30B) for improved analysis quality.
All models can generate action plans using the V2 action library.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Add project root to path for action library access
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from actions.action_loader import ActionLibrary


class ImageAnalyzerQwen3:
    """Image analyzer using Qwen3-VL for detailed image understanding"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct", action_library_path: str = None):
        """Initialize the Qwen3-VL model for image analysis
        
        Args:
            model_name: Model to use (e.g., 4B, 8B, or 30B variant)
            action_library_path: Path to action library JSON (for action planning)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if this is a "Thinking" model (requires special JSON extraction)
        self.is_thinking = "Thinking" in model_name
        
        print(f"🚀 Loading Qwen3-VL model: {model_name}")
        print(f"📱 Using device: {self.device}")
        if self.is_thinking:
            print("🧠 Thinking model: Using JSON extraction for responses")
        
        # Always load action library (for action planning capability)
        if action_library_path is None:
            action_library_path = PROJECT_ROOT / "actions" / "action_library_v2.json"
        print(f"📚 Loading action library: {action_library_path}")
        self.action_library = ActionLibrary(str(action_library_path))
        print(f"   ✓ Loaded {len(self.action_library.get_all_actions())} V2 actions")
        
        # Load processor and model
        try:
            print("  Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            print("  Loading Qwen3-VL model (this may take a few minutes)...")
            # Use Auto class for both 4B and 30B (handles MoE automatically)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.model.eval()
            
            model_size = "~30B" if "30B" in self.model_name else ("~8B" if "8B" in self.model_name else "~4B")
            print("✅ Qwen3-VL model loaded successfully!")
            print(f"   Model size: {model_size} parameters")
            if torch.cuda.is_available():
                print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error loading Qwen3-VL model: {str(e)}")
            print("💡 Tip: Make sure transformers is up to date: pip install --upgrade transformers")
            raise
    
    def analyze_image(self, image_path: str, image_id: str = None) -> Dict[str, Any]:
        """
        Analyze a single image and return detailed JSON description
        
        Args:
            image_path: Path to the image file
            image_id: Optional ID for the image
            
        Returns:
            Dictionary containing detailed image analysis
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_id is None:
            image_id = os.path.basename(image_path)
        
        # Load and process image
        image = Image.open(image_path)
        
        # Create comprehensive analysis prompt (enhanced for Qwen3)
        # For Thinking models, we need to be very explicit about JSON-only output
        if self.is_thinking:
            analysis_prompt = """You must output ONLY valid JSON, no explanations or reasoning.

Output this exact JSON structure:
{
    "image_id": "filename",
    "basic_info": {
        "dimensions": [width, height],
        "dominant_colors": ["color1", "color2", "color3"],
        "overall_style": "style description",
        "mood": "mood description",
        "quality": "quality assessment"
    },
    "objects": [
        {
            "name": "object name",
            "description": "detailed description",
            "location": "precise location description",
            "size": "relative size (small/medium/large)",
            "confidence": 0.95,
            "attributes": ["attribute1", "attribute2"]
        }
    ],
    "scene_description": "detailed scene description",
    "composition": {
        "layout": "composition layout type",
        "focal_point": "main focal point with location",
        "depth": "depth and layering description",
        "lighting": "detailed lighting description",
        "perspective": "perspective and viewpoint description"
    },
    "style_analysis": {
        "art_style": "artistic style",
        "technique": "technique used",
        "color_palette": "detailed color palette description",
        "texture": "texture description",
        "mood_keywords": ["keyword1", "keyword2", "keyword3"]
    },
    "technical_quality": {
        "sharpness": "assessment of sharpness",
        "exposure": "assessment of exposure",
        "contrast": "assessment of contrast",
        "color_balance": "assessment of color balance"
    },
    "editing_suggestions": [
        "specific suggestion 1",
        "specific suggestion 2"
    ],
    "spatial_map": "spatial relationships description"
}

Output ONLY the JSON, nothing else."""
        else:
            analysis_prompt = """
        Analyze this image in detail and provide a comprehensive JSON description with the following structure:
        
        {
            "image_id": "filename",
            "basic_info": {
                "dimensions": [width, height],
                "dominant_colors": ["color1", "color2", "color3"],
                "overall_style": "style description",
                "mood": "mood description",
                "quality": "quality assessment"
            },
            "objects": [
                {
                    "name": "object name",
                    "description": "detailed description",
                    "location": "precise location description (e.g., center, top-left, bottom-right with percentages if possible)",
                    "size": "relative size (small/medium/large)",
                    "confidence": 0.95,
                    "attributes": ["attribute1", "attribute2"]
                }
            ],
            "scene_description": "detailed and comprehensive scene description",
            "composition": {
                "layout": "composition layout type (rule of thirds, centered, etc.)",
                "focal_point": "main focal point with specific location",
                "depth": "depth and layering description",
                "lighting": "detailed lighting description (direction, intensity, type)",
                "perspective": "perspective and viewpoint description"
            },
            "style_analysis": {
                "art_style": "artistic style (photorealistic, painting, digital art, etc.)",
                "technique": "technique used",
                "color_palette": "detailed color palette description",
                "texture": "texture description",
                "mood_keywords": ["keyword1", "keyword2", "keyword3"]
            },
            "technical_quality": {
                "sharpness": "assessment of sharpness",
                "exposure": "assessment of exposure",
                "contrast": "assessment of contrast",
                "color_balance": "assessment of color balance"
            },
            "editing_suggestions": [
                "specific and actionable suggestion 1",
                "specific and actionable suggestion 2", 
                "specific and actionable suggestion 3"
            ],
            "spatial_map": "description of spatial relationships between objects"
        }
        
        Provide only the JSON response, no additional text. Be as detailed and precise as possible.
        """
        
        # Prepare messages for Qwen3-VL (same format as Qwen2-VL)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": analysis_prompt},
                ],
            }
        ]
        
        # Process the vision info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=text, 
            images=image_inputs, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move all inputs to the same device as the model
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Generate analysis with Qwen3-VL optimized parameters
        print("  ⏳ Generating detailed analysis (this takes ~60-90 seconds)...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Increased for more detailed analysis
                do_sample=True,
                temperature=0.1,  # Low temperature for consistent JSON
                top_p=0.9,
                repetition_penalty=1.1  # Reduce repetition
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        print("  ✓ Analysis generated, parsing JSON...")
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            # For Thinking models, extract JSON from reasoning text
            clean_response = response.strip()
            
            # Try to find JSON in the response (Thinking models may wrap it in reasoning)
            if self.is_thinking and not clean_response.startswith('{'):
                # Look for JSON object in the response
                import re
                # Find the first { and last } to extract JSON
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    clean_response = json_match.group(0)
                    print("  ✓ Extracted JSON from reasoning text")
                else:
                    print("  ⚠️ Could not find JSON in response, trying full text...")
            
            # Clean up markdown code blocks if present
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]  # Remove ```json
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]   # Remove ```
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]  # Remove closing ```
            
            clean_response = clean_response.strip()
            
            # Try to parse JSON response
            analysis_result = json.loads(clean_response)
            analysis_result["image_id"] = image_id
            analysis_result["image_path"] = image_path
            analysis_result["model_used"] = self.model_name
            analysis_result["model_version"] = "qwen3-vl"
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            print(f"⚠️  Warning: Could not parse JSON response for {image_path}: {str(e)}")
            print(f"📄 Raw response preview: {response[:200]}...")
            analysis_result = {
                "image_id": image_id,
                "image_path": image_path,
                "model_used": self.model_name,
                "model_version": "qwen3-vl",
                "raw_response": response,
                "error": f"Failed to parse JSON response: {str(e)}",
                "basic_info": {
                    "dimensions": list(image.size),
                    "format": image.format
                }
            }
        
        return analysis_result
    
    def generate_action_plan(self, image_path: str, user_prompt: str, analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate action plan using the loaded VLM model
        
        Args:
            image_path: Path to the image
            user_prompt: User's transformation request (e.g., "transform to beach sunset")
            analysis: Optional pre-computed analysis (if None, will analyze first)
            
        Returns:
            Action plan dictionary with V2 actions
        """
        # Action library is now always loaded, so any model can generate action plans
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format action library for prompt
        actions = self.action_library.get_all_actions()
        action_library_text = "Available Actions:\n\n"
        for action in actions:
            action_id = action.get('action_id', action.get('id'))  # Support both 'id' and 'action_id'
            action_library_text += f"- {action_id}: {action['description']}\n"
            action_library_text += f"  Parameters: {', '.join(action['parameters'].keys())}\n"
        
        # Create system message - Chain-of-Thought for better planning
        # NOW INCLUDES: hidream_prompt generation for HiDream-E1
        # NOW INCLUDES: per-action reasoning for better model understanding
        # FOCUS: THEME & STYLE transformations
        system_message = f"""You are an expert THEME & STYLE transformation planner using Chain-of-Thought reasoning.

Your specialty: Planning BOLD, NOTICEABLE transformations that change:
- THEME: The scene's setting, atmosphere, mood, or environmental context (9 actions)
- STYLE: The artistic rendering or visual medium (1 action)

{action_library_text}

Action Categories:
- THEME Actions (9): location_setting, architecture_style, time_period_era, time_of_day, season_cycle, weather_conditions, mood_lighting, color_grading, atmospheric_effects
- STYLE Actions (1): artistic_medium

Instructions:
STEP 1: ANALYZE & REASON (Theme/Style Focus)
- Review the IMAGE ANALYSIS provided (if available) - what's in the image
- What is the current THEME (setting, mood, atmosphere, time)?
- What is the current STYLE (photorealistic, painting, artistic)?
- What THEME or STYLE does the user want?
- What actions would create a BOLD, NOTICEABLE transformation?
- Prioritize dramatic changes over subtle adjustments

STEP 2: PLAN ACTIONS WITH REASONING
- Select 2-5 appropriate actions from the library
- Focus on THEME transformations (scene/setting) and/or STYLE transformations (artistic rendering)
- Each action must contribute to a NOTICEABLE change
- For EACH action, provide "reasoning" explaining WHY it's needed for THIS image
- Set clear priorities (1=highest)
- **CRITICAL: Keep parameter 'description' SHORT (5-10 words max)**
- NO action-level 'description' field (only inside parameters)

PER-ACTION REASONING REQUIREMENTS:
- Reference SPECIFIC image elements (objects, composition, current style from analysis)
- Explain WHY this action creates theme/style transformation
- Mention dependencies (e.g., "location_setting must come first to establish scene")
- Keep concise (1-2 sentences, ~20-40 words)
- Avoid generic statements ("to improve" or "to enhance")

STEP 3: GENERATE EDIT PROMPT (hidream_prompt)
- Create a concise instruction for the image editor
- **MAXIMUM 77 TOKENS** (STRICT limit due to CLIP tokenizer)
- **MUST start with "style_transformation_mode"**
- If changing STYLE: "style_transformation_mode Apply {{artistic_style}} with {{characteristics}}"
- If changing THEME only: "style_transformation_mode Transform to {{theme}}. Maintain photorealistic quality."
- Focus on BOLD, noticeable changes

Examples of GOOD theme/style transformations:
✅ THEME: "Transform urban street to tropical beach" (location + weather + mood)
✅ STYLE: "Apply watercolor painting style with soft edges" (artistic medium)
✅ THEME: "Make it a moonlit night scene with eerie atmosphere" (time + mood + lighting)

Examples of BAD (too subtle):
❌ "Adjust brightness slightly"
❌ "Tweak color saturation"

Examples of GOOD per-action reasoning:
✅ "The modern glass buildings define this as contemporary urban. Complete location transformation to beach is the foundation, as all subsequent tropical elements depend on removing city infrastructure first."
✅ "Current photorealistic rendering doesn't match the artistic goal. Watercolor medium adds the dreamlike quality requested, transforming the visual style entirely."

Examples of GOOD parameter descriptions:
✅ "Change autumn foliage to spring greenery" (6 words)
✅ "Add clear sunny weather" (4 words)
✅ "Replace houses with beach huts" (5 words)

Examples of GOOD hidream_prompts:
✅ "style_transformation_mode Transform city to tropical beach with palm trees and sunny weather. Maintain photorealistic quality." (18 tokens)
✅ "style_transformation_mode Apply watercolor painting style with soft edges and flowing colors." (14 tokens)
✅ "style_transformation_mode Change autumn to spring with green foliage and blooming flowers. Preserve photorealistic style." (17 tokens)

Output ONLY valid JSON in this EXACT format:

{{
  "reasoning": "First, I observe [current image state]. The user wants to [goal]. To achieve this, I will [approach]. This requires [actions] because [justification]. I will preserve [elements] to maintain [quality].",
  "overall_instruction": "brief description",
  "actions": [
    {{
      "action_id": "location_setting",
      "reasoning": "The modern glass buildings and wet asphalt define this as urban. A complete location swap is required as foundation, since all other tropical elements depend on removing the city infrastructure first.",
      "parameters": {{
        "source_location": "urban_city",
        "target_location": "beach_coast",
        "replace_mode": "complete",
        "preserve_foreground": true,
        "description": "Transform city to beach"
      }},
      "priority": 1
    }},
    {{
      "action_id": "weather_conditions",
      "reasoning": "Clear tropical weather enhances the beach atmosphere and provides the bright, sunny lighting needed to match the user's 'tropical paradise' vision.",
      "parameters": {{
        "weather_type": "clear",
        "intensity": "moderate",
        "visibility": "high",
        "description": "Add clear tropical weather"
      }},
      "priority": 2
    }}
  ],
  "hidream_prompt": "style_transformation_mode [your concise instruction here, MAX 77 TOKENS]"
}}

CRITICAL: 
- "reasoning" MUST come FIRST in the JSON
- Each action MUST have its own "reasoning" field
- Think through the problem before selecting actions
- Output ONLY the JSON object."""

        # Format analysis for inclusion in prompt (if provided)
        analysis_text = ""
        if analysis:
            # Create a concise, structured summary of the analysis
            analysis_parts = []
            
            if "scene_type" in analysis:
                analysis_parts.append(f"Scene: {analysis['scene_type']}")
            
            if "objects" in analysis and analysis["objects"]:
                objects_list = [obj["name"] for obj in analysis["objects"][:10]]  # Top 10 objects
                analysis_parts.append(f"Objects: {', '.join(objects_list)}")
            
            if "colors" in analysis and analysis["colors"]:
                colors_list = [c["name"] for c in analysis["colors"][:5]]  # Top 5 colors
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
            
            if "composition" in analysis:
                comp = analysis["composition"]
                comp_parts = []
                if "layout" in comp:
                    comp_parts.append(f"layout: {comp['layout']}")
                if "focal_point" in comp:
                    comp_parts.append(f"focus: {comp['focal_point']}")
                if comp_parts:
                    analysis_parts.append(f"Composition: {', '.join(comp_parts)}")
            
            if analysis_parts:
                analysis_text = "\n\nIMAGE ANALYSIS:\n" + "\n".join(f"- {part}" for part in analysis_parts)

        # Create user message with analysis
        user_message = f"User request: {user_prompt}{analysis_text}\n\nGenerate an action plan to transform this image."
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": user_message},
                ],
            }
        ]
        
        # Process the vision info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=text, 
            images=image_inputs, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move all inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Generate action plan
        model_size = "30B" if "30B" in self.model_name else ("8B" if "8B" in self.model_name else "4B")
        print(f"  ⏳ Generating action plan with {model_size} model...")
        
        # Set random seed for reproducible variation
        # This ensures different action plans for the same prompt when cycling
        import time
        seed = int(time.time() * 1000) % (2**32)  # Use timestamp for variation
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,   # Enable sampling for variation
                temperature=0.7,  # Moderate temperature for diversity while maintaining quality
                top_p=0.9,
                repetition_penalty=1.1  # Reduce repetition in action plans
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse JSON response
        try:
            # For 30B-Thinking model, extract JSON from reasoning text
            clean_response = response.strip()
            
            # Try to find JSON in the response (30B-Thinking may wrap it in reasoning)
            if not clean_response.startswith('{'):
                # Look for JSON object in the response
                import re
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    clean_response = json_match.group(0)
                    print("  ✓ Extracted JSON from reasoning text")
                else:
                    print("  ⚠️ Could not find JSON in response")
            
            # Clean up markdown code blocks if present
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            action_plan = json.loads(clean_response)
            
            # Debug: Print the structure
            print(f"  🐛 DEBUG: action_plan keys: {list(action_plan.keys())}")
            print(f"  🐛 DEBUG: actions field: {action_plan.get('actions', 'NOT FOUND')}")
            
            # Validate actions against library (support both 'id' and 'action_id')
            valid_action_ids = {a.get('action_id', a.get('id')) for a in actions}
            for action in action_plan.get('actions', []):
                action_id = action.get('action_id', action.get('id'))
                if action_id not in valid_action_ids:
                    print(f"⚠️  Warning: Invalid action_id '{action_id}' in plan")
            
            print(f"  ✓ Action plan generated with {len(action_plan.get('actions', []))} actions")
            return action_plan
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Could not parse action plan JSON: {str(e)}")
            print(f"📄 Raw response: {response[:200]}...")
            
            # Fallback plan
            fallback_plan = {
                "overall_instruction": user_prompt,
                "actions": [
                    {
                        "action_id": "location_setting",
                        "parameters": {
                            "target_location": "beach_coast",
                            "preserve_foreground": True
                        },
                        "priority": 1,
                        "description": "Fallback action (parsing failed)"
                    }
                ],
                "error": f"Failed to parse JSON: {str(e)}",
                "raw_response": response
            }
            return fallback_plan
    
    def analyze_and_plan(self, image_path: str, user_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze image and generate action plan (convenience method)
        
        Args:
            image_path: Path to the image
            user_prompt: User's transformation request
            
        Returns:
            Tuple of (analysis, action_plan)
        """
        print(f"🔍 Analyzing and planning for: {os.path.basename(image_path)}")
        
        # Step 1: Analyze image
        analysis = self.analyze_image(image_path)
        
        # Step 2: Generate action plan
        action_plan = self.generate_action_plan(image_path, user_prompt, analysis)
        
        return analysis, action_plan
    
    def analyze_batch(self, image_paths: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple images and return batch results
        
        Args:
            image_paths: List of image file paths
            output_file: Optional file to save results
            
        Returns:
            List of analysis results
        """
        results = []
        
        print(f"🔍 Analyzing {len(image_paths)} images using Qwen3-VL...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\nAnalyzing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = self.analyze_image(image_path)
                results.append(result)
                print(f"✅ Successfully analyzed: {os.path.basename(image_path)}")
                
            except Exception as e:
                print(f"❌ Error analyzing {image_path}: {str(e)}")
                error_result = {
                    "image_id": os.path.basename(image_path),
                    "image_path": image_path,
                    "error": str(e),
                    "model_used": self.model_name,
                    "model_version": "qwen3-vl"
                }
                results.append(error_result)
        
        # Save results if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def __del__(self):
        """Cleanup method to free GPU memory"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def main():
    """Main function for testing the Qwen3-VL image analyzer"""
    import glob
    
    print("=" * 60)
    print("🧪 Qwen3-VL Image Analyzer Test")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ImageAnalyzerQwen3()
    
    # Find generated images
    image_dir = "../generated_images"
    if os.path.exists(image_dir):
        image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        
        if image_paths:
            print(f"\n📁 Found {len(image_paths)} images to analyze")
            
            # Analyze all images
            results = analyzer.analyze_batch(
                image_paths, 
                output_file="../analyzed_images/qwen3_analysis_results.json"
            )
            
            print(f"\n🎉 Analysis complete! Processed {len(results)} images.")
            print(f"📊 Success: {sum(1 for r in results if 'error' not in r)}/{len(results)}")
            
        else:
            print("⚠️  No images found in generated_images directory")
    else:
        print("⚠️  Generated images directory not found")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

