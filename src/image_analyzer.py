#!/usr/bin/env python3
"""
Image Analysis Module using Qwen-VL
Analyzes generated images and creates JSON representations describing object locations and content
"""

import json
import os
from typing import Dict, List, Any
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class ImageAnalyzer:
    """Image analyzer using Qwen-VL for detailed image understanding"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """Initialize the Qwen-VL model for image analysis"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Qwen-VL model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("Qwen-VL model loaded successfully!")
    
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
        
        # Create comprehensive analysis prompt
        analysis_prompt = """
        Analyze this image in detail and provide a comprehensive JSON description with the following structure:
        
        {
            "image_id": "filename",
            "basic_info": {
                "dimensions": [width, height],
                "dominant_colors": ["color1", "color2", "color3"],
                "overall_style": "style description",
                "mood": "mood description"
            },
            "objects": [
                {
                    "name": "object name",
                    "description": "detailed description",
                    "location": "location description (e.g., center, left, top-right)",
                    "size": "relative size (small/medium/large)",
                    "confidence": 0.95
                }
            ],
            "scene_description": "detailed scene description",
            "composition": {
                "layout": "composition layout",
                "focal_point": "main focal point",
                "depth": "depth description",
                "lighting": "lighting description"
            },
            "style_analysis": {
                "art_style": "artistic style",
                "technique": "technique used",
                "color_palette": "color palette description",
                "texture": "texture description"
            },
            "editing_suggestions": [
                "suggestion 1",
                "suggestion 2", 
                "suggestion 3"
            ]
        }
        
        Provide only the JSON response, no additional text.
        """
        
        # Prepare messages for Qwen-VL
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
            text=text, images=image_inputs, return_tensors="pt", padding=True
        )
        
        # Move all inputs to the same device as the model
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Generate analysis
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            # Clean up response - remove markdown code blocks if present
            clean_response = response.strip()
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
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            print(f"Warning: Could not parse JSON response for {image_path}: {str(e)}")
            analysis_result = {
                "image_id": image_id,
                "image_path": image_path,
                "model_used": self.model_name,
                "raw_response": response,
                "error": f"Failed to parse JSON response: {str(e)}",
                "basic_info": {
                    "dimensions": list(image.size),
                    "format": image.format
                }
            }
        
        return analysis_result
    
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
        
        print(f"Analyzing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"Analyzing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = self.analyze_image(image_path)
                results.append(result)
                print(f"✓ Successfully analyzed: {os.path.basename(image_path)}")
                
            except Exception as e:
                print(f"✗ Error analyzing {image_path}: {str(e)}")
                error_result = {
                    "image_id": os.path.basename(image_path),
                    "image_path": image_path,
                    "error": str(e),
                    "model_used": self.model_name
                }
                results.append(error_result)
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        
        return results


def main():
    """Main function for testing the image analyzer"""
    import glob
    
    # Initialize analyzer
    analyzer = ImageAnalyzer()
    
    # Find generated images
    image_dir = "../generated_images"
    if os.path.exists(image_dir):
        image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        
        if image_paths:
            print(f"Found {len(image_paths)} images to analyze")
            
            # Analyze all images
            results = analyzer.analyze_batch(
                image_paths, 
                output_file="../analyzed_images/analysis_results.json"
            )
            
            print(f"\nAnalysis complete! Processed {len(results)} images.")
            
        else:
            print("No images found in generated_images directory")
    else:
        print("Generated images directory not found")


if __name__ == "__main__":
    main()
