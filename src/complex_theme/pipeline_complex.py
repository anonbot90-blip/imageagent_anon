#!/usr/bin/env python3
"""
Complex Theme Pipeline Wrapper
Extends the main pipeline to use BatchImageGeneratorComplex for complex theme prompts
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Import main pipeline
from pipeline import ImageAgentPipeline

# Import complex batch generator
from complex_theme.batch_image_generator_complex import BatchImageGeneratorComplex


class ImageAgentPipelineComplex(ImageAgentPipeline):
    """
    Extended pipeline for complex theme dataset
    Uses BatchImageGeneratorComplex for image generation
    """
    
    def generate_images_with_structure(self, 
                                      num_images: int,
                                      output_dir: str,
                                      prompts_file: str = None) -> Dict:
        """
        Generate images using complex theme batch generator with folder structure
        
        Args:
            num_images: Number of images to generate
            output_dir: Base output directory
            prompts_file: Path to complex theme prompts.json
            
        Returns:
            Manifest dictionary with image data
        """
        print(f"🎨 Generating {num_images} new complex theme images...")
        
        # Default prompts file for complex theme
        if prompts_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            prompts_file = os.path.join(base_dir, 'config', 'complex_prompts', 'prompts_complex_theme_400.json')
        
        # Initialize complex batch generator
        generator = BatchImageGeneratorComplex(
            prompts_file=prompts_file,
            output_base_dir=output_dir,
            gpu_id=os.environ.get('CUDA_VISIBLE_DEVICES')
        )
        
        # Generate images
        manifest_path = generator.generate_batch(num_images=num_images)
        
        # Load and return manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        print(f"✅ Generated {len(manifest['images'])} complex theme images with folder structure")
        return manifest


# Make it easy to use the complex pipeline as a drop-in replacement
def create_complex_pipeline(model_analyzer=None, model_planner=None, model_editor=None, 
                           hidream_checkpoint=None, model_reward=None, action_library_path=None):
    """
    Factory function to create a complex theme pipeline instance
    
    Args:
        Same as ImageAgentPipeline.__init__
        
    Returns:
        ImageAgentPipelineComplex instance
    """
    return ImageAgentPipelineComplex(
        model_analyzer=model_analyzer,
        model_planner=model_planner,
        model_editor=model_editor,
        hidream_checkpoint=hidream_checkpoint,
        model_reward=model_reward,
        action_library_path=action_library_path
    )


if __name__ == "__main__":
    # When run directly, use the main pipeline's CLI with complex support
    import argparse
    
    parser = argparse.ArgumentParser(description='ImageAgent Complex Theme Pipeline')
    parser.add_argument('edit_prompt', help='The editing instruction')
    parser.add_argument('--num-images', '-n', type=int, default=5, help='Number of images to process')
    parser.add_argument('--generate-new', '-g', action='store_true', help='Generate new images')
    parser.add_argument('--output-dir', '-o', help='Output directory for results')
    parser.add_argument('--prompts-file', '-pf', help='Custom complex theme prompts file')
    parser.add_argument('--style-prompts', '-s', action='store_true', help='Use style transformation prompts')
    parser.add_argument('--model-analyzer', '-ma', type=str, required=True, help='Model for image analysis')
    parser.add_argument('--model-planner', '-mp', type=str, required=True, help='Model for action planning')
    parser.add_argument('--model-editor', '-me', type=str, required=True, choices=['qwen', 'hidream'], help='Image editor')
    parser.add_argument('--hidream-checkpoint', type=str, default=None, help='HiDream-E1 checkpoint path')
    parser.add_argument('--reward-model', '-rm', type=str, default=None, help='Model for reward evaluation')
    parser.add_argument('--action-library', '-al', type=str, default=None, help='Path to action library JSON')
    
    args = parser.parse_args()
    
    try:
        # Initialize complex pipeline
        pipeline = create_complex_pipeline(
            model_analyzer=args.model_analyzer,
            model_planner=args.model_planner,
            model_editor=args.model_editor,
            hidream_checkpoint=args.hidream_checkpoint,
            model_reward=args.reward_model,
            action_library_path=args.action_library
        )
        
        # Determine prompts file
        prompts_file = args.prompts_file
        if args.style_prompts and not prompts_file:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            prompts_file = os.path.join(base_dir, 'config', 'complex_prompts', 'prompts_complex_theme_400.json')
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            edit_prompt=args.edit_prompt,
            num_images=args.num_images,
            generate_new=args.generate_new,
            output_dir=args.output_dir,
            prompts_file=prompts_file
        )
        
        print(f"\n✅ Complex theme pipeline completed successfully!")
        print(f"📂 Results saved to: {results['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

