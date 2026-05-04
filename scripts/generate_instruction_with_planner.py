"""
Generate Instruction with Planner - Bridge planner predictions with context

This script:
1. Takes image + user prompt
2. Uses Qwen3-VL planner to predict action_plan.json
3. Loads analysis.json (detailed context)
4. Merges them using action_templates.py
5. Returns rich instruction for HiDream-E1
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from actions.planner_inference import create_planner


def load_analysis(analysis_path: str):
    """Load analysis.json with detailed image context."""
    if not os.path.exists(analysis_path):
        print(f"⚠️ Analysis file not found: {analysis_path}")
        return None
    
    with open(analysis_path, 'r') as f:
        return json.load(f)


def save_action_plan(action_plan, output_path: str):
    """Save predicted action plan to JSON."""
    with open(output_path, 'w') as f:
        json.dump(action_plan, indent=2, fp=f)
    print(f"✓ Action plan saved: {output_path}")


def generate_instruction(
    image_path: str,
    user_prompt: str,
    analysis_path: str = None,
    output_dir: str = None,
    planner_checkpoint: str = None,
    max_actions: int = 5,
    temperature: float = 0.7,
    use_planner: bool = True
) -> str:
    """
    Generate editing instruction using planner + context merger.
    
    Args:
        image_path: Path to input image
        user_prompt: User's editing request
        analysis_path: Path to analysis.json (optional, will auto-detect)
        output_dir: Where to save action_plan.json
        planner_checkpoint: Path to fine-tuned planner LoRA
        max_actions: Max actions to predict (1-5)
        temperature: Sampling temperature
        use_planner: If False, skip planner and use prompt directly
    
    Returns:
        Rich instruction string for HiDream-E1
    """
    print("=" * 60)
    print("🧠 Generating Instruction with Action Planner")
    print("=" * 60)
    
    # Auto-detect analysis path if not provided
    if analysis_path is None:
        image_dir = Path(image_path).parent
        analysis_path = image_dir / "analysis.json"
    
    # Load existing analysis (context)
    analysis = load_analysis(str(analysis_path))
    
    if not use_planner:
        print("⚠️ Planner disabled, using prompt directly")
        return user_prompt
    
    # Initialize planner
    print(f"\n1. Initializing Action Planner...")
    planner = create_planner(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        lora_checkpoint=planner_checkpoint,
        device="cuda"
    )
    
    # Predict action plan
    print(f"\n2. Predicting Action Plan...")
    print(f"   Image: {image_path}")
    print(f"   Prompt: {user_prompt}")
    
    action_plan = planner.predict_action_plan(
        image_path=image_path,
        user_prompt=user_prompt,
        max_actions=max_actions,
        temperature=temperature,
        return_reasoning=True
    )
    
    # Validate
    is_valid, msg = planner.validate_action_plan(action_plan)
    print(f"\n3. Validation: {msg}")
    
    if not is_valid:
        print(f"⚠️ Invalid action plan, using fallback")
    
    # Display predicted actions
    print(f"\n4. Predicted Actions:")
    for i, action in enumerate(action_plan.get("actions", [])):
        action_id = action.get("action_id", "unknown")
        priority = action.get("priority", "?")
        print(f"   [{i+1}] {action_id} (priority: {priority})")
    
    if "reasoning" in action_plan:
        print(f"\n   Reasoning: {action_plan['reasoning']}")
    
    # Save action plan
    if output_dir:
        output_path = Path(output_dir) / "action_plan.json"
        save_action_plan(action_plan, str(output_path))
    
    # Merge with context
    print(f"\n5. Merging Action Plan with Context...")
    
    if analysis:
        print(f"   ✓ Using detailed context from analysis.json")
        print(f"   ✓ Objects detected: {len(analysis.get('objects', []))}")
        print(f"   ✓ Spatial map available: {bool(analysis.get('spatial_map'))}")
    else:
        print(f"   ⚠️ No analysis.json found, using action plan only")
    
    # Use model-generated hidream_prompt
    instruction = action_plan.get("hidream_prompt", action_plan.get("overall_instruction", ""))
    
    # Display final instruction
    print(f"\n6. Generated Instruction:")
    print("=" * 60)
    print(instruction)
    print("=" * 60)
    
    return instruction


def main():
    parser = argparse.ArgumentParser(
        description="Generate editing instruction using action planner"
    )
    
    parser.add_argument(
        "image_path",
        help="Path to input image"
    )
    
    parser.add_argument(
        "prompt",
        help="User editing request"
    )
    
    parser.add_argument(
        "--analysis",
        help="Path to analysis.json (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory to save action_plan.json"
    )
    
    parser.add_argument(
        "--planner-checkpoint",
        help="Path to fine-tuned planner LoRA checkpoint"
    )
    
    parser.add_argument(
        "--max-actions",
        type=int,
        default=5,
        help="Maximum actions to predict (1-5)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy, 1.0 = creative)"
    )
    
    parser.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip planner, use prompt directly"
    )
    
    parser.add_argument(
        "--output-instruction",
        help="Path to save generated instruction"
    )
    
    args = parser.parse_args()
    
    # Generate instruction
    instruction = generate_instruction(
        image_path=args.image_path,
        user_prompt=args.prompt,
        analysis_path=args.analysis,
        output_dir=args.output_dir,
        planner_checkpoint=args.planner_checkpoint,
        max_actions=args.max_actions,
        temperature=args.temperature,
        use_planner=not args.no_planner
    )
    
    # Save instruction if requested
    if args.output_instruction:
        with open(args.output_instruction, 'w') as f:
            f.write(instruction)
        print(f"\n✓ Instruction saved: {args.output_instruction}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

