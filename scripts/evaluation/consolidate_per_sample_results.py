#!/usr/bin/env python3
"""
Consolidate per-sample evaluation results from multiple models
Creates comparison images and consolidated detailed JSON
"""

import json
import shutil
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def normalize_gpt4o_metrics(data):
    """Normalize GPT-4o's metric names to match standard naming."""
    if isinstance(data, dict):
        # Planner metrics normalization
        if 'planner_action_f1_mean' not in data and 'f1_mean' in data:
            data['planner_action_f1_mean'] = data.pop('f1_mean')
        if 'planner_action_precision_mean' not in data and 'precision_mean' in data:
            data['planner_action_precision_mean'] = data.pop('precision_mean')
        if 'planner_action_recall_mean' not in data and 'recall_mean' in data:
            data['planner_action_recall_mean'] = data.pop('recall_mean')
        if 'planner_sequence_accuracy_mean' not in data and 'sequence_accuracy_mean' in data:
            data['planner_sequence_accuracy_mean'] = data.pop('sequence_accuracy_mean')
        
        # Recursively normalize nested dicts
        for key, value in list(data.items()):
            if isinstance(value, dict):
                normalize_gpt4o_metrics(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        normalize_gpt4o_metrics(item)
    return data


def create_comparison_image(labels_list, paths_list, output_path, user_prompt=None):
    """Create a side-by-side comparison image with labels and optional prompt."""
    # Load images
    loaded_images = []
    target_height = 512
    
    for path, label in zip(paths_list, labels_list):
        if path.exists():
            img = Image.open(path).convert('RGB')
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            img = img.resize((new_width, target_height), Image.LANCZOS)
            loaded_images.append(img)
        else:
            # Placeholder if image missing
            placeholder = Image.new('RGB', (target_height, target_height), (200, 200, 200))
            draw_temp = ImageDraw.Draw(placeholder)
            draw_temp.text((target_height//4, target_height//2), "Missing", fill=(100, 100, 100))
            loaded_images.append(placeholder)
    
    # Create canvas with space for labels and prompt
    label_height = 85  # For 48pt bold labels
    prompt_height = 240 if user_prompt else 20  # For 42pt bold prompt
    total_width = sum(img.width for img in loaded_images) + 20 * (len(loaded_images) - 1)  # 20px gaps
    total_height = target_height + label_height + prompt_height
    
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts (48pt bold for labels, 42pt bold for prompt)
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_prompt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    except:
        font_large = ImageFont.load_default()
        font_prompt = ImageFont.load_default()
    
    # Paste images and add labels
    x_offset = 0
    for img, label in zip(loaded_images, labels_list):
        # Paste image
        canvas.paste(img, (x_offset, label_height))
        
        # Add label above image (centered, bold)
        bbox = draw.textbbox((0, 0), label, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 20), label, fill="black", font=font_large)
        
        x_offset += img.width + 20
    
    # Add user prompt at bottom if available (bold, large)
    if user_prompt:
        prompt_y = target_height + label_height + 15
        # Wrap text for 42pt font
        max_chars_per_line = 90  # Adjusted for larger font
        words = user_prompt.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw prompt lines with bold font (max 4 lines, 52px line height)
        for i, line in enumerate(lines[:4]):
            draw.text((15, prompt_y + i * 52), line, fill="black", font=font_prompt)
    
    # Save comparison image
    canvas.save(output_path)


def consolidate_per_sample_results(
    baseline_dir: Path,
    edit_only_dir: Path,
    standard_text_dir: Path,
    rl_text_dir: Path,
    rw_text_dir: Path,
    dpo_text_dir: Path,
    sw_text_dir: Path,
    gpt4o_dir: Path,
    output_dir: Path
):
    """Consolidate per-sample results from all models."""
    # Create output samples directory
    output_samples_dir = output_dir / "samples"
    output_samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample directories from baseline
    baseline_samples = baseline_dir / "samples"
    if not baseline_samples.exists():
        raise FileNotFoundError(f"Samples directory not found: {baseline_samples}")
    
    sample_dirs = sorted([d for d in baseline_samples.iterdir() if d.is_dir()])
    print(f"Found {len(sample_dirs)} samples to consolidate")
    print("")
    
    # Process each sample
    for i, baseline_sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples"), 1):
        sample_id = baseline_sample_dir.name
        
        # Create output sample directory
        output_sample_dir = output_samples_dir / sample_id
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Get corresponding directories for other models
        edit_only_sample_dir = edit_only_dir / "samples" / sample_id
        standard_text_sample_dir = standard_text_dir / "samples" / sample_id
        rl_text_sample_dir = rl_text_dir / "samples" / sample_id
        rw_text_sample_dir = rw_text_dir / "samples" / sample_id
        dpo_text_sample_dir = dpo_text_dir / "samples" / sample_id
        sw_text_sample_dir = sw_text_dir / "samples" / sample_id
        gpt4o_sample_dir = gpt4o_dir / "samples" / sample_id
        
        # Copy original and ground truth (same for all models)
        for filename in ["original.png", "ground_truth.png"]:
            src = baseline_sample_dir / filename
            if src.exists():
                shutil.copy2(src, output_sample_dir / filename)
        
        # Copy results from each model
        model_configs = [
            ("baseline", baseline_sample_dir),
            ("edit_only", edit_only_sample_dir),
            ("standard_text", standard_text_sample_dir),
            ("rl_text", rl_text_sample_dir),
            ("rw_text", rw_text_sample_dir),
            ("dpo_text", dpo_text_sample_dir),
            ("sw_text", sw_text_sample_dir),
            ("gpt4o", gpt4o_sample_dir),
        ]
        
        for model_name, model_sample_dir in model_configs:
            for filename in ["predicted_edit.png", "predicted_plan.json"]:
                src = model_sample_dir / filename
                if src.exists():
                    # Rename predicted_plan.json to plan.json for consistency
                    dst_name = filename.replace("predicted_plan", "plan")
                    dst = output_sample_dir / f"{model_name}_{dst_name}"
                    shutil.copy2(src, dst)
        
        # Get user prompt for this sample
        user_prompt = None
        baseline_plan_file = output_sample_dir / "baseline_plan.json"
        if baseline_plan_file.exists():
            try:
                with open(baseline_plan_file, 'r') as f:
                    plan_data = json.load(f)
                    # Try both 'overall_instruction' and 'user_prompt' fields
                    user_prompt = plan_data.get('overall_instruction', plan_data.get('user_prompt', ''))
            except:
                pass
        
        # Create 10-way comparison (with Ground Truth)
        # [Original | Baseline | Edit-Only | Standard | RL | DPO | RW | SW | GPT-4o | Ground Truth]
        try:
            labels_10way = ['Original', 'Baseline', 'Edit-Only', 'Standard', 'RL', 'DPO', 'RW', 'SW', 'GPT-4o', 'Ground Truth']
            paths_10way = [
                output_sample_dir / "original.png",
                output_sample_dir / "baseline_predicted_edit.png",
                output_sample_dir / "edit_only_predicted_edit.png",
                output_sample_dir / "standard_text_predicted_edit.png",
                output_sample_dir / "rl_text_predicted_edit.png",
                output_sample_dir / "dpo_text_predicted_edit.png",
                output_sample_dir / "rw_text_predicted_edit.png",
                output_sample_dir / "sw_text_predicted_edit.png",
                output_sample_dir / "gpt4o_predicted_edit.png",
                output_sample_dir / "ground_truth.png"
            ]
            create_comparison_image(labels_10way, paths_10way, output_sample_dir / "comparison_10way.png", user_prompt)
            
            # Create 9-way comparison (without Ground Truth)
            labels_9way = ['Original', 'Baseline', 'Edit-Only', 'Standard', 'RL', 'DPO', 'RW', 'SW', 'GPT-4o']
            paths_9way = [
                output_sample_dir / "original.png",
                output_sample_dir / "baseline_predicted_edit.png",
                output_sample_dir / "edit_only_predicted_edit.png",
                output_sample_dir / "standard_text_predicted_edit.png",
                output_sample_dir / "rl_text_predicted_edit.png",
                output_sample_dir / "dpo_text_predicted_edit.png",
                output_sample_dir / "rw_text_predicted_edit.png",
                output_sample_dir / "sw_text_predicted_edit.png",
                output_sample_dir / "gpt4o_predicted_edit.png"
            ]
            create_comparison_image(labels_9way, paths_9way, output_sample_dir / "comparison_9way.png", user_prompt)
            
        except Exception as e:
            print(f"    ⚠️  Failed to create comparison image for {sample_id}: {e}")
    
    print(f"\n✅ Consolidated {len(sample_dirs)} samples")


def create_consolidated_detailed_json(
    baseline_dir: Path,
    edit_only_dir: Path,
    standard_text_dir: Path,
    rl_text_dir: Path,
    rw_text_dir: Path,
    dpo_text_dir: Path,
    sw_text_dir: Path,
    gpt4o_dir: Path,
    output_dir: Path
):
    """Create consolidated_detailed.json by merging all detailed results."""
    # Load detailed results - files are named evaluation_results_all.json
    # Try both naming conventions for compatibility
    def load_results_file(model_dir: Path, model_name: str = ""):
        """Try to load results file with fallback naming."""
        # Try evaluation_results_all.json first (current format)
        file1 = model_dir / "evaluation_results_all.json"
        if file1.exists():
            data = json.load(open(file1))
            # Extract results array if it exists
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
            else:
                return []
        
        # Fallback to detailed_results_all.json (old format)
        file2 = model_dir / "detailed_results_all.json"
        if file2.exists():
            data = json.load(open(file2))
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
            else:
                return []
        
        # Try evaluation_results_test.json for edit_only
        if model_name == "edit_only":
            file3 = model_dir / "evaluation_results_test.json"
            if file3.exists():
                data = json.load(open(file3))
                if isinstance(data, dict) and "results" in data:
                    return data["results"]
                elif isinstance(data, list):
                    return data
        
        return []
    
    baseline_detailed = load_results_file(baseline_dir)
    edit_only_detailed = load_results_file(edit_only_dir, "edit_only")
    standard_text_detailed = load_results_file(standard_text_dir)
    rl_text_detailed = load_results_file(rl_text_dir)
    rw_text_detailed = load_results_file(rw_text_dir)
    dpo_text_detailed = load_results_file(dpo_text_dir)
    sw_text_detailed = load_results_file(sw_text_dir)
    gpt4o_detailed = load_results_file(gpt4o_dir)
    
    # Detect experiment type from directory structure
    # Check if this is vision or text by looking at checkpoint/model names
    # Default to text_only_8model, but can be overridden
    experiment_type = "text_only_8model"
    if "vision" in str(standard_text_dir).lower() or "vision" in str(output_dir).lower():
        experiment_type = "vision_8model"
    
    # Create consolidated detailed results
    consolidated_detailed = {
        "experiment_type": experiment_type,
        "num_samples": len(baseline_detailed),
        "samples": []
    }
    
    # Merge per-sample results
    for b_sample in baseline_detailed:
        sample_id = b_sample["sample_id"]
        
        # Find corresponding samples in other models
        e_sample = next((s for s in edit_only_detailed if s["sample_id"] == sample_id), None)
        st_sample = next((s for s in standard_text_detailed if s["sample_id"] == sample_id), None)
        rt_sample = next((s for s in rl_text_detailed if s["sample_id"] == sample_id), None)
        rw_sample = next((s for s in rw_text_detailed if s["sample_id"] == sample_id), None)
        dpo_sample = next((s for s in dpo_text_detailed if s["sample_id"] == sample_id), None)
        sw_sample = next((s for s in sw_text_detailed if s["sample_id"] == sample_id), None)
        gpt4o_sample = next((s for s in gpt4o_detailed if s["sample_id"] == sample_id), None)
        
        merged_sample = {
            "sample_id": sample_id,
            "user_prompt": b_sample.get("user_prompt", "N/A"),
            "baseline": b_sample,
            "edit_only": e_sample if e_sample else {},
            "standard_text": st_sample if st_sample else {},
            "rl_text": rt_sample if rt_sample else {},
            "rw_text": rw_sample if rw_sample else {},
            "dpo_text": dpo_sample if dpo_sample else {},
            "sw_text": sw_sample if sw_sample else {},
            "gpt4o": gpt4o_sample if gpt4o_sample else {}
        }
        
        consolidated_detailed["samples"].append(merged_sample)
    
    # Apply normalization to GPT-4o data in consolidated detailed results
    for sample in consolidated_detailed['samples']:
        if 'gpt4o' in sample and sample['gpt4o']:
            normalize_gpt4o_metrics(sample['gpt4o'])
    
    # Save consolidated detailed results
    output_file = output_dir / "consolidated_detailed.json"
    with open(output_file, 'w') as f:
        json.dump(consolidated_detailed, f, indent=2)
    
    print(f"✅ Created consolidated_detailed.json with {len(consolidated_detailed['samples'])} samples")
    print(f"   Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate per-sample evaluation results from multiple models"
    )
    parser.add_argument("--baseline-dir", type=str, required=True, help="Baseline evaluation results directory")
    parser.add_argument("--edit-only-dir", type=str, required=True, help="Edit-Only evaluation results directory")
    parser.add_argument("--standard-text-dir", type=str, required=True, help="Standard Text evaluation results directory")
    parser.add_argument("--rl-text-dir", type=str, required=True, help="RL Text evaluation results directory")
    parser.add_argument("--rw-text-dir", type=str, required=True, help="RW Text evaluation results directory")
    parser.add_argument("--dpo-text-dir", type=str, required=True, help="DPO Text evaluation results directory")
    parser.add_argument("--sw-text-dir", type=str, required=True, help="SW Text evaluation results directory")
    parser.add_argument("--gpt4o-dir", type=str, required=True, help="GPT-4o Planner evaluation results directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for consolidated results")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    baseline_dir = Path(args.baseline_dir)
    edit_only_dir = Path(args.edit_only_dir)
    standard_text_dir = Path(args.standard_text_dir)
    rl_text_dir = Path(args.rl_text_dir)
    rw_text_dir = Path(args.rw_text_dir)
    dpo_text_dir = Path(args.dpo_text_dir)
    sw_text_dir = Path(args.sw_text_dir)
    gpt4o_dir = Path(args.gpt4o_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directories exist
    for name, dir_path in [
        ("baseline", baseline_dir),
        ("edit_only", edit_only_dir),
        ("standard_text", standard_text_dir),
        ("rl_text", rl_text_dir),
        ("rw_text", rw_text_dir),
        ("dpo_text", dpo_text_dir),
        ("sw_text", sw_text_dir),
        ("gpt4o", gpt4o_dir),
    ]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {name} = {dir_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("════════════════════════════════════════════════════════════════════════════════")
    print("  Consolidating Per-Sample Results")
    print("════════════════════════════════════════════════════════════════════════════════")
    print("")
    
    # Step 1: Consolidate per-sample files and create comparison images
    print("Step 1: Copying files and creating comparison images...")
    consolidate_per_sample_results(
        baseline_dir, edit_only_dir, standard_text_dir, rl_text_dir,
        rw_text_dir, dpo_text_dir, sw_text_dir, gpt4o_dir, output_dir
    )
    
    print("")
    
    # Step 2: Create consolidated_detailed.json
    print("Step 2: Creating consolidated_detailed.json...")
    create_consolidated_detailed_json(
        baseline_dir, edit_only_dir, standard_text_dir, rl_text_dir,
        rw_text_dir, dpo_text_dir, sw_text_dir, gpt4o_dir, output_dir
    )
    
    print("")
    print("✅ Per-sample consolidation complete!")


if __name__ == "__main__":
    main()

