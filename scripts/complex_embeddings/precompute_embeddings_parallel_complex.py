#!/usr/bin/env python3
"""
Parallel Image Embeddings Pre-computation (Qwen2-VL-2B + GPU-specific)
"""

import os
import sys
import json
import h5py
import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up 2 levels: complex_embeddings -> scripts -> project root
sys.path.insert(0, str(PROJECT_ROOT))


def extract_vision_embeddings(model, processor, image_path: str):
    """Extract vision embeddings from a single image."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Create minimal conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "describe"}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            if hasattr(model, 'visual'):
                vision_outputs = model.visual(
                    inputs['pixel_values'],
                    grid_thw=inputs['image_grid_thw']
                )
            elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
                vision_outputs = model.model.visual(
                    inputs['pixel_values'],
                    grid_thw=inputs['image_grid_thw']
                )
            else:
                outputs = model(**inputs, output_hidden_states=True)
                vision_outputs = outputs.hidden_states[0]
        
        # Handle tuple output
        if isinstance(vision_outputs, tuple):
            vision_outputs = vision_outputs[0]
        
        # Convert to CPU and float32 for storage
        embeddings = vision_outputs.cpu().float()
        return embeddings.squeeze(0)  # Remove batch dimension
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def precompute_embeddings_parallel(
    data_path: str,
    output_dir: str,
    gpu_id: int,
    dataset_name: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
):
    """GPU-specific embedding computation for parallel processing."""
    
    print(f"🚀 GPU {gpu_id}: Starting {dataset_name} embedding computation")
    print(f"   Model: {model_name}")
    print(f"   Data: {data_path}")
    print(f"   Output: {output_dir}")
    
    # Set GPU device
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            print(f"❌ GPU {gpu_id} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
            sys.exit(1)
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"   Device: {device}")
        print(f"   GPU Name: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = "cpu"
        print(f"   Device: {device} (CUDA not available)")
    
    # Load training data
    print(f"\nLoading training data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"Found {len(samples)} samples")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model on GPU {gpu_id}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": device},  # Use the specified GPU device
        trust_remote_code=True
    )
    model.eval()
    print(f"✓ Model loaded on device: {model.device}")
    
    # Process samples
    results = []
    project_root = PROJECT_ROOT  # Use the global PROJECT_ROOT (already calculated correctly at top of file)
    
    # Check if this is DPO data (has metadata with chosen_id/rejected_id)
    is_dpo_data = len(samples) > 0 and 'metadata' in samples[0] and 'chosen_id' in samples[0]['metadata']
    
    if is_dpo_data:
        print(f"🎯 Detected DPO dataset - processing chosen and rejected samples separately")
        
        # For DPO data, we need to process both chosen and rejected images
        unique_images = {}  # To avoid duplicates
        
        for sample in samples:
            metadata = sample['metadata']
            chosen_id = metadata['chosen_id']
            rejected_id = metadata['rejected_id']
            
            # Add chosen image
            if chosen_id not in unique_images:
                chosen_folder = metadata['chosen_folder']
                chosen_path = project_root / f"{chosen_folder}/original.png"
                unique_images[chosen_id] = chosen_path
            
            # Add rejected image
            if rejected_id not in unique_images:
                rejected_folder = metadata['rejected_folder']
                rejected_path = project_root / f"{rejected_folder}/original.png"
                unique_images[rejected_id] = rejected_path
        
        print(f"Found {len(unique_images)} unique images from {len(samples)} DPO pairs")
        
        # Process unique images
        for image_id, image_path in tqdm(unique_images.items(), desc=f"GPU {gpu_id} - {dataset_name}"):
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            
            # Extract embeddings
            embeddings = extract_vision_embeddings(model, processor, str(image_path))
            
            if embeddings is not None:
                results.append({
                    'sample_id': image_id,
                    'embedding': embeddings.numpy(),
                    'image_path': str(image_path)
                })
            else:
                print(f"Failed to extract embeddings for: {image_id}")
    
    else:
        # Standard processing for non-DPO data
        print(f"\nProcessing {len(samples)} images...")
        for sample in tqdm(samples, desc=f"GPU {gpu_id} - {dataset_name}"):
            sample_id = sample['id']
            image_path_str = sample.get('image_path', sample.get('image'))
            image_path = project_root / image_path_str
            
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            
            # Extract embeddings
            embeddings = extract_vision_embeddings(model, processor, str(image_path))
            
            if embeddings is not None:
                results.append({
                    'sample_id': sample_id,
                    'embedding': embeddings.numpy(),
                    'image_path': str(image_path)
                })
            else:
                print(f"Failed to extract embeddings for: {sample_id}")
    
    # Save to HDF5
    h5_path = output_path / "vision_embeddings.h5"
    print(f"\nSaving {len(results)} embeddings to: {h5_path}")
    
    manifest = {}
    
    with h5py.File(h5_path, 'w') as h5f:
        for result in tqdm(results, desc=f"GPU {gpu_id} - Writing HDF5"):
            sample_id = result['sample_id']
            embedding_data = result['embedding']
            
            h5f.create_dataset(
                sample_id,
                data=embedding_data,
                compression="gzip",
                compression_opts=4
            )
            
            manifest[sample_id] = {
                "image_path": result['image_path'],
                "shape": list(embedding_data.shape)
            }
    
    # Save manifest
    manifest_path = output_path / "embeddings_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ GPU {gpu_id}: Completed {dataset_name}!")
    print(f"   Saved {len(results)} embeddings")
    print(f"   HDF5 file: {h5_path}")
    print(f"   Manifest: {manifest_path}")
    
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Parallel embedding pre-computation")
    parser.add_argument("--data-path", required=True, help="Path to training data JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--gpu-id", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--dataset-name", required=True, help="Dataset name for logging")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct", help="Model name")
    
    args = parser.parse_args()
    
    # Validate GPU
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            print(f"❌ GPU {args.gpu_id} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
            sys.exit(1)
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Run embedding computation
    num_embeddings = precompute_embeddings_parallel(
        args.data_path,
        args.output_dir,
        args.gpu_id,
        args.dataset_name,
        args.model_name
    )
    
    print(f"\n🎉 GPU {args.gpu_id}: Successfully computed {num_embeddings} embeddings for {args.dataset_name}")


if __name__ == "__main__":
    main()
