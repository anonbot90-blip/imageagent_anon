#!/usr/bin/env python3
"""
Test script for Qwen3-VL Image Analyzer
Tests model loading, inference, and compares with Qwen2-VL if available
"""

import os
import sys
import json
import time
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_analyzer_qwen3 import ImageAnalyzerQwen3


def print_separator(title=""):
    """Print a nice separator"""
    if title:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print("-" * 70)


def check_gpu_memory():
    """Check and display GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🔋 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("⚠️  No GPU available")


def test_model_loading():
    """Test 1: Model Loading"""
    print_separator("Test 1: Model Loading")
    
    try:
        print("🔧 Attempting to load Qwen3-VL-4B-Instruct...")
        start_time = time.time()
        
        analyzer = ImageAnalyzerQwen3()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        check_gpu_memory()
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure transformers is up to date:")
        print("      pip install --upgrade transformers")
        print("   2. Check if you have enough GPU memory:")
        print("      nvidia-smi")
        print("   3. Try with CPU (slower): device_map='cpu'")
        return None


def test_single_image_inference(analyzer, test_image_path):
    """Test 2: Single Image Inference"""
    print_separator("Test 2: Single Image Inference")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return None
    
    print(f"📸 Testing with: {test_image_path}")
    
    try:
        start_time = time.time()
        
        result = analyzer.analyze_image(test_image_path)
        
        inference_time = time.time() - start_time
        print(f"✅ Analysis completed in {inference_time:.2f} seconds")
        
        check_gpu_memory()
        
        return result
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def validate_json_structure(result):
    """Test 3: Validate JSON Structure"""
    print_separator("Test 3: JSON Structure Validation")
    
    if result is None:
        print("❌ No result to validate")
        return False
    
    required_fields = [
        "image_id", "image_path", "model_used", "model_version"
    ]
    
    optional_fields = [
        "basic_info", "objects", "scene_description", 
        "composition", "style_analysis", "editing_suggestions"
    ]
    
    print("✓ Checking required fields...")
    all_present = True
    for field in required_fields:
        if field in result:
            print(f"  ✅ {field}: present")
        else:
            print(f"  ❌ {field}: MISSING")
            all_present = False
    
    print("\n✓ Checking optional fields...")
    for field in optional_fields:
        if field in result:
            print(f"  ✅ {field}: present")
        else:
            print(f"  ⚠️  {field}: missing (optional)")
    
    if "error" in result:
        print(f"\n⚠️  Error in result: {result['error']}")
        if "raw_response" in result:
            print(f"📄 Raw response preview:")
            print(result['raw_response'][:300] + "...")
        return False
    
    return all_present


def display_analysis_summary(result):
    """Display a summary of the analysis"""
    print_separator("Analysis Summary")
    
    if result is None or "error" in result:
        print("❌ Cannot display summary - analysis failed")
        return
    
    print(f"🆔 Image ID: {result.get('image_id', 'N/A')}")
    print(f"🤖 Model: {result.get('model_used', 'N/A')}")
    print(f"📦 Version: {result.get('model_version', 'N/A')}")
    
    if "basic_info" in result:
        basic = result["basic_info"]
        print(f"\n📊 Basic Info:")
        print(f"   Style: {basic.get('overall_style', 'N/A')}")
        print(f"   Mood: {basic.get('mood', 'N/A')}")
        print(f"   Colors: {', '.join(basic.get('dominant_colors', []))}")
    
    if "objects" in result:
        print(f"\n🎯 Objects Detected: {len(result['objects'])}")
        for i, obj in enumerate(result['objects'][:3], 1):
            print(f"   {i}. {obj.get('name', 'N/A')} at {obj.get('location', 'N/A')}")
    
    if "composition" in result:
        comp = result["composition"]
        print(f"\n🎨 Composition:")
        print(f"   Focal Point: {comp.get('focal_point', 'N/A')}")
        print(f"   Lighting: {comp.get('lighting', 'N/A')[:60]}...")
    
    if "editing_suggestions" in result:
        print(f"\n💡 Editing Suggestions:")
        for i, sug in enumerate(result['editing_suggestions'][:3], 1):
            print(f"   {i}. {sug}")


def save_test_results(result, output_path):
    """Save test results to file"""
    print_separator("Saving Results")
    
    if result is None:
        print("❌ No results to save")
        return
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")


def main():
    """Main test runner"""
    print("\n" + "=" * 70)
    print("  🧪 QWEN3-VL IMAGE ANALYZER TEST SUITE")
    print("=" * 70)
    
    # Configuration
    test_image = None
    
    # Look for a test image in imageagent_results
    possible_paths = [
        "./path/to/image.png"  # update path,
        "./path/to/image.png"  # update path,
        "./path/to/image.png"  # update path,
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image is None:
        print("❌ No test image found. Please provide a path.")
        print("   Looked in:")
        for path in possible_paths:
            print(f"   - {path}")
        sys.exit(1)
    
    print(f"\n📸 Test image: {test_image}")
    check_gpu_memory()
    
    # Run tests
    analyzer = test_model_loading()
    if analyzer is None:
        print("\n❌ Test suite failed: Could not load model")
        sys.exit(1)
    
    result = test_single_image_inference(analyzer, test_image)
    
    is_valid = validate_json_structure(result)
    
    if is_valid:
        display_analysis_summary(result)
    
    # Save results
    output_path = "./path/to/image.png"  # update path
    save_test_results(result, output_path)
    
    # Final summary
    print_separator("Test Summary")
    print(f"✅ Model Loading: PASSED")
    print(f"{'✅' if result else '❌'} Inference: {'PASSED' if result else 'FAILED'}")
    print(f"{'✅' if is_valid else '❌'} JSON Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    if is_valid:
        print("\n🎉 All tests PASSED!")
        print("💡 Next steps:")
        print("   1. Compare with Qwen2-VL output")
        print("   2. Test with multiple images")
        print("   3. Integrate into pipeline.py")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    print("=" * 70 + "\n")
    
    # Cleanup
    del analyzer
    torch.cuda.empty_cache()
    check_gpu_memory()


if __name__ == "__main__":
    main()

