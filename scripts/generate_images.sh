#!/bin/bash

# HiDream-I1 Batch Image Generation Script
# Generates multiple images efficiently using batch processing

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ===== USER CONFIGURATION (Defaults - can be overridden by CLI args) =====
DEFAULT_PROMPTS_FILE="$PROJECT_ROOT/config/prompts.json"
DEFAULT_NUM_IMAGES=3
DEFAULT_OUTPUT_DIR="imageagent_results_10_29_2025"  # If empty, auto-generate: generation_output_TIMESTAMP
DEFAULT_GPU=1  # Which GPU to use (0-7), set to empty "" to use all available
# ========================================================================

# GPU Configuration
if [ -n "$DEFAULT_GPU" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$DEFAULT_GPU
    echo "🎮 Using GPU: $CUDA_VISIBLE_DEVICES"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
    echo "🎨 HiDream-I1 Batch Image Generator"
    echo "===================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --prompts FILE      Path to prompts.json (default: config/prompts.json)"
    echo "  -n, --num-images NUM    Number of images to generate (default: 5)"
    echo "  -o, --output-dir DIR    Output directory (default: auto-generated)"
    echo "  -g, --gpu ID            GPU ID to use (default: 1)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Generate 5 images with defaults"
    echo "  $0 -n 3                                   # Generate 3 images"
    echo "  $0 -o my_generation                       # Custom output directory"
    echo "  $0 -p custom_prompts.json -n 10          # Custom prompts, 10 images"
    echo "  $0 -g 2 -n 5                             # Use GPU 2, generate 5 images"
    echo ""
}

# Function to activate environment
activate_environment() {
    print_color $BLUE "🔧 Activating img-agent environment..."
    
    # First, deactivate any existing virtualenv to avoid conflicts
    if [ -n "$VIRTUAL_ENV" ]; then
        print_color $YELLOW "⚠️  Deactivating existing virtualenv: $VIRTUAL_ENV"
        unset VIRTUAL_ENV
        unset VIRTUAL_ENV_PROMPT
    fi
    
    # Source conda and activate
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate img-agent || {
        print_color $RED "❌ Failed to activate img-agent environment"
        exit 1
    }
    
    # Force PATH to prioritize conda environment
    export PATH="$(conda info --base)/envs/img-agent/bin:$PATH"
    
    # Verify correct Python
    PYTHON_PATH=$(which python)
    print_color $GREEN "✅ Environment activated!"
    print_color $CYAN "🐍 Using Python: $PYTHON_PATH"
}

# Main script logic
main() {
    # Initialize from configuration defaults
    local prompts_file="$DEFAULT_PROMPTS_FILE"
    local num_images=$DEFAULT_NUM_IMAGES
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local gpu_id="$DEFAULT_GPU"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -p|--prompts)
                prompts_file="$2"
                shift 2
                ;;
            -n|--num-images)
                num_images="$2"
                shift 2
                ;;
            -o|--output-dir)
                output_dir="$2"
                shift 2
                ;;
            -g|--gpu)
                gpu_id="$2"
                export CUDA_VISIBLE_DEVICES="$gpu_id"
                shift 2
                ;;
            -*)
                print_color $RED "❌ Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                print_color $RED "❌ Unexpected argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate prompts file exists
    if [ ! -f "$prompts_file" ]; then
        print_color $RED "❌ Prompts file not found: $prompts_file"
        exit 1
    fi
    
    # Validate num_images
    if ! [[ "$num_images" =~ ^[0-9]+$ ]] || [ "$num_images" -lt 1 ]; then
        print_color $RED "❌ Number of images must be a positive integer"
        exit 1
    fi
    
    # Show header
    print_color $PURPLE "🎨 HiDream-I1 Batch Image Generator"
    print_color $PURPLE "===================================="
    echo ""
    
    # Activate environment
    activate_environment
    
    # Show configuration
    print_color $CYAN "📝 Configuration:"
    print_color $CYAN "   Prompts file: $prompts_file"
    print_color $CYAN "   Number of images: $num_images"
    if [ -n "$output_dir" ]; then
        print_color $CYAN "   Output directory: $output_dir"
    else
        print_color $CYAN "   Output directory: (auto-generated)"
    fi
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        print_color $CYAN "   GPU: $CUDA_VISIBLE_DEVICES"
    fi
    echo ""
    
    # Build command
    local cmd="python $PROJECT_ROOT/src/batch_image_generator.py"
    cmd="$cmd --prompts \"$prompts_file\""
    cmd="$cmd --num-images $num_images"
    
    if [ -n "$output_dir" ]; then
        cmd="$cmd --output-dir \"$output_dir\""
    fi
    
    # Run the batch generator
    print_color $BLUE "🏃 Executing batch generation..."
    echo ""
    
    eval $cmd
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        print_color $GREEN "🎉 Image generation completed successfully!"
        
        # If output_dir was auto-generated, find it
        if [ -z "$output_dir" ]; then
            output_dir=$(ls -td $PROJECT_ROOT/generation_output_* 2>/dev/null | head -1)
        fi
        
        if [ -n "$output_dir" ]; then
            print_color $YELLOW "📁 Output location:"
            print_color $YELLOW "   $output_dir"
            echo ""
            print_color $YELLOW "📂 Generated folders:"
            ls -1 "$output_dir" | grep "^image_" | head -10
            if [ $(ls -1 "$output_dir" | grep "^image_" | wc -l) -gt 10 ]; then
                echo "   ..."
            fi
        fi
    else
        print_color $RED "❌ Image generation failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Run main function with all arguments
main "$@"

