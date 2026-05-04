# Data Generation Guide

Complete guide for generating ImageAgent datasets.

## 📊 Dataset Directory Mapping

The ImageAgent project uses three main datasets, each stored in specific directories:

| Dataset Directory | Dataset Name | Description | Size |
|------------------|--------------|-------------|------|
| `imageagent_results_10000_cot` | **simple** | Basic image transformations, 10K samples | 10,000 |
| `imageagent_results_normal_cot` | **normal** | Complex transformations, 10K samples | 10,000 |
| `imageagent_results_complex_v2_10k_cot` | **complex** | Advanced transformations, 10K samples | 10,000 |

**Note:** These directories are large and excluded from file searches. Do not rename them without updating all references.

## 🎨 Image Generation Scripts

### Location
All data generation scripts are located in `/src/`:

- `batch_image_generator.py` - Simple dataset generation
- `complex_theme/batch_image_generator_complex.py` - Complex dataset generation
- `pipeline.py` - Main pipeline orchestrator
- `complex_theme/pipeline_complex.py` - Complex theme pipeline wrapper

## 🚀 Generating Simple Dataset

The simple dataset uses basic image transformations.

### Using Batch Generator

```bash
python src/batch_image_generator.py \
    --prompts config/prompts.json \
    --output-dir imageagent_results_10000_cot \
    --num-images 10000 \
    --gpu 0
```

### Using Pipeline

```bash
python src/pipeline.py \
    "Transform to modern style" \
    --num-images 10000 \
    --generate-new \
    --output-dir imageagent_results_10000_cot \
    --model-editor qwen
```

### Parameters

- `--prompts`: Path to prompts JSON file
- `--output-dir`: Output directory (will be created)
- `--num-images`: Number of images to generate
- `--gpu`: GPU ID to use (optional)

## 🎭 Generating Normal Dataset

The normal dataset uses complex transformations (10K samples).

### Using Complex Batch Generator

```bash
python src/complex_theme/batch_image_generator_complex.py \
    --prompts config/complex_prompts/prompts_complex_theme_400.json \
    --output-dir imageagent_results_normal_cot \
    --num-images 10000 \
    --gpu 0
```

### Using Complex Pipeline

```bash
python src/complex_theme/pipeline_complex.py \
    "Transform to Victorian style with candlelit lighting" \
    --num-images 10000 \
    --generate-new \
    --output-dir imageagent_results_normal_cot \
    --model-editor qwen
```

## 🔬 Generating Complex Dataset

The complex dataset uses advanced transformations (10K samples).

### Using Complex Batch Generator

```bash
python src/complex_theme/batch_image_generator_complex.py \
    --prompts config/complex_prompts/prompts_complex_v2_400.json \
    --output-dir imageagent_results_complex_v2_10k_cot \
    --num-images 10000 \
    --gpu 0
```

## 📁 Output Structure

Each dataset directory contains:

```
imageagent_results_XXXXX_cot/
├── image_<hash>_<idx>_<theme>/
│   ├── original.png              # Generated image
│   ├── edited.png                # Edited image (if applicable)
│   ├── analysis.json             # Image analysis
│   ├── action_plan.json          # Action plan
│   └── metadata.json             # Sample metadata
├── manifest.json                  # Generation manifest
└── generation_log.json           # Generation log
```

## 🔧 Pipeline Components

### 1. Image Generation (HiDream-I1)
- Generates base images from text prompts
- Supports multiple styles/themes
- Batch processing for efficiency

### 2. Image Analysis (Qwen-VL)
- Analyzes generated images
- Extracts scene descriptions
- Identifies objects and relationships

### 3. Action Planning (VLM-based)
- Generates action plans for transformations
- Uses fine-tuned models (4B/8B variants)
- Supports text-only and vision modes

### 4. Image Editing
- **Qwen-Image-Edit**: Direct editing model
- **HiDream-E1**: Alternative editing model
- Applies transformations based on action plans

## 📝 Prompt Formats

### Simple Dataset Prompts

```json
{
  "prompts": [
    {
      "id": 1,
      "text": "A modern living room with minimalist furniture",
      "style": "modern_living_room",
      "model": "fast",
      "resolution": "1024x1024"
    }
  ]
}
```

### Complex Dataset Prompts

```json
{
  "prompts": [
    {
      "id": 1,
      "text": "A modern living room with minimalist furniture",
      "theme": "modern_living_room",
      "model": "fast",
      "resolution": "1024x1024",
      "edit_info": {
        "text": "Transform to Victorian style AND add candlelit lighting",
        "expected_actions": ["architecture_style", "mood_lighting"]
      }
    }
  ]
}
```

## ⚙️ Configuration

### Environment Setup

```bash
# Activate conda environment
conda activate img-agent

# Set GPU
export CUDA_VISIBLE_DEVICES=0
```

### Model Paths

- **HiDream-I1**: `HiDream-I1/` (for generation)
- **HiDream-E1**: `HiDream-E1/` (for editing)
- **Qwen-VL**: Auto-downloaded from HuggingFace

## 🔍 Verification

After generation, verify the dataset:

```bash
# Count samples
find imageagent_results_10000_cot -type d -name "image_*" | wc -l

# Check manifest
cat imageagent_results_10000_cot/manifest.json | jq '.total_images'

# Verify structure
ls imageagent_results_10000_cot/image_*/original.png | wc -l
```

## 📊 Dataset Statistics

After generation, each dataset should contain:

- **Simple**: ~10,000 samples
- **Normal**: ~10,000 samples
- **Complex**: ~10,000 samples

Each sample includes:
- Original image
- Analysis results
- Action plan
- Metadata

## 🚨 Important Notes

1. **Directory Names**: Do not rename dataset directories. They are referenced throughout the codebase.

2. **Large Files**: Dataset directories are excluded from file searches due to size.

3. **GPU Memory**: Generation requires significant GPU memory (16GB+ recommended).

4. **Time**: Full dataset generation can take hours/days depending on size and GPU.

5. **Storage**: Each dataset requires significant disk space (100GB+ per dataset).

## 🔗 Next Steps

After generating datasets:

1. **Split Data**: Create train/test splits (see [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md))
2. **Generate Training Data**: Extract training samples (see [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md))
3. **Train Models**: Fine-tune action planners (see [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md))
4. **Evaluate**: Run evaluations (see [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md))

---

**Last Updated:** 2026-02-21

