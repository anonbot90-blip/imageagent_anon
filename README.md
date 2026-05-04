# ImageAgent

A complete pipeline for image generation, analysis, editing, and action planning using vision-language models.

## 📋 Overview

ImageAgent is an end-to-end system that:
1. **Generates** images using HiDream-I1
2. **Analyzes** images using Qwen-VL models
3. **Plans** actions using fine-tuned vision-language models
4. **Edits** images using Qwen-Image-Edit (20B) — HiDream-E1 is also supported as an alternative
5. **Evaluates** results using GPT-4o and standard metrics

## 🗂️ Project Structure

```
ImageAgent/
├── src/                    # Core pipeline scripts
│   ├── batch_image_generator.py
│   ├── pipeline.py
│   ├── image_analyzer.py
│   ├── image_editor.py
│   └── complex_theme/      # Complex dataset support
│
├── scripts/                # Training and evaluation scripts
│   ├── training/          # Training data generation and model training
│   │   ├── simple/        # Simple dataset training
│   │   ├── normal/        # Normal dataset training
│   │   └── complex/       # Complex dataset training
│   ├── evaluation/        # Model evaluation scripts
│   │   ├── simple/        # Simple dataset evaluation
│   │   ├── normal/        # Normal dataset evaluation
│   │   └── complex/       # Complex dataset evaluation
│   ├── simple/            # Simple dataset full pipelines
│   ├── normal/            # Normal dataset full pipelines
│   └── complex/           # Complex dataset full pipelines
│
├── training/              # Training code
├── checkpoints/          # Model checkpoints
├── evaluation_results/   # Evaluation outputs
├── consolidated_results/ # Cross-dataset analysis
└── human_eval/          # Human evaluation data
```

## 📚 Documentation

- **[DATA_GENERATION.md](DATA_GENERATION.md)** - How to generate datasets
- **[TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md)** - Training and evaluation workflows

## 🚀 Quick Start

### 1. Data Generation

Generate images and create datasets:

```bash
# Simple dataset
python src/batch_image_generator.py \
    --prompts config/prompts.json \
    --output-dir imageagent_results_10000_cot \
    --num-images 10000

# Normal dataset
python src/complex_theme/batch_image_generator_complex.py \
    --prompts config/complex_prompts/prompts_complex_theme_400.json \
    --output-dir imageagent_results_normal_cot \
    --num-images 10000
```

See [DATA_GENERATION.md](DATA_GENERATION.md) for detailed instructions.

### 2. Training

Train action planner models:

```bash
# Simple dataset - Text-only 4B
bash scripts/simple/text_4b_run_full_pipeline.sh

# Normal dataset - Vision 8B
bash scripts/normal/vision_8b_run_full_pipeline.sh

# Complex dataset - Text-only 4B
bash scripts/complex/text_4b_run_full_pipeline.sh
```

See [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md) for detailed training instructions.

### 3. Evaluation

Evaluate trained models:

```bash
# Simple dataset evaluation
bash scripts/evaluation/simple/text_4b_eval_all.sh

# Normal dataset evaluation
bash scripts/evaluation/normal/text_4b_eval_all.sh

# Complex dataset evaluation
bash scripts/evaluation/complex/text_4b_eval_all.sh
```

See [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md) for detailed evaluation instructions.

## 📊 Dataset Mapping

The project uses three main datasets:

| Dataset Directory | Dataset Name | Description |
|------------------|--------------|-------------|
| `imageagent_results_10000_cot` | **simple** | 10K samples, 1–2 step atomic edits |
| `imageagent_results_normal_cot` | **normal (Regular)** | 10K samples, 3–5 step compositional edits with 10 interior design themes |
| `imageagent_results_complex_v2_10k_cot` | **complex** | 10K samples, 3–5 step compositional edits with 83 diverse themes |

> **Note:** The **normal** dataset in this codebase corresponds to the **Regular** dataset referred to in the paper. The names are used interchangeably.

## 🔧 Requirements

- Python 3.10+
- CUDA-capable GPU (8+ GPUs recommended for training)
- Conda (Miniconda or Anaconda)
- HiDream-I1 (image generation)
- Qwen-Image-Edit (image editing, default) or HiDream-E1 (alternative editor)
- Qwen-VL models (4B/8B variants, for analysis and action planning)

## ⚙️ Environment Setup

Set up the conda environment with all required dependencies:

```bash
# Run the setup script
bash setup_environment.sh
```

This will:
1. Create a conda environment named `img-agent` from `environment.yml`
2. Install PyTorch with CUDA 12.4 support
3. Install Flash Attention (optional, may take time to compile)
4. Install additional packages (bitsandbytes, qwen-vl-utils, h5py, wandb, matplotlib)
5. Verify all critical packages are installed

**After setup:**
```bash
# Activate the environment
conda activate img-agent

# Or if conda is not initialized in your shell:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate img-agent
```

**Next steps:**
1. Login to HuggingFace: `huggingface-cli login`
2. Accept Llama 3.1 license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
3. Verify installation: `python -c 'from training.planner_training.train_planner_standard_trajectory_text import load_config; print("✅ OK")'`

## 📖 Key Components

### Data Generation
- **HiDream-I1**: Image generation model
- **Qwen-Image-Edit (20B)**: Primary image editing model (default: `--model-editor qwen`)
- **Batch generators**: Efficient multi-image generation
- **Pipeline orchestrator**: End-to-end data creation

### Training
- **Standard**: Baseline fine-tuning
- **RL**: Reinforcement learning with reward filtering
- **RW**: Reward-weighted training
- **DPO**: Direct preference optimization
- **SW**: Standardized weighted training

### Evaluation
- **GPT-4o**: Action planning and image quality assessment
- **Standard metrics**: LPIPS, PSNR, SSIM, CLIP Score
- **Planner metrics**: F1, IoU, Precision, Recall
- **Parallel evaluation**: Multi-GPU evaluation support

## 🔗 Related Documentation

- [DATA_GENERATION.md](DATA_GENERATION.md) - Complete data generation guide
- [TRAINING_AND_EVAL.md](TRAINING_AND_EVAL.md) - Training and evaluation workflows
- [consolidated_results/README.md](consolidated_results/README.md) - Results analysis

## 📝 Notes

- All dataset directories are large and excluded from searches
- Training requires significant GPU resources (8× GPUs recommended)
- Evaluation can run in parallel across multiple GPUs
- Checkpoints are saved in `checkpoints/{dataset}/{model_size}/{method}/`

---

**Last Updated:** 2026-02-21

