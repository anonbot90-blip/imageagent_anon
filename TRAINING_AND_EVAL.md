# Training and Evaluation Guide

Complete guide for training action planner models and evaluating their performance.

## 📊 Dataset Mapping

| Dataset Directory | Dataset Name | Training Scripts | Evaluation Scripts |
|------------------|--------------|------------------|-------------------|
| `imageagent_results_10000_cot` | **simple** | `scripts/training/simple/` | `scripts/evaluation/simple/` |
| `imageagent_results_normal_cot` | **normal** | `scripts/training/normal/` | `scripts/evaluation/normal/` |
| `imageagent_results_complex_v2_10k_cot` | **complex** | `scripts/training/complex/` | `scripts/evaluation/complex/` |

## 🎓 Training Workflow

### Overview

Training involves:
1. **Data Preparation**: Split datasets and generate training data
2. **Model Training**: Fine-tune Qwen-VL models with different methods
3. **Checkpoint Management**: Save and manage model checkpoints

### Training Methods

Five training methods are supported:

| Method | Description | Training Script |
|--------|-------------|----------------|
| **Standard** | Baseline fine-tuning | `generate_standard_trajectory.sh` |
| **RL** | Reinforcement learning (reward ≥ 3.0) | `generate_rl_trajectory.sh` |
| **RW** | Reward-weighted training (reward ≥ 3.5) | `generate_rw_trajectory.sh` |
| **DPO** | Direct preference optimization | `generate_dpo_trajectory.sh` |
| **SW** | Standardized weighted training | `generate_sw_trajectory.sh` |

### Full Pipeline Scripts

Each dataset has full pipeline scripts that orchestrate the entire workflow:

#### Simple Dataset

```bash
# Text-only 4B model
bash scripts/simple/text_4b_run_full_pipeline.sh

# Text-only 8B model
bash scripts/simple/text_8b_run_full_pipeline.sh

# Vision 4B model
bash scripts/simple/vision_4b_run_full_pipeline.sh

# Vision 8B model
bash scripts/simple/vision_8b_run_full_pipeline.sh
```

#### Normal Dataset

```bash
# Text-only 4B model
bash scripts/normal/text_4b_run_full_pipeline.sh

# Text-only 8B model
bash scripts/normal/text_8b_run_full_pipeline.sh

# Vision 4B model
bash scripts/normal/vision_4b_run_full_pipeline.sh

# Vision 8B model
bash scripts/normal/vision_8b_run_full_pipeline.sh
```

#### Complex Dataset

```bash
# Text-only 4B model
bash scripts/complex/text_4b_run_full_pipeline.sh

# Text-only 8B model
bash scripts/complex/text_8b_run_full_pipeline.sh

# Vision 4B model
bash scripts/complex/vision_4b_run_full_pipeline.sh

# Vision 8B model
bash scripts/complex/vision_8b_run_full_pipeline.sh
```

### Step-by-Step Training

#### Step 1: Create Train/Test Split

```bash
# Simple dataset
python scripts/training/simple/split_trajectories.py \
    --results-dir imageagent_results_10000_cot \
    --test-threshold 5.5 \
    --test-count 200 \
    --reward-metric overall_quality \
    --output-dir training_data/simple/cot_4b_trajectory \
    --prefix cot_4b
```

#### Step 2: Generate Training Data

```bash
# Set trajectory prefix
export TRAJECTORY_PREFIX="cot_4b"

# Generate Standard training data
RESULTS_DIR="imageagent_results_10000_cot" \
bash scripts/training/simple/generate_standard_trajectory.sh

# Generate RL training data
RESULTS_DIR="imageagent_results_10000_cot" \
bash scripts/training/simple/generate_rl_trajectory.sh

# Generate RW training data
RESULTS_DIR="imageagent_results_10000_cot" \
bash scripts/training/simple/generate_rw_trajectory.sh

# Generate DPO training data
RESULTS_DIR="imageagent_results_10000_cot" \
bash scripts/training/simple/generate_dpo_trajectory.sh

# Generate SW training data
RESULTS_DIR="imageagent_results_10000_cot" \
bash scripts/training/simple/generate_sw_trajectory.sh
```

#### Step 3: Train Models

```bash
cd training/planner_training

# Standard Training (4B)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29510 \
    train_planner_standard_trajectory_text.py \
    --config configs_simple_4b/planner_config_standard_trajectory_text.yaml

# RL Training (4B)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29511 \
    train_planner_rl_trajectory_text.py \
    --config configs_simple_4b/planner_config_rl_trajectory_text.yaml

# RW Training (4B)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29512 \
    train_planner_rw_trajectory_text.py \
    --config configs_simple_4b/planner_config_rw_trajectory_text.yaml

# DPO Training (4B)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29513 \
    train_planner_dpo_trajectory_text.py \
    --config configs_simple_4b/planner_config_dpo_trajectory_text.yaml

# SW Training (4B)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29518 \
    train_planner_sw_trajectory_text.py \
    --config configs_simple_4b/planner_config_sw_trajectory_text.yaml
```

### Checkpoint Locations

Checkpoints are saved in:

```
checkpoints/
├── simple/
│   ├── cot_4b_trajectory/
│   │   ├── standard/final/
│   │   ├── rl/final/
│   │   ├── rw/final/
│   │   ├── dpo/final/
│   │   └── sw/final/
│   └── cot_8b_trajectory/
│       └── ...
├── normal/
│   └── ...
└── complex/
    └── ...
```

## 📈 Evaluation Workflow

### Overview

Evaluation involves:
1. **Model Evaluation**: Run models on test samples
2. **Metric Calculation**: Compute action planning and image quality metrics
3. **Result Consolidation**: Aggregate results into comparison tables

### Evaluation Scripts

#### Simple Dataset

```bash
# Text-only 4B evaluation
bash scripts/evaluation/simple/text_4b_eval_all.sh

# Text-only 8B evaluation
bash scripts/evaluation/simple/text_8b_eval_all.sh

# Vision 4B evaluation
bash scripts/evaluation/simple/vision_4b_eval_all.sh

# Vision 8B evaluation
bash scripts/evaluation/simple/vision_8b_eval_all.sh
```

#### Normal Dataset

```bash
# Text-only 4B evaluation
bash scripts/evaluation/normal/text_4b_eval_all.sh

# Text-only 8B evaluation
bash scripts/evaluation/normal/text_8b_eval_all.sh

# Vision 4B evaluation
bash scripts/evaluation/normal/vision_4b_eval_all.sh

# Vision 8B evaluation
bash scripts/evaluation/normal/vision_8b_eval_all.sh
```

#### Complex Dataset

```bash
# Text-only 4B evaluation
bash scripts/evaluation/complex/text_4b_eval_all.sh

# Text-only 8B evaluation
bash scripts/evaluation/complex/text_8b_eval_all.sh

# Vision 4B evaluation
bash scripts/evaluation/complex/vision_4b_eval_all.sh

# Vision 8B evaluation
bash scripts/evaluation/complex/vision_8b_eval_all.sh
```

### Parallel Evaluation

Evaluation scripts support parallel execution across multiple GPUs:

```bash
# Uses 7 GPUs + GPT-4o API
bash scripts/evaluation/simple/text_4b_eval_all.sh
```

Models evaluated in parallel:
1. Baseline (Qwen3-VL zero-shot)
2. Edit-Only (direct editing, no planning)
3. Standard Text
4. RL Text
5. RW Text
6. DPO Text
7. SW Text
8. GPT-4o Planner (API-based)

### Evaluation Metrics

#### Action Planning Metrics
- **F1 Score**: Action plan accuracy
- **IoU**: Intersection over Union
- **Precision**: Action precision
- **Recall**: Action recall
- **Priority Correlation**: Priority alignment

#### Image Quality Metrics
- **LPIPS↓**: Perceptual distance (lower is better)
- **PSNR↑**: Peak signal-to-noise ratio (higher is better)
- **SSIM↑**: Structural similarity (higher is better)
- **CLIP Score↑**: Semantic similarity (higher is better)

#### GPT-4o Assessment
- **Action Planning Quality**: Completeness, correctness, efficiency, relevance
- **Image Quality**: Semantic accuracy, visual quality, coherence, technical execution

### Evaluation Outputs

Results are saved in:

```
evaluation_results/
├── simple/
│   ├── text_parallel_cot_4b_trajectory/
│   │   ├── baseline/
│   │   ├── edit_only/
│   │   ├── standard_text/
│   │   ├── rl_text/
│   │   ├── rw_text/
│   │   ├── dpo_text/
│   │   ├── sw_text/
│   │   ├── gpt4o/
│   │   └── consolidated_text/
│   │       ├── consolidated_summary.json
│   │       ├── consolidated_detailed.json
│   │       ├── gpt4o_action_judge_table.png
│   │       ├── gpt4o_image_quality_table.png
│   │       └── FINAL_SUMMARY.md
│   └── ...
├── normal/
│   └── ...
└── complex/
    └── ...
```

### Consolidation

After evaluation, consolidate results:

```bash
bash scripts/evaluation/consolidate_text_results.sh \
    --baseline-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/baseline \
    --edit-only-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/edit_only \
    --standard-text-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/standard_text \
    --rl-text-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/rl_text \
    --rw-text-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/rw_text \
    --dpo-text-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/dpo_text \
    --sw-text-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/sw_text \
    --gpt4o-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/gpt4o \
    --output-dir evaluation_results/simple/text_parallel_cot_4b_trajectory/consolidated_text
```

## ⚙️ Configuration Files

Evaluation configurations are in `scripts/evaluation/configs/`:

- `simple_text_4b.yaml`
- `simple_text_8b.yaml`
- `simple_vision_4b.yaml`
- `simple_vision_8b.yaml`
- `normal_text_4b.yaml`
- `normal_text_8b.yaml`
- `normal_vision_4b.yaml`
- `normal_vision_8b.yaml`
- `complex_text_4b.yaml`
- `complex_text_8b.yaml`
- `complex_vision_4b.yaml`
- `complex_vision_8b.yaml`

## 🔧 Requirements

### Training
- 8× GPUs (recommended)
- 16GB+ GPU memory per GPU
- Significant disk space for checkpoints
- Conda environment: `img-agent`

### Evaluation
- 7× GPUs for parallel evaluation
- GPT-4o API access (for GPT-4o planner)
- Conda environment: `img-agent`

## 📊 Results Analysis

After evaluation, analyze results:

```bash
# View consolidated summary
cat evaluation_results/simple/text_parallel_cot_4b_trajectory/consolidated_text/consolidated_summary.json | jq

# View detailed results
cat evaluation_results/simple/text_parallel_cot_4b_trajectory/consolidated_text/consolidated_detailed.json | jq

# View markdown report
cat evaluation_results/simple/text_parallel_cot_4b_trajectory/consolidated_text/FINAL_SUMMARY.md

# View comparison images
ls evaluation_results/simple/text_parallel_cot_4b_trajectory/consolidated_text/samples/*/comparison_9way.png
```

## 🚨 Important Notes

1. **GPU Resources**: Training requires 8 GPUs. Evaluation can use 7 GPUs in parallel.

2. **Time**: 
   - Training: Hours to days depending on dataset size
   - Evaluation: 30-60 minutes for test sets

3. **Checkpoints**: Always verify checkpoints exist before evaluation:
   ```bash
   ls checkpoints/simple/cot_4b_trajectory/*/final/
   ```

4. **Test Samples**: Ensure test split exists:
   ```bash
   ls training_data/simple/cot_4b_trajectory/test_samples_cot_4b.txt
   ```

5. **Environment**: Always activate conda environment:
   ```bash
   conda activate img-agent
   ```

## 🔗 Related Documentation

- [README.md](README.md) - Project overview
- [DATA_GENERATION.md](DATA_GENERATION.md) - Data generation guide
- [consolidated_results/README.md](consolidated_results/README.md) - Results analysis

---

**Last Updated:** 2026-02-21

