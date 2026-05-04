#!/usr/bin/env python3
"""
Generate comparison tables from judge_combined_*.json files.
Produces 6 tables per experiment: image quality + action judge × 3 judges.

Usage:
    python scripts/evaluation/generate_judge_tables.py \
        --experiment-dir evaluation_results/simple/text_parallel_cot_4b_trajectory

    # All 4 experiments in simple:
    python scripts/evaluation/generate_judge_tables.py \
        --dataset-dir evaluation_results/simple
"""

import json
import argparse
from pathlib import Path
from typing import Optional, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available")

JUDGES = ["gpt54", "gemini", "claude_opus"]
JUDGE_LABELS = {"gpt54": "GPT-5.4", "gemini": "Gemini 2.5 Flash", "claude_opus": "Claude Opus 4.6"}

BASELINE_SHORT = {
    "baseline": "B",
    "edit_only": "E",
    "standard_text": "S",
    "rl_text": "R",
    "rw_text": "RW",
    "dpo_text": "D",
    "sw_text": "SW",
    "gpt4o": "GPT-4o",
}

IMAGE_METRICS = [
    ("Instruction Following", "instruction_following"),
    ("Visual Quality",        "visual_quality"),
    ("Transformation Strength", "transformation_strength"),
    ("Coherence",             "coherence"),
    ("Semantic Accuracy",     "semantic_accuracy"),
    ("Technical Execution",   "technical_execution"),
    ("Overall Image Score",   "overall_image_score"),
]

ACTION_METRICS = [
    ("Relevance",              "relevance"),
    ("Theme / Style Focus",    "theme_style_focus"),
    ("Completeness",           "completeness"),
    ("Efficiency",             "efficiency"),
    ("Correctness",            "correctness"),
    ("Reasoning Conciseness",  "reasoning_conciseness"),
    ("Reasoning Completeness", "reasoning_completeness"),
    ("Reasoning Specificity",  "reasoning_specificity"),
    ("Overall Action Quality", "overall_action_quality"),
    ("Overall Reasoning Quality", "overall_reasoning_quality"),
    ("Overall Score",          "overall_score"),
]

# Styling (matches existing gpt4o tables)
HEADER_COLOR  = "#2E86AB"
HEADER_TEXT   = "white"
ROW_COLORS    = ["white", "#F5F5F5"]
WINNER_COLOR  = "#E8F4F8"
BEST_COLOR    = "#2E86AB"


def load_combined(experiment_dir: Path, judge_id: str) -> Optional[dict]:
    f = experiment_dir / f"judge_combined_{judge_id}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def fmt(v, decimals=2) -> str:
    if v == "N/A" or v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def find_best(values: list) -> int:
    valid = [(i, v) for i, v in enumerate(values) if isinstance(v, float)]
    return max(valid, key=lambda x: x[1])[0] if valid else -1


def make_table(title: str, col_headers: List[str], rows: List[list],
               best_positions: set, output_path: Path):
    """Render a matplotlib table and save as PNG."""
    n_rows = len(rows)
    n_cols = len(col_headers) + 1  # +1 for metric name col + winner col
    fig_h = max(3.5, 0.45 * (n_rows + 1) + 1.0)
    fig, ax = plt.subplots(figsize=(max(10, 1.3 * n_cols), fig_h))
    ax.axis("off")

    # Build cell data: [metric_name, val1, val2, ..., winner_label]
    cell_data = []
    winner_labels = []
    for row_idx, (metric_label, *values, winner) in enumerate(rows):
        cell_data.append([metric_label] + [fmt(v) for v in values] + [winner])
        winner_labels.append(winner)

    all_headers = ["Metric"] + col_headers + ["Winner"]
    tbl = ax.table(
        cellText=cell_data,
        colLabels=all_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.0)

    n_data_cols = len(col_headers)
    # Style header
    for col in range(len(all_headers)):
        cell = tbl[(0, col)]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(weight="bold", color=HEADER_TEXT, fontsize=10)

    # Style data rows
    for row in range(1, n_rows + 1):
        row_color = ROW_COLORS[(row - 1) % 2]
        for col in range(len(all_headers)):
            cell = tbl[(row, col)]
            if col == 0:
                cell.set_facecolor(row_color)
                cell.set_text_props(weight="bold", fontsize=9.5)
            elif col == len(all_headers) - 1:  # winner col
                cell.set_facecolor(WINNER_COLOR)
                cell.set_text_props(weight="bold", color=HEADER_COLOR)
            else:
                cell.set_facecolor(row_color)
            # Bold best value
            if (row - 1, col - 1) in best_positions:
                cell.set_text_props(weight="bold", color=HEADER_COLOR, fontsize=10.5)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Legend
    legend = "  |  ".join(f"{s} = {l}" for s, l in BASELINE_SHORT.items()
                           if s in [h for h in col_headers])
    fig.text(0.5, 0.01, legend, ha="center", fontsize=8, color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_path.name}")


def generate_tables_for_experiment(experiment_dir: Path):
    experiment_dir = Path(experiment_dir)
    out_dir = experiment_dir / "consolidated_text"
    out_dir.mkdir(exist_ok=True)

    if not HAS_MATPLOTLIB:
        print("❌ matplotlib required")
        return

    for judge_id in JUDGES:
        data = load_combined(experiment_dir, judge_id)
        if data is None:
            print(f"  ⚠️  Missing judge_combined_{judge_id}.json — skipping")
            continue

        judge_label = JUDGE_LABELS[judge_id]
        exp_name = experiment_dir.name

        # Collect baseline columns (in order, only those present)
        baselines = [b for b in BASELINE_SHORT if b in data]
        col_headers = [BASELINE_SHORT[b] for b in baselines]

        # ── Image quality table ────────────────────────────────────────────
        rows = []
        best_positions = set()
        for row_idx, (metric_label, metric_key) in enumerate(IMAGE_METRICS):
            values = []
            for b in baselines:
                v = data[b].get("image_scores", {}).get(metric_key)
                values.append(float(v) if v is not None else None)
            best_i = find_best(values)
            if best_i >= 0:
                best_positions.add((row_idx, best_i))
            winner = col_headers[best_i] if best_i >= 0 else "-"
            rows.append([metric_label] + values + [winner])

        title = f"{judge_label} Image Quality Assessment\n({exp_name})"
        make_table(title, col_headers, rows, best_positions,
                   out_dir / f"{judge_id}_image_quality_table.png")

        # ── Action judge table ─────────────────────────────────────────────
        rows = []
        best_positions = set()
        for row_idx, (metric_label, metric_key) in enumerate(ACTION_METRICS):
            values = []
            for b in baselines:
                # edit_only has no action scores
                if b == "edit_only":
                    values.append("N/A")
                else:
                    v = data[b].get("action_scores", {}).get(metric_key)
                    values.append(float(v) if v is not None else None)
            numeric = [(i, v) for i, v in enumerate(values) if isinstance(v, float)]
            best_i = max(numeric, key=lambda x: x[1])[0] if numeric else -1
            if best_i >= 0:
                best_positions.add((row_idx, best_i))
            winner = col_headers[best_i] if best_i >= 0 else "-"
            rows.append([metric_label] + values + [winner])

        title = f"{judge_label} Action Plan Assessment\n({exp_name})"
        make_table(title, col_headers, rows, best_positions,
                   out_dir / f"{judge_id}_action_judge_table.png")

    print(f"Done: {experiment_dir.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Single experiment dir")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Dataset dir — generates tables for all 4 base experiments")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    base_exps = [
        "text_parallel_cot_4b_trajectory",
        "text_parallel_cot_8b_trajectory",
        "vision_parallel_cot_4b_trajectory",
        "vision_parallel_cot_8b_trajectory",
    ]

    if args.experiment_dir:
        d = Path(args.experiment_dir)
        if not d.is_absolute():
            d = project_root / d
        print(f"\n📊 {d.name}")
        generate_tables_for_experiment(d)

    elif args.dataset_dir:
        d = Path(args.dataset_dir)
        if not d.is_absolute():
            d = project_root / d
        for exp in base_exps:
            exp_dir = d / exp
            if exp_dir.exists():
                print(f"\n📊 {exp_dir.name}")
                generate_tables_for_experiment(exp_dir)
            else:
                print(f"⚠️  Not found: {exp_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
