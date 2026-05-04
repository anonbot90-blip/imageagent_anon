#!/usr/bin/env python3
"""
Generate LaTeX tables for the NeurIPS 2026 resubmission.

Reads `judge_combined_{judge_id}.json` under each
`evaluation_results/<dataset>/<experiment>/` directory and produces:

  (1) One LaTeX table per configuration with three-judge columns side-by-side.
  (2) A flat CSV/markdown summary of overall image scores for quick reference.

Usage:
    python scripts/evaluation/generate_multi_judge_latex.py \
        --results-root evaluation_results \
        --out-dir latex/neurips2026/generated_tables
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

JUDGES = ["gpt54", "gemini_31fl"]
JUDGE_LABELS = {
    "gpt54": "GPT-5.4",
    "gemini_31fl": "Gemini 3.1",
}

# method_key -> (short_label, latex_macro_or_plain)
METHODS = [
    ("baseline",      "B",   "B"),
    ("edit_only",     "E",   "E"),
    ("standard_text", "S",   "S"),
    ("rl_text",       "R",   r"\rlfull"),
    ("rw_text",       "RW",  r"\rw"),
    ("sw_text",       "SW",  r"\sw"),
    ("dpo_text",      "D",   r"\dpo"),
    ("gemini25",      "G2.5", "Gemini 2.5"), # zero-shot reference (planner)
]

TRAINED_METHOD_KEYS = {"baseline", "edit_only", "standard_text", "rl_text",
                       "rw_text", "sw_text", "dpo_text"}
ZERO_SHOT_REF_KEYS = {"gemini25"}

DATASETS = [
    ("simple",  "Simple"),
    ("normal",  "Regular"),
    ("complex", "Complex"),
]

CONFIGS = [
    ("text_parallel_cot_4b_trajectory",   "Text-4B"),
    ("text_parallel_cot_8b_trajectory",   "Text-8B"),
    ("vision_parallel_cot_4b_trajectory", "Vision-4B"),
    ("vision_parallel_cot_8b_trajectory", "Vision-8B"),
]


def load_combined(exp_dir: Path, judge_id: str) -> Optional[dict]:
    f = exp_dir / f"judge_combined_{judge_id}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def score_for(combined: dict, method_key: str) -> Optional[float]:
    if combined is None:
        return None
    m = combined.get(method_key)
    if m is None:
        return None
    v = m.get("image_scores", {}).get("overall_image_score")
    if v is None:
        return None
    # Stored on 0-10 scale; paper uses 0-100 scale.
    return float(v) * 10.0


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.2f}"


def best_trained_index(scores: List[Optional[float]]) -> int:
    """Return index of best score among trained methods only (exclude GPT-4o)."""
    best_i, best_v = -1, float("-inf")
    for i, (mk, _, _) in enumerate(METHODS):
        if mk not in TRAINED_METHOD_KEYS:
            continue
        v = scores[i]
        if v is None:
            continue
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def build_config_table(
    dataset_key: str,
    dataset_label: str,
    exp_key: str,
    config_label: str,
    results_root: Path,
) -> Optional[str]:
    """Build a LaTeX table for one (dataset, config) cell, three-judge columns."""
    exp_dir = results_root / dataset_key / exp_key
    if not exp_dir.exists():
        return None

    # Preload combined for all 3 judges
    combined_by_judge = {j: load_combined(exp_dir, j) for j in JUDGES}
    if all(c is None for c in combined_by_judge.values()):
        return None

    # scores[judge][method_idx] = float or None
    scores = {j: [score_for(combined_by_judge[j], mk) for (mk, _, _) in METHODS]
              for j in JUDGES}

    best_idx = {j: best_trained_index(scores[j]) for j in JUDGES}

    # Modality + size label: keep both "text" and "vision" in key
    mod = "text" if "text_parallel" in exp_key else "vision"
    size = exp_key.split("_cot_")[1].split("_trajectory")[0]  # "4b" or "8b"
    label_key = f"tab:mj_{dataset_key}_{mod}_{size}"
    caption = (
        f"\\textbf{{{dataset_label} {config_label}}}: "
        f"Image quality (0--100) under two independent judges. "
        f"Bold = best trained method (B/E/S/R/RW/SW/D) per judge; "
        f"Gemini 2.5 (grey) is a zero-shot reference planner."
    )

    # Header
    col_spec = "l" + "c" * len(JUDGES)
    judge_hdr = " & ".join(JUDGE_LABELS[j] for j in JUDGES)
    header = (
        "\\begin{table}[h]\n"
        "\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label_key}}}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"Method & {judge_hdr} \\\\\n"
        "\\midrule\n"
    )

    body_lines = []
    _midrule_placed = False
    for i, (mk, short, latex_label) in enumerate(METHODS):
        # Zero-shot reference rows (GPT-4o, Gemini 2.5) styled grey after a separating \midrule
        if mk in ZERO_SHOT_REF_KEYS:
            if not _midrule_placed:
                body_lines.append("\\midrule")
                _midrule_placed = True
            grey = lambda x: f"\\textcolor{{gray}}{{{x}}}"
            row = f"{grey(latex_label)} & " + " & ".join(
                grey(fmt(scores[j][i])) for j in JUDGES
            ) + " \\\\"
            body_lines.append(row)
            continue

        cells = []
        for j in JUDGES:
            val = fmt(scores[j][i])
            if best_idx[j] == i:
                cells.append(f"\\textbf{{{val}}}")
            else:
                cells.append(val)
        # Bold label if it wins any judge
        if any(best_idx[j] == i for j in JUDGES):
            label_cell = f"\\textbf{{{latex_label}}}"
        else:
            label_cell = latex_label
        body_lines.append(f"{label_cell} & " + " & ".join(cells) + " \\\\")

    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    return header + "\n".join(body_lines) + "\n" + footer


def build_consolidated_table(
    judge_id: str,
    results_root: Path,
) -> Optional[str]:
    """Build a single ICML-style consolidated table for one judge.

    Rows = 12 (config, dataset), grouped by dataset via \\multicolumn dividers.
    Cols = 7 trained methods (B/E/S/R/RW/SW/D) + grey Gemini 2.5 (zero-shot).
    Bold = best trained method per row.
    """
    judge_label = JUDGE_LABELS[judge_id]

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n\\small\n"
        f"\\caption{{\\textbf{{Overall Image Quality under {judge_label} judge "
        f"across all 12 configurations.}} Bold = best trained method (B/E/S/R/RW/SW/D) per row; "
        f"Gemini 2.5 (grey) is a zero-shot reference planner.}}\n"
        f"\\label{{tab:mj_all_{judge_id}}}\n"
        "\\begin{tabular}{lccccccc|c}\n"
        "\\toprule\n"
        "\\textbf{Configuration} & \\textbf{B} & \\textbf{E} & \\textbf{S} & "
        "\\textbf{\\rlfull} & \\textbf{\\rw} & \\textbf{\\sw} & \\textbf{\\dpo} & "
        "\\textcolor{gray}{\\textbf{G2.5}} \\\\\n"
        "\\midrule\n"
    )

    body_lines = []
    any_cells_written = False
    for ds_idx, (ds_key, ds_label) in enumerate(DATASETS):
        # Gather rows for this dataset
        ds_rows = []
        for exp_key, cfg_label in CONFIGS:
            exp_dir = results_root / ds_key / exp_key
            if not exp_dir.exists():
                continue
            combined = load_combined(exp_dir, judge_id)
            if combined is None:
                continue
            scores = [score_for(combined, mk) for (mk, _, _) in METHODS]
            bi = best_trained_index(scores)

            cells = []
            for i, (mk, _, _) in enumerate(METHODS):
                val = fmt(scores[i])
                if mk in ZERO_SHOT_REF_KEYS:
                    cells.append(f"\\textcolor{{gray}}{{{val}}}")
                elif i == bi:
                    cells.append(f"\\textbf{{{val}}}")
                else:
                    cells.append(val)
            ds_rows.append(f"{cfg_label} & " + " & ".join(cells) + " \\\\")

        if not ds_rows:
            continue

        if any_cells_written:
            body_lines.append("\\midrule")
        body_lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{ds_label} Dataset}}}} \\\\")
        body_lines.extend(ds_rows)
        any_cells_written = True

    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    if not any_cells_written:
        return None
    return header + "\n".join(body_lines) + "\n" + footer


def build_summary_markdown(results_root: Path) -> str:
    """Build a markdown summary grid of overall image scores for quick reference."""
    lines = ["# Overall Image Score — Multi-Judge Summary (0-100 scale)\n"]
    for ds_key, ds_label in DATASETS:
        lines.append(f"\n## {ds_label} ({ds_key})\n")
        lines.append(
            "| Config | Judge | " + " | ".join(s for (_, s, _) in METHODS) + " | Winner (trained) |"
        )
        lines.append(
            "|---|---|" + "---|" * (len(METHODS) + 1)
        )
        for exp_key, cfg_label in CONFIGS:
            exp_dir = results_root / ds_key / exp_key
            if not exp_dir.exists():
                continue
            for j in JUDGES:
                combined = load_combined(exp_dir, j)
                if combined is None:
                    continue
                scores = [score_for(combined, mk) for (mk, _, _) in METHODS]
                bi = best_trained_index(scores)
                winner = METHODS[bi][1] if bi >= 0 else "-"
                vals = " | ".join(fmt(v) for v in scores)
                lines.append(f"| {cfg_label} | {JUDGE_LABELS[j]} | {vals} | **{winner}** |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-configuration tables
    for ds_key, ds_label in DATASETS:
        for exp_key, cfg_label in CONFIGS:
            tex = build_config_table(ds_key, ds_label, exp_key, cfg_label, results_root)
            if tex is None:
                continue
            short_cfg = exp_key.replace("_parallel_cot_", "_").replace("_trajectory", "")
            out_path = out_dir / f"mj_{ds_key}_{short_cfg}.tex"
            out_path.write_text(tex)
            print(f"  wrote {out_path.relative_to(out_dir.parent) if out_dir.parent.exists() else out_path}")

    # Consolidated per-judge tables (ICML-style)
    for judge_id in JUDGES:
        tex = build_consolidated_table(judge_id, results_root)
        if tex is None:
            continue
        out_path = out_dir / f"mj_all_{judge_id}.tex"
        out_path.write_text(tex)
        print(f"  wrote {out_path}")

    # Summary grid
    md = build_summary_markdown(results_root)
    summary_path = out_dir / "multi_judge_summary.md"
    summary_path.write_text(md)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
