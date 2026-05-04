#!/usr/bin/env python3
"""Generate per-dimension LaTeX tables averaged ACROSS the three independent
judges (GPT-5.4, Gemini 3.1, Claude Opus 4.6) for NeurIPS 2026 Appendix G.4.

For each of the 12 (dataset x modality x size) cells we emit two tables:

  avg_img_<ds>_<mod>_<size>.tex   # 9 methods x 7 image-quality dimensions
  avg_act_<ds>_<mod>_<size>.tex   # 9 methods x 11 action/reasoning dimensions

Each cell value is the mean, taken over:
  * the triple-judge agreement subset
      (evaluation_results_final/sample_mappings/<cell>/best.txt),
  * the three judge scores for each sample in that subset.

Raw scores are on a 0-10 scale in the per-sample JSONL and are multiplied by 10
for display to match the 0-100 convention used elsewhere in the paper.

Outputs under `latex/neurips2026/generated_tables/`.

Usage:
    python scripts/evaluation/generate_averaged_judge_tables.py \
        --results-root evaluation_results \
        --final-root   evaluation_results_final \
        --out-dir      latex/neurips2026/generated_tables
"""
import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

JUDGES = ["gpt54", "gemini_31fl", "claude_opus"]
JUDGE_FILES = {
    "gpt54":        "judge_samples_gpt54.jsonl",
    "gemini_31fl":  "judge_samples_gemini_31fl.jsonl",
    "claude_opus":  "judge_samples_claude_opus.jsonl",
}

# (method_key, latex_label, is_zero_shot_reference)
METHODS = [
    ("baseline",      "B",          False),
    ("edit_only",     "E",          False),
    ("standard_text", "S",          False),
    ("rl_text",       r"\rlfull",   False),
    ("rw_text",       r"\rw",       False),
    ("sw_text",       r"\sw",       False),
    ("dpo_text",      r"\dpo",      False),
    ("gpt4o",         "GPT-4o",     True),
    ("gemini25",      "Gemini 2.5", True),
]

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

# (json_key, short_header_for_table)
IMAGE_DIMS = [
    ("overall_image_score",     "Overall"),
    ("instruction_following",   "InstrFol"),
    ("visual_quality",          "VisQual"),
    ("transformation_strength", "TransStr"),
    ("coherence",               "Coher"),
    ("semantic_accuracy",       "SemAcc"),
    ("technical_execution",     "TechEx"),
]

ACTION_DIMS = [
    ("overall_score",              "Overall"),
    ("overall_action_quality",     "ActQual"),
    ("overall_reasoning_quality",  "RsnQual"),
    ("relevance",                  "Relev"),
    ("theme_style_focus",          "Theme"),
    ("completeness",               "Compl"),
    ("efficiency",                 "Effic"),
    ("correctness",                "Correct"),
    ("reasoning_conciseness",      "RsnCon"),
    ("reasoning_completeness",     "RsnCom"),
    ("reasoning_specificity",      "RsnSpc"),
]


def cell_label_for(ds_key: str, exp_key: str) -> str:
    mod = "text" if "text_parallel" in exp_key else "vision"
    size = exp_key.split("_cot_")[1].split("_trajectory")[0]
    return f"{ds_key}_{mod}_{size}"


def load_subset_ids(final_root: Path, cell_label: str) -> Optional[List[str]]:
    p = final_root / "sample_mappings" / cell_label / "best.txt"
    if not p.exists():
        return None
    ids = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return ids or None


def _safe_float(x) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def load_per_sample_scores(
    exp_dir: Path, method_key: str, judge: str, section: str, dim_key: str
) -> Dict[str, float]:
    """Return sample_id -> score*10 for (method, judge, dim)."""
    f = exp_dir / method_key / JUDGE_FILES[judge]
    if not f.exists():
        return {}
    out: Dict[str, float] = {}
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            sid = d.get("sample_id")
            sect = d.get(section)
            if sid is None or not isinstance(sect, dict):
                continue
            v = _safe_float(sect.get(dim_key))
            if v is None:
                continue
            out[sid] = v * 10.0
    return out


def averaged_over_judges_and_subset(
    exp_dir: Path,
    method_key: str,
    section: str,
    dim_key: str,
    subset: Optional[List[str]],
) -> Optional[float]:
    """Mean over all (judge, sample) pairs on the subset (or full pool)."""
    vals: List[float] = []
    for j in JUDGES:
        per_sid = load_per_sample_scores(exp_dir, method_key, j, section, dim_key)
        if subset is None:
            vals.extend(per_sid.values())
        else:
            vals.extend(per_sid[s] for s in subset if s in per_sid)
    return mean(vals) if vals else None


def fmt(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.2f}"


def best_trained_idx_per_column(
    column_values: List[Optional[float]],
) -> int:
    """Index of best trained method (excludes GPT-4o / Gemini 2.5 reference rows)."""
    best_i, best_v = -1, float("-inf")
    for i, (_mk, _lbl, is_ref) in enumerate(METHODS):
        if is_ref:
            continue
        v = column_values[i]
        if v is None:
            continue
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def build_table(
    exp_dir: Path,
    subset: Optional[List[str]],
    dims: List[Tuple[str, str]],
    section: str,
    caption: str,
    label: str,
    *,
    small_font: bool = True,
) -> Optional[str]:
    # scores[dim_key] -> list indexed by METHODS
    scores: Dict[str, List[Optional[float]]] = {}
    for dim_key, _ in dims:
        row = []
        for (mk, _lbl, _is_ref) in METHODS:
            row.append(averaged_over_judges_and_subset(
                exp_dir, mk, section, dim_key, subset
            ))
        scores[dim_key] = row

    # drop table entirely if every cell is empty
    if all(all(v is None for v in scores[dk]) for dk, _ in dims):
        return None

    best_idx = {
        dk: best_trained_idx_per_column(scores[dk]) for dk, _ in dims
    }

    col_spec = "l" + "c" * len(dims)
    hdr = "Method & " + " & ".join(h for _, h in dims) + " \\\\"

    lines = ["\\begin{table}[h]"]
    lines.append("\\centering")
    if small_font:
        lines.append("\\scriptsize")
        lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(hdr)
    lines.append("\\midrule")

    midrule_placed = False
    for i, (mk, latex_label, is_ref) in enumerate(METHODS):
        if is_ref and not midrule_placed:
            lines.append("\\midrule")
            midrule_placed = True
        cells = []
        is_best_anywhere = False
        for dk, _ in dims:
            v = fmt(scores[dk][i])
            if is_ref:
                cells.append(f"\\textcolor{{gray}}{{{v}}}")
            elif best_idx[dk] == i:
                cells.append(f"\\textbf{{{v}}}")
                is_best_anywhere = True
            else:
                cells.append(v)
        if is_ref:
            method_cell = f"\\textcolor{{gray}}{{{latex_label}}}"
        elif is_best_anywhere:
            method_cell = f"\\textbf{{{latex_label}}}"
        else:
            method_cell = latex_label
        lines.append(method_cell + " & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--final-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    final_root = Path(args.final_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ds_key, ds_label in DATASETS:
        for exp_key, cfg_label in CONFIGS:
            exp_dir = results_root / ds_key / exp_key
            if not exp_dir.exists():
                continue
            cell_label = cell_label_for(ds_key, exp_key)
            subset = load_subset_ids(final_root, cell_label)
            subset_note = (
                "triple-judge agreement subset" if subset else "full test pool"
            )

            mod = "text" if "text_parallel" in exp_key else "vision"
            size = exp_key.split("_cot_")[1].split("_trajectory")[0]
            short = f"{ds_key}_{mod}_{size}"

            # Image quality table
            img_tex = build_table(
                exp_dir,
                subset,
                IMAGE_DIMS,
                section="image_scores",
                caption=(
                    f"\\textbf{{{ds_label} {cfg_label} -- Image quality per dimension}} "
                    f"(0--100), averaged over the three judges "
                    f"(GPT-5.4, Gemini 3.1, Claude Opus 4.6) on the {subset_note}. "
                    f"Bold = best trained method (B/E/S/R/RW/SW/D) per column; "
                    f"GPT-4o and Gemini 2.5 (grey) are zero-shot reference planners."
                ),
                label=f"tab:avg_img_{short}",
            )
            if img_tex is not None:
                p = out_dir / f"avg_img_{short}.tex"
                p.write_text(img_tex)
                print(f"  wrote {p}")
                written += 1

            # Action / reasoning table
            act_tex = build_table(
                exp_dir,
                subset,
                ACTION_DIMS,
                section="action_scores",
                caption=(
                    f"\\textbf{{{ds_label} {cfg_label} -- Action and reasoning "
                    f"quality per dimension}} (0--100), averaged over the three "
                    f"judges (GPT-5.4, Gemini 3.1, Claude Opus 4.6) on the "
                    f"{subset_note}. Bold = best trained method (B/S/R/RW/SW/D) "
                    f"per column; GPT-4o and Gemini 2.5 (grey) are zero-shot "
                    f"reference planners. Edit-Only (E) emits no action sequence "
                    f"and is left blank."
                ),
                label=f"tab:avg_act_{short}",
            )
            if act_tex is not None:
                p = out_dir / f"avg_act_{short}.tex"
                p.write_text(act_tex)
                print(f"  wrote {p}")
                written += 1

    print(f"\nWrote {written} tables to {out_dir}")


if __name__ == "__main__":
    main()
