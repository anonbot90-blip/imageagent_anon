#!/usr/bin/env python3
"""Generate TRIPLE-judge LaTeX tables (GPT-5.4, Gemini 3.1, Claude Opus 4.6)
for NeurIPS 2026. Per-cell tables are aggregated OVER THE TRIPLE-JUDGE SUBSET
from `evaluation_results_final/sample_mappings/<cell>/best.txt` when feasible;
cells with no feasible subset fall back to the full pool.

Reads per-sample JSONL (`judge_samples_<judge>.jsonl`) so that aggregation is
bound to the subset ID list. Score scaling mirrors the dual-judge generator:
raw `image_scores.overall_image_score` on a 0-10 scale is multiplied by 10.

Outputs under `latex/neurips2026/generated_tables/`:
  mj_<dataset>_<mod>_<size>.tex   # 12 per-cell 3-judge tables
  mj_all_<judge>.tex              # 3 consolidated-per-judge tables (full pool)
  multi_judge_summary.md          # flat markdown grid

No pool / sample-size / subset-size text appears in captions (per request).

Usage:
    python scripts/evaluation/generate_triple_judge_tables.py \
        --results-root evaluation_results \
        --final-root   evaluation_results_final \
        --out-dir      latex/neurips2026/generated_tables
"""
import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

JUDGES = ["gpt54", "gemini_31fl", "claude_opus"]
JUDGE_LABELS = {
    "gpt54":        "GPT-5.4",
    "gemini_31fl":  "Gemini 3.1",
    "claude_opus":  "Claude Opus 4.6",
}
JUDGE_FILES = {
    "gpt54":        "judge_samples_gpt54.jsonl",
    "gemini_31fl":  "judge_samples_gemini_31fl.jsonl",
    "claude_opus":  "judge_samples_claude_opus.jsonl",
}

# method_key -> (short_label, latex_macro_or_plain)
METHODS = [
    ("baseline",      "B",     "B"),
    ("edit_only",     "E",     "E"),
    ("standard_text", "S",     "S"),
    ("rl_text",       "R",     r"\rlfull"),
    ("rw_text",       "RW",    r"\rw"),
    ("sw_text",       "SW",    r"\sw"),
    ("dpo_text",      "D",     r"\dpo"),
    ("gpt4o",         "GPT4o", "GPT-4o"),          # zero-shot reference planner
    ("gemini25",      "G2.5",  "Gemini 2.5"),  # zero-shot reference planner
]

TRAINED_METHOD_KEYS = {"baseline", "edit_only", "standard_text", "rl_text",
                       "rw_text", "sw_text", "dpo_text"}
ZERO_SHOT_REF_KEYS = {"gpt4o", "gemini25"}

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


def cell_label_for(ds_key: str, exp_key: str) -> str:
    mod = "text" if "text_parallel" in exp_key else "vision"
    size = exp_key.split("_cot_")[1].split("_trajectory")[0]  # "4b" or "8b"
    return f"{ds_key}_{mod}_{size}"


def load_subset_ids(final_root: Path, cell_label: str) -> Optional[List[str]]:
    p = final_root / "sample_mappings" / cell_label / "best.txt"
    if not p.exists():
        return None
    ids = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return ids or None


def load_samples(exp_dir: Path, method_key: str, judge: str) -> Dict[str, float]:
    """sample_id -> overall_image_score × 10 for one (method, judge)."""
    f = exp_dir / method_key / JUDGE_FILES[judge]
    if not f.exists():
        return {}
    out = {}
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            sid = d.get("sample_id")
            s = d.get("image_scores", {}).get("overall_image_score")
            if sid is None or not isinstance(s, (int, float)):
                continue
            out[sid] = float(s) * 10.0
    return out


def agg_mean(scores_by_sid: Dict[str, float], subset: Optional[List[str]]) -> Optional[float]:
    if subset is None:
        vals = list(scores_by_sid.values())
    else:
        vals = [scores_by_sid[s] for s in subset if s in scores_by_sid]
    if not vals:
        return None
    return mean(vals)


def fmt(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.2f}"


def best_trained_index(scores: List[Optional[float]]) -> int:
    best_i, best_v = -1, float("-inf")
    for i, (mk, _, _) in enumerate(METHODS):
        if mk not in TRAINED_METHOD_KEYS:
            continue
        v = scores[i]
        if v is None:
            continue
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def build_config_table(
    ds_key: str,
    ds_label: str,
    exp_key: str,
    cfg_label: str,
    results_root: Path,
    final_root: Path,
) -> Optional[str]:
    exp_dir = results_root / ds_key / exp_key
    if not exp_dir.exists():
        return None

    cell_label = cell_label_for(ds_key, exp_key)
    subset = load_subset_ids(final_root, cell_label)  # None => full pool fallback
    subset_note = "triple-judge agreement subset" if subset else "full test pool"

    scores: Dict[str, List[Optional[float]]] = {j: [] for j in JUDGES}
    for (mk, _, _) in METHODS:
        for j in JUDGES:
            s_by_sid = load_samples(exp_dir, mk, j)
            scores[j].append(agg_mean(s_by_sid, subset))

    if all(all(v is None for v in scores[j]) for j in JUDGES):
        return None

    best_idx = {j: best_trained_index(scores[j]) for j in JUDGES}

    mod = "text" if "text_parallel" in exp_key else "vision"
    size = exp_key.split("_cot_")[1].split("_trajectory")[0]
    label_key = f"tab:mj_{ds_key}_{mod}_{size}"

    caption = (
        f"\\textbf{{{ds_label} {cfg_label}}}: "
        f"Image quality (0--100) under three independent judges "
        f"(GPT-5.4, Gemini 3.1, Claude Opus 4.6), evaluated on the {subset_note}. "
        f"Bold = best trained method (B/E/S/R/RW/SW/D) per judge; "
        f"Gemini 2.5 (grey) is a zero-shot reference planner."
    )

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

    body = []
    midrule_placed = False
    for i, (mk, _, latex_label) in enumerate(METHODS):
        if mk in ZERO_SHOT_REF_KEYS:
            if not midrule_placed:
                body.append("\\midrule")
                midrule_placed = True
            grey = lambda x: f"\\textcolor{{gray}}{{{x}}}"
            row = f"{grey(latex_label)} & " + " & ".join(
                grey(fmt(scores[j][i])) for j in JUDGES
            ) + " \\\\"
            body.append(row)
            continue

        cells = []
        for j in JUDGES:
            v = fmt(scores[j][i])
            cells.append(f"\\textbf{{{v}}}" if best_idx[j] == i else v)
        label_cell = f"\\textbf{{{latex_label}}}" if any(best_idx[j] == i for j in JUDGES) else latex_label
        body.append(f"{label_cell} & " + " & ".join(cells) + " \\\\")

    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(body) + "\n" + footer


def build_consolidated_table(judge_id: str, results_root: Path) -> Optional[str]:
    """Consolidated table for one judge over the FULL POOL across all 12 cells."""
    judge_label = JUDGE_LABELS[judge_id]
    header = (
        "\\begin{table}[h]\n"
        "\\centering\n\\small\n"
        f"\\caption{{\\textbf{{Overall Image Quality under {judge_label} judge "
        f"across all 12 configurations.}} Bold = best trained method (B/E/S/R/RW/SW/D) per row; "
        f"Gemini 2.5 (grey) is a zero-shot reference planner.}}\n"
        f"\\label{{tab:mj_all_{judge_id}}}\n"
        "\\begin{tabular}{lccccccc|cc}\n"
        "\\toprule\n"
        "\\textbf{Configuration} & \\textbf{B} & \\textbf{E} & \\textbf{S} & "
        "\\textbf{\\rlfull} & \\textbf{\\rw} & \\textbf{\\sw} & \\textbf{\\dpo} & "
        "\\textcolor{gray}{\\textbf{GPT-4o}} & \\textcolor{gray}{\\textbf{G2.5}} \\\\\n"
        "\\midrule\n"
    )

    body = []
    any_written = False
    for ds_key, ds_label in DATASETS:
        ds_rows = []
        for exp_key, cfg_label in CONFIGS:
            exp_dir = results_root / ds_key / exp_key
            if not exp_dir.exists():
                continue
            scores = []
            have_any = False
            for (mk, _, _) in METHODS:
                s_by_sid = load_samples(exp_dir, mk, judge_id)
                v = agg_mean(s_by_sid, None)
                if v is not None:
                    have_any = True
                scores.append(v)
            if not have_any:
                continue
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
        if any_written:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{ds_label} Dataset}}}} \\\\")
        body.extend(ds_rows)
        any_written = True

    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    if not any_written:
        return None
    return header + "\n".join(body) + "\n" + footer


def build_summary_markdown(results_root: Path, final_root: Path) -> str:
    lines = ["# Overall Image Score — Triple-Judge Summary (0-100 scale)\n",
             "Cells whose triple-judge subset is feasible use that subset; the 3 infeasible simple cells fall back to the full pool.\n"]
    for ds_key, ds_label in DATASETS:
        lines.append(f"\n## {ds_label} ({ds_key})\n")
        lines.append("| Config | Judge | Subset | " + " | ".join(s for (_, s, _) in METHODS) + " | Winner (trained) |")
        lines.append("|---|---|---|" + "---|" * (len(METHODS) + 1))
        for exp_key, cfg_label in CONFIGS:
            exp_dir = results_root / ds_key / exp_key
            if not exp_dir.exists():
                continue
            cell_label = cell_label_for(ds_key, exp_key)
            subset = load_subset_ids(final_root, cell_label)
            subset_tag = "triple-agree" if subset else "full-pool"
            for j in JUDGES:
                scores = []
                have_any = False
                for (mk, _, _) in METHODS:
                    s_by_sid = load_samples(exp_dir, mk, j)
                    v = agg_mean(s_by_sid, subset)
                    if v is not None:
                        have_any = True
                    scores.append(v)
                if not have_any:
                    continue
                bi = best_trained_index(scores)
                winner = METHODS[bi][1] if bi >= 0 else "-"
                vals = " | ".join(fmt(v) for v in scores)
                lines.append(f"| {cfg_label} | {JUDGE_LABELS[j]} | {subset_tag} | {vals} | **{winner}** |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--final-root", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    results_root = Path(args.results_root)
    final_root = Path(args.final_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_key, ds_label in DATASETS:
        for exp_key, cfg_label in CONFIGS:
            tex = build_config_table(ds_key, ds_label, exp_key, cfg_label, results_root, final_root)
            if tex is None:
                continue
            short = exp_key.replace("_parallel_cot_", "_").replace("_trajectory", "")
            out_path = out_dir / f"mj_{ds_key}_{short}.tex"
            out_path.write_text(tex)
            print(f"  wrote {out_path}")

    for j in JUDGES:
        tex = build_consolidated_table(j, results_root)
        if tex is None:
            continue
        out_path = out_dir / f"mj_all_{j}.tex"
        out_path.write_text(tex)
        print(f"  wrote {out_path}")

    md = build_summary_markdown(results_root, final_root)
    (out_dir / "multi_judge_summary.md").write_text(md)
    print(f"\nSummary: {out_dir / 'multi_judge_summary.md'}")


if __name__ == "__main__":
    main()
