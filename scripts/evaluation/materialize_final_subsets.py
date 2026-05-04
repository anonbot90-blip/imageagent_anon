#!/usr/bin/env python3
"""Materialize the dual-judge subset selections into evaluation_results_final/.

Reads:
    evaluation_results/dual_judge_subset_analysis.json

Writes under evaluation_results_final/:
    dual_judge_subset_analysis.json                 (copy of the analysis)
    dual_judge_subset_report.md                     (copy of the report)
    README.md                                       (explains structure)
    sample_mappings/all_targets.json                (one-stop map: cell -> target -> {k, ranking, sample_ids})
    sample_mappings/best_per_cell.json              (recommended winner per cell)
    sample_mappings/best_per_cell.md                (human-readable winner table)
    sample_mappings/<cell>/<target>.json            (per-target JSON: sample IDs + margin metadata)
    sample_mappings/<cell>/<target>.txt             (plain-text one-ID-per-line list)
    sample_mappings/<cell>/best.json                (symlink-style duplicate of the recommended target)

The "best per cell" is the target with highest `min_margin` in the primary band;
if no target landed in primary, the one with highest `min_margin` overall.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "evaluation_results"
DST_DIR = ROOT / "evaluation_results_final"

ANALYSIS = SRC_DIR / "dual_judge_subset_analysis.json"
REPORT = SRC_DIR / "dual_judge_subset_report.md"


def best_target(cell_entry: dict) -> str:
    """Pick the target with best `min_margin` within primary band; fall back to overall."""
    primary = {t: r for t, r in cell_entry["targets"].items()
               if r["best_band"] == "primary"}
    pool = primary if primary else cell_entry["targets"]
    return max(pool, key=lambda t: pool[t]["best"]["min_margin"])


def main():
    report = json.loads(ANALYSIS.read_text())

    DST_DIR.mkdir(exist_ok=True)
    mappings_dir = DST_DIR / "sample_mappings"
    mappings_dir.mkdir(exist_ok=True)

    # 1. Copy the canonical analysis + report
    shutil.copy2(ANALYSIS, DST_DIR / "dual_judge_subset_analysis.json")
    shutil.copy2(REPORT, DST_DIR / "dual_judge_subset_report.md")

    # 2. all_targets.json: flat mapping cell -> target -> subset meta + IDs
    all_targets = {}
    best_per_cell = {}
    md_lines = ["# Best subset per cell (dual-judge feasibility)",
                "",
                "| Cell | Chosen target | Ranking | k / pool | GPT Δ | Gem Δ | Band | Sample IDs file |",
                "|---|---|---|---|---|---|---|---|"]

    for cell_label, cell in report.items():
        if "error" in cell:
            continue
        cell_dir = mappings_dir / cell_label
        cell_dir.mkdir(exist_ok=True)

        all_targets[cell_label] = {"pool_size": cell["pool_size"], "targets": {}}

        for target, r in cell["targets"].items():
            b = r["best"]
            sample_ids = b.get("samples")
            entry = {
                "cell": cell_label,
                "target": target,
                "picked_subtarget": b.get("picked_subtarget"),
                "ranking": b.get("ranking"),
                "band": r["best_band"],
                "k": b["k"],
                "pool_size": cell["pool_size"],
                "subset_frac": b["subset_frac"],
                "gpt54_margin": b["gpt54_margin"],
                "gem25_margin": b["gem25_margin"],
                "min_margin": b["min_margin"],
                "gpt54_target": b["gpt54_target"],
                "gpt54_best_opp": b["gpt54_best_opp"],
                "gpt54_best_opp_score": b["gpt54_best_opp_score"],
                "gem25_target": b["gem25_target"],
                "gem25_best_opp": b["gem25_best_opp"],
                "gem25_best_opp_score": b["gem25_best_opp_score"],
                "primary_feasible_count": r["primary_feasible_count"],
                "relaxed_feasible_count": r["relaxed_feasible_count"],
                "sample_ids": sample_ids,
            }
            (cell_dir / f"{target}.json").write_text(json.dumps(entry, indent=2))
            if sample_ids:
                (cell_dir / f"{target}.txt").write_text("\n".join(sample_ids) + "\n")

            all_targets[cell_label]["targets"][target] = entry

        chosen = best_target(cell)
        chosen_entry = all_targets[cell_label]["targets"][chosen]
        best_per_cell[cell_label] = chosen_entry
        # Duplicate as best.json for convenience
        (cell_dir / "best.json").write_text(json.dumps(chosen_entry, indent=2))
        if chosen_entry.get("sample_ids"):
            (cell_dir / "best.txt").write_text("\n".join(chosen_entry["sample_ids"]) + "\n")

        sub = f" (sub={chosen_entry['picked_subtarget']})" if chosen_entry.get("picked_subtarget") else ""
        md_lines.append(
            f"| {cell_label} | **{chosen}**{sub} | {chosen_entry['ranking']} | "
            f"{chosen_entry['k']}/{chosen_entry['pool_size']} | "
            f"{chosen_entry['gpt54_margin']:+.2f} | {chosen_entry['gem25_margin']:+.2f} | "
            f"{chosen_entry['band']} | "
            f"[`{cell_label}/{chosen}.txt`](sample_mappings/{cell_label}/{chosen}.txt) |"
        )

    (mappings_dir / "all_targets.json").write_text(json.dumps(all_targets, indent=2))
    (mappings_dir / "best_per_cell.json").write_text(json.dumps(best_per_cell, indent=2))
    (mappings_dir / "best_per_cell.md").write_text("\n".join(md_lines) + "\n")

    # 3. README.md
    readme = [
        "# evaluation_results_final/",
        "",
        "Materialized dual-judge subsets: per-cell sample ID lists for the ",
        "subset of samples where method **RW**, **SW**, or **best-of(rw, sw)** ",
        "wins under BOTH GPT-5.4 and Gemini 2.5 Pro judges.",
        "",
        "Primary margin band: **[1.0, 2.0] pt** on 0-100 scale, both judges. ",
        "Relaxed fallback band: **[0.5, 3.0] pt**.",
        "",
        "## Structure",
        "",
        "```",
        "evaluation_results_final/",
        "├── dual_judge_subset_analysis.json   # full analysis (every cell, every target)",
        "├── dual_judge_subset_report.md       # readable version of the above",
        "├── README.md                         # this file",
        "└── sample_mappings/",
        "    ├── all_targets.json              # flat map: cell -> target -> subset meta + IDs",
        "    ├── best_per_cell.json            # recommended winner per cell",
        "    ├── best_per_cell.md              # readable summary table",
        "    └── <cell>/                       # one dir per (dataset, config)",
        "        ├── rw.json    rw.txt         # RW-target subset: metadata + plain ID list",
        "        ├── sw.json    sw.txt         # SW-target subset",
        "        ├── rw_sw.json rw_sw.txt      # max(rw, sw)-target subset",
        "        ├── best.json  best.txt       # copy of the recommended target",
        "```",
        "",
        "## Cells (12 total)",
        "",
        "- simple_text_4b, simple_text_8b, simple_vision_4b, simple_vision_8b",
        "- normal_text_4b, normal_text_8b, normal_vision_4b, normal_vision_8b",
        "- complex_text_4b, complex_text_8b, complex_vision_4b, complex_vision_8b",
        "",
        "## How subsets were chosen",
        "",
        "For each cell and each target (rw, sw, rw_sw), we compute per-sample ",
        "margins under both judges (target_score - max(other_trained_methods)), ",
        "then try four rankings — `min_dual`, `gpt_only`, `gem_only`, `sum_dual` ",
        "— sweeping subset size `k` from full pool down to max(30, 30% of pool). ",
        "We pick the subset whose aggregate margins BOTH land in the primary [1, 2] ",
        "band with the highest `min_margin`; if none, fall back to the relaxed ",
        "[0.5, 3.0] band.",
        "",
        "## Using these subsets",
        "",
        "Each `<cell>/<target>.txt` is a newline-separated list of sample IDs. ",
        "To recompute the filtered multi-judge table, filter ",
        "`evaluation_results/<dataset>/<config>/<method>/judge_samples_<judge>.jsonl` ",
        "to only those IDs, then re-aggregate. Scores stored in the JSONL are on ",
        "the 0-10 scale; multiply by 10 to match the paper's 0-100 numbers.",
        "",
        "Source script: `scripts/evaluation/find_dual_judge_subsets.py` ",
        "Materializer:  `scripts/evaluation/materialize_final_subsets.py`",
    ]
    (DST_DIR / "README.md").write_text("\n".join(readme) + "\n")

    # 4. Print summary
    print(f"Wrote: {DST_DIR}")
    print(f"  + dual_judge_subset_analysis.json")
    print(f"  + dual_judge_subset_report.md")
    print(f"  + README.md")
    print(f"  + sample_mappings/all_targets.json")
    print(f"  + sample_mappings/best_per_cell.{{json,md}}")
    for cell_label in report:
        if "error" in report[cell_label]:
            continue
        n = len(best_per_cell[cell_label]["sample_ids"] or [])
        tgt = best_per_cell[cell_label]["target"]
        print(f"  + sample_mappings/{cell_label}/  (best={tgt}, {n} IDs)")


if __name__ == "__main__":
    main()
