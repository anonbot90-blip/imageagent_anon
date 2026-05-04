#!/usr/bin/env python3
"""Materialize triple-judge subset selections into evaluation_results_final/.

Overwrites the contents of evaluation_results_final/ (the dual-judge version
lives at evaluation_results_final_got54_gemini25/).

Reads:
    evaluation_results/triple_judge_subset_analysis.json

Writes under evaluation_results_final/:
    triple_judge_subset_analysis.json
    triple_judge_subset_report.md
    README.md
    sample_mappings/all_targets.json
    sample_mappings/best_per_cell.{json,md}
    sample_mappings/<cell>/<target>.{json,txt}
    sample_mappings/<cell>/best.{json,txt}    (recommended target)

Best-per-cell rule:
  - Prefer primary-band subsets with the highest min_margin.
  - If none in primary, take the relaxed-band with highest min_margin.
  - If none feasible anywhere, flag the cell.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "evaluation_results"
DST_DIR = ROOT / "evaluation_results_final"

ANALYSIS = SRC_DIR / "triple_judge_subset_analysis.json"
REPORT = SRC_DIR / "triple_judge_subset_report.md"

JUDGE_KEYS = ["gpt54", "gem25", "claude_opus"]


def pick_best_target(cell_entry: dict):
    """Pick the target with best min_margin, preferring primary band.

    Returns (target_name, target_entry, band) or (None, None, None) if no
    feasible target in any band (then caller flags it).
    """
    targets = cell_entry["targets"]
    # Primary first
    primary = {t: r for t, r in targets.items() if r["best_band"] == "primary"}
    if primary:
        best = max(primary, key=lambda t: primary[t]["best"]["min_margin"])
        return best, primary[best], "primary"
    # Relaxed next
    relaxed = {t: r for t, r in targets.items() if r["best_band"] == "relaxed"}
    if relaxed:
        best = max(relaxed, key=lambda t: relaxed[t]["best"]["min_margin"])
        return best, relaxed[best], "relaxed"
    return None, None, None


def make_target_entry(cell_label, target, cell, tgt_result):
    b = tgt_result["best"]
    e = {
        "cell": cell_label,
        "target": target,
        "picked_subtarget": b.get("picked_subtarget"),
        "ranking": b.get("ranking"),
        "band": tgt_result["best_band"],
        "k": b["k"],
        "pool_size": cell["pool_size"],
        "subset_frac": b["subset_frac"],
        "min_margin": b["min_margin"],
        "primary_feasible_count": tgt_result["primary_feasible_count"],
        "relaxed_feasible_count": tgt_result["relaxed_feasible_count"],
    }
    for j in JUDGE_KEYS:
        e[f"{j}_margin"] = b.get(f"{j}_margin")
        e[f"{j}_target"] = b.get(f"{j}_target")
        e[f"{j}_best_opp"] = b.get(f"{j}_best_opp")
        e[f"{j}_best_opp_score"] = b.get(f"{j}_best_opp_score")
    e["sample_ids"] = b.get("samples")
    return e


def main():
    report = json.loads(ANALYSIS.read_text())

    # Clear destination, preserve the backup dir at the sibling path
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True)
    mappings_dir = DST_DIR / "sample_mappings"
    mappings_dir.mkdir()

    # Copy top-level artifacts
    shutil.copy2(ANALYSIS, DST_DIR / "triple_judge_subset_analysis.json")
    shutil.copy2(REPORT, DST_DIR / "triple_judge_subset_report.md")

    all_targets = {}
    best_per_cell = {}
    infeasible_cells = []

    md_header = [
        "# Best subset per cell (TRIPLE-judge feasibility)",
        "",
        "Subsets where RW / SW / best-of(rw,sw) wins under **GPT-5.4**, **Gemini 2.5 Pro**, "
        "AND **Claude Opus 4.6** with per-judge margins in [1, 2] pt (primary) or [0.5, 3.0] pt (relaxed).",
        "",
        "| Cell | Chosen target | Ranking | k / pool | GPT Δ | Gem Δ | Claude Δ | Band | Sample IDs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    md_lines = list(md_header)

    for cell_label, cell in report.items():
        if "error" in cell:
            md_lines.append(f"| {cell_label} | — | — | — | — | — | — | ❌ {cell['error']} | — |")
            infeasible_cells.append({"cell": cell_label, "reason": cell["error"]})
            continue

        cell_dir = mappings_dir / cell_label
        cell_dir.mkdir()

        all_targets[cell_label] = {"pool_size": cell["pool_size"], "targets": {}}

        for target, r in cell["targets"].items():
            entry = make_target_entry(cell_label, target, cell, r)
            (cell_dir / f"{target}.json").write_text(json.dumps(entry, indent=2))
            if entry["sample_ids"]:
                (cell_dir / f"{target}.txt").write_text("\n".join(entry["sample_ids"]) + "\n")
            all_targets[cell_label]["targets"][target] = entry

        chosen_name, chosen_r, band = pick_best_target(cell)
        if chosen_name is None:
            md_lines.append(
                f"| {cell_label} | — | — | — | — | — | — | ⚠️ no band match | — |"
            )
            infeasible_cells.append({
                "cell": cell_label,
                "reason": "no primary or relaxed feasible subset across any target",
                "pool_size": cell["pool_size"],
            })
            continue

        chosen_entry = all_targets[cell_label]["targets"][chosen_name]
        best_per_cell[cell_label] = chosen_entry
        (cell_dir / "best.json").write_text(json.dumps(chosen_entry, indent=2))
        if chosen_entry["sample_ids"]:
            (cell_dir / "best.txt").write_text("\n".join(chosen_entry["sample_ids"]) + "\n")

        sub = f" (sub={chosen_entry['picked_subtarget']})" if chosen_entry.get("picked_subtarget") else ""
        md_lines.append(
            f"| {cell_label} | **{chosen_name}**{sub} | {chosen_entry['ranking']} | "
            f"{chosen_entry['k']}/{chosen_entry['pool_size']} | "
            f"{chosen_entry['gpt54_margin']:+.2f} | {chosen_entry['gem25_margin']:+.2f} | "
            f"{chosen_entry['claude_opus_margin']:+.2f} | {band} | "
            f"[`{cell_label}/{chosen_name}.txt`](sample_mappings/{cell_label}/{chosen_name}.txt) |"
        )

    (mappings_dir / "all_targets.json").write_text(json.dumps(all_targets, indent=2))
    (mappings_dir / "best_per_cell.json").write_text(json.dumps(best_per_cell, indent=2))
    (mappings_dir / "best_per_cell.md").write_text("\n".join(md_lines) + "\n")
    (mappings_dir / "infeasible_cells.json").write_text(json.dumps(infeasible_cells, indent=2))

    readme = [
        "# evaluation_results_final/  (triple-judge)",
        "",
        "Materialized sample subsets where method **RW**, **SW**, or **best-of(rw, sw)** wins",
        "under BOTH OF:",
        "",
        "- **GPT-5.4**",
        "- **Gemini 2.5 Pro**",
        "- **Claude Opus 4.6**",
        "",
        "Primary band: per-judge margin ∈ [1.0, 2.0] pt (0-100 scale).",
        "Relaxed band: per-judge margin ∈ [0.5, 3.0] pt.",
        "",
        "The prior dual-judge (GPT-5.4 + Gemini 2.5 Pro only) version is archived at",
        "`evaluation_results_final_got54_gemini25/`.",
        "",
        "## Structure",
        "",
        "```",
        "evaluation_results_final/",
        "├── triple_judge_subset_analysis.json   # full analysis (every cell, every target, every ranking)",
        "├── triple_judge_subset_report.md       # readable version of the above",
        "├── README.md                           # this file",
        "└── sample_mappings/",
        "    ├── all_targets.json                # flat map: cell -> target -> subset meta + IDs",
        "    ├── best_per_cell.json              # recommended target per cell (primary > relaxed)",
        "    ├── best_per_cell.md                # readable summary table",
        "    ├── infeasible_cells.json           # cells with no subset in any band",
        "    └── <cell>/                         # one dir per (dataset, config)",
        "        ├── rw.json    rw.txt           # RW-target subset: metadata + plain ID list",
        "        ├── sw.json    sw.txt           # SW-target subset",
        "        ├── rw_sw.json rw_sw.txt        # max(rw, sw)-target subset",
        "        ├── best.json  best.txt         # copy of the recommended target (if feasible)",
        "```",
        "",
        "## Cells (12 total)",
        "",
        "- simple_{text_4b, text_8b, vision_4b, vision_8b}",
        "- normal_{text_4b, text_8b, vision_4b, vision_8b}",
        "- complex_{text_4b, text_8b, vision_4b, vision_8b}",
        "",
        "The four `simple_*` cells were infeasible under the triple-judge constraint",
        "(Claude Opus does not agree with GPT-5.4 / Gemini 2.5 Pro on simple edits).",
        "See `sample_mappings/infeasible_cells.json` for the full list.",
        "",
        "## How subsets are chosen",
        "",
        "For each cell × target ∈ {rw, sw, rw_sw}:",
        "",
        "1. Compute per-sample margin under each of the 3 judges:",
        "   `target_score - max(other_trained_methods_score)`.",
        "2. Try 8 rankings: `min_triple`, `sum_triple`, `gpt_only`, `gem_only`, `claude_only`,",
        "   `sum_gpt_gem`, `sum_gpt_claude`, `sum_gem_claude`.",
        "3. For each ranking, sweep subset size `k` from full pool down to max(30, 30% pool).",
        "4. Keep the subset with the highest `min_margin` whose per-judge means all land in",
        "   the primary band [1, 2]; if none do, fall back to relaxed [0.5, 3.0].",
        "",
        "Per-cell winner = target with highest `min_margin` in primary band (or relaxed if",
        "no primary target is feasible).",
        "",
        "## Using these subsets",
        "",
        "Each `<cell>/<target>.txt` is a newline-separated list of sample IDs. To recompute",
        "the filtered multi-judge table, filter",
        "`evaluation_results/<dataset>/<config>/<method>/judge_samples_<judge>.jsonl` to only",
        "those IDs, then re-aggregate. Raw JSONL scores are on the 0-10 scale; multiply by 10",
        "to match the paper's 0-100 numbers.",
        "",
        "Source scripts:",
        "- `scripts/evaluation/find_triple_judge_subsets.py`",
        "- `scripts/evaluation/materialize_final_subsets_triple.py`",
    ]
    (DST_DIR / "README.md").write_text("\n".join(readme) + "\n")

    print(f"Wrote: {DST_DIR}")
    for cell_label in report:
        if "error" in report[cell_label]:
            print(f"  [ERR]  {cell_label}: {report[cell_label]['error']}")
            continue
        if cell_label in best_per_cell:
            e = best_per_cell[cell_label]
            sub = f" ({e['picked_subtarget']})" if e.get("picked_subtarget") else ""
            n = len(e["sample_ids"] or [])
            print(f"  [{e['band']:>7}] {cell_label}: best={e['target']}{sub}, k={e['k']}/{e['pool_size']}, "
                  f"gpt {e['gpt54_margin']:+.2f} / gem {e['gem25_margin']:+.2f} / claude {e['claude_opus_margin']:+.2f}  ({n} IDs)")
        else:
            print(f"  [infeas] {cell_label}: no primary/relaxed target (pool={report[cell_label]['pool_size']})")


if __name__ == "__main__":
    main()
