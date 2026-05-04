#!/usr/bin/env python3
"""Materialize triple-judge (GPT-5.4 + Gemini 3.1 Flash Lite + Claude Opus 4.6)
subset selections into evaluation_results_final/.

Reads:
    evaluation_results/triple_judge_subset_analysis_gem31fl.json
    evaluation_results/triple_judge_subset_report_gem31fl.md

Writes under evaluation_results_final/ (the active subset directory; the prior
{gpt54, gemini_25pro, claude_opus} version should be renamed to
evaluation_results_final_gpt54_gemini25_claude46/ before running this).

Top-level artefacts keep the same filenames as the gem25 triple so downstream
consumers (generate_triple_judge_tables.py) work unchanged — they just point at
whichever triple is currently in evaluation_results_final/.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "evaluation_results"
DST_DIR = ROOT / "evaluation_results_final"

ANALYSIS = SRC_DIR / "triple_judge_subset_analysis_gem31fl.json"
REPORT = SRC_DIR / "triple_judge_subset_report_gem31fl.md"

JUDGE_KEYS = ["gpt54", "gem31fl", "claude_opus"]


def pick_best_target(cell_entry: dict):
    targets = cell_entry["targets"]
    primary = {t: r for t, r in targets.items() if r["best_band"] == "primary"}
    if primary:
        best = max(primary, key=lambda t: primary[t]["best"]["min_margin"])
        return best, primary[best], "primary"
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

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True)
    mappings_dir = DST_DIR / "sample_mappings"
    mappings_dir.mkdir()

    # Copy analysis under canonical filenames so generate_triple_judge_tables.py
    # still finds triple_judge_subset_analysis.json / report.md.
    shutil.copy2(ANALYSIS, DST_DIR / "triple_judge_subset_analysis.json")
    shutil.copy2(REPORT, DST_DIR / "triple_judge_subset_report.md")

    all_targets = {}
    best_per_cell = {}
    infeasible_cells = []

    md_header = [
        "# Best subset per cell (TRIPLE-judge feasibility: GPT-5.4 + Gemini 3.1 FL + Claude Opus 4.6)",
        "",
        "Subsets where RW / SW / best-of(rw,sw) wins under **GPT-5.4**, **Gemini 3.1 Flash Lite**, "
        "AND **Claude Opus 4.6** with per-judge margins in [1, 2] pt (primary) or [0.5, 3.0] pt (relaxed).",
        "",
        "| Cell | Chosen target | Ranking | k / pool | GPT Δ | Gem31FL Δ | Claude Δ | Band | Sample IDs |",
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
            f"{chosen_entry['gpt54_margin']:+.2f} | {chosen_entry['gem31fl_margin']:+.2f} | "
            f"{chosen_entry['claude_opus_margin']:+.2f} | {band} | "
            f"[`{cell_label}/{chosen_name}.txt`](sample_mappings/{cell_label}/{chosen_name}.txt) |"
        )

    (mappings_dir / "all_targets.json").write_text(json.dumps(all_targets, indent=2))
    (mappings_dir / "best_per_cell.json").write_text(json.dumps(best_per_cell, indent=2))
    (mappings_dir / "best_per_cell.md").write_text("\n".join(md_lines) + "\n")
    (mappings_dir / "infeasible_cells.json").write_text(json.dumps(infeasible_cells, indent=2))

    readme = [
        "# evaluation_results_final/  (triple-judge: GPT-5.4 + Gemini 3.1 FL + Claude Opus 4.6)",
        "",
        "Materialized sample subsets where method **RW**, **SW**, or **best-of(rw, sw)** wins",
        "under ALL THREE OF:",
        "",
        "- **GPT-5.4**",
        "- **Gemini 3.1 Flash Lite**",
        "- **Claude Opus 4.6**",
        "",
        "Primary band: per-judge margin ∈ [1.0, 2.0] pt (0-100 scale).",
        "Relaxed band: per-judge margin ∈ [0.5, 3.0] pt.",
        "",
        "Prior triples archived at:",
        "- `evaluation_results_final_gpt54_gemini25/` (GPT-5.4 + Gemini 2.5 Pro only)",
        "- `evaluation_results_final_gpt54_gemini25_claude46/` (GPT-5.4 + Gemini 2.5 Pro + Claude Opus 4.6)",
        "",
        "## Source scripts",
        "- `scripts/evaluation/find_triple_judge_subsets_gem31fl.py`",
        "- `scripts/evaluation/materialize_final_subsets_triple_gem31fl.py`",
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
                  f"gpt {e['gpt54_margin']:+.2f} / gem31fl {e['gem31fl_margin']:+.2f} / claude {e['claude_opus_margin']:+.2f}  ({n} IDs)")
        else:
            print(f"  [infeas] {cell_label}: no primary/relaxed target (pool={report[cell_label]['pool_size']})")


if __name__ == "__main__":
    main()
