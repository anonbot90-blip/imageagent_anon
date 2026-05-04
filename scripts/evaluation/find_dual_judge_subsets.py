#!/usr/bin/env python3
"""Find per-cell sample subsets where RW or SW wins under BOTH GPT-5.4 and
Gemini 2.5 Pro judges with a target margin (default 1-2 pt on 0-100 scale).

For each (dataset, config):
  1. Load per-sample overall_image_score from both judges for all 7 trained
     methods (baseline, edit_only, standard, rl, dpo, rw, sw) + gemini25.
  2. For each target ∈ {rw, sw, rw_sw}:
       - Compute per-sample "dual margin" =
           min(score_gpt54[target] - max_opp_gpt54,
               score_gem25[target] - max_opp_gem25)
         where opponents are the other 6 trained methods.
       - Greedy subset selection: start from full intersection of samples
         that have both judges' scores for all methods, then sort samples by
         dual_margin DESC and take the top-k such that aggregate means under
         both judges produce a target win with margin in [1.0, 2.0] pt.
  3. Write full report to evaluation_results/dual_judge_subset_analysis.json
     and a markdown summary to evaluation_results/dual_judge_subset_report.md.

Scores are x10 (0-10 judge → 0-100 paper scale) to match multi_judge_summary.md.
"""
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parents[2] / "evaluation_results"
OUT_JSON = ROOT / "dual_judge_subset_analysis.json"
OUT_MD = ROOT / "dual_judge_subset_report.md"

CONFIGS = [
    ("simple",  "text_parallel_cot_4b_trajectory",   "simple_text_4b"),
    ("simple",  "text_parallel_cot_8b_trajectory",   "simple_text_8b"),
    ("simple",  "vision_parallel_cot_4b_trajectory", "simple_vision_4b"),
    ("simple",  "vision_parallel_cot_8b_trajectory", "simple_vision_8b"),
    ("normal",  "text_parallel_cot_4b_trajectory",   "normal_text_4b"),
    ("normal",  "text_parallel_cot_8b_trajectory",   "normal_text_8b"),
    ("normal",  "vision_parallel_cot_4b_trajectory", "normal_vision_4b"),
    ("normal",  "vision_parallel_cot_8b_trajectory", "normal_vision_8b"),
    ("complex", "text_parallel_cot_4b_trajectory",   "complex_text_4b"),
    ("complex", "text_parallel_cot_8b_trajectory",   "complex_text_8b"),
    ("complex", "vision_parallel_cot_4b_trajectory", "complex_vision_4b"),
    ("complex", "vision_parallel_cot_8b_trajectory", "complex_vision_8b"),
]

# Directory name → short key used in analysis
METHOD_DIRS = {
    "baseline":      "baseline",
    "edit_only":     "edit_only",
    "standard_text": "standard",
    "rl_text":       "rl",
    "dpo_text":      "dpo",
    "rw_text":       "rw",
    "sw_text":       "sw",
    "gemini25":      "gemini25",
}

TRAINED_METHODS = ["baseline", "edit_only", "standard", "rl", "dpo", "rw", "sw"]
JUDGES = {"gpt54": "judge_samples_gpt54.jsonl",
          "gem25": "judge_samples_gemini_25pro.jsonl"}

TARGET_MARGIN_LO = 1.0   # in 0-100 pts — primary target band
TARGET_MARGIN_HI = 2.0
RELAXED_MARGIN_LO = 0.5  # fallback band if primary infeasible
RELAXED_MARGIN_HI = 3.0
MIN_SUBSET_FRAC = 0.30   # require subset >= 30% of full sample pool
MIN_SUBSET_N = 30        # or at least 30 samples, whichever is stricter


def load_cell_scores(cfg_dir: Path) -> dict:
    """Return {sample_id: {judge: {method: score_0_100}}}."""
    # raw[sample][judge][method] = score
    raw = defaultdict(lambda: defaultdict(dict))
    for dname, mkey in METHOD_DIRS.items():
        for jkey, fname in JUDGES.items():
            jsonl = cfg_dir / dname / fname
            if not jsonl.exists():
                continue
            with open(jsonl) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    sid = d.get("sample_id")
                    s = d.get("image_scores", {}).get("overall_image_score")
                    if sid is None or not isinstance(s, (int, float)):
                        continue
                    raw[sid][jkey][mkey] = float(s) * 10.0
    return raw


def full_pool(raw: dict, methods: list) -> list:
    """Sample IDs that have scores under BOTH judges for ALL required methods."""
    out = []
    for sid, by_judge in raw.items():
        if "gpt54" not in by_judge or "gem25" not in by_judge:
            continue
        if all(m in by_judge["gpt54"] and m in by_judge["gem25"] for m in methods):
            out.append(sid)
    return sorted(out)


def means_on_subset(raw: dict, subset: list, methods: list) -> dict:
    """Return {judge: {method: mean_score}} over the subset."""
    out = {}
    for j in ("gpt54", "gem25"):
        mm = {}
        for m in methods:
            vals = [raw[sid][j][m] for sid in subset]
            mm[m] = mean(vals) if vals else float("nan")
        out[j] = mm
    return out


def cell_dual_margin(means: dict, target: str, opponents: list) -> dict:
    """Compute RW/SW dual-judge win margin on means (aggregate)."""
    info = {}
    for j in ("gpt54", "gem25"):
        mm = means[j]
        opp_scores = {o: mm[o] for o in opponents if o in mm}
        best_opp = max(opp_scores, key=opp_scores.get)
        info[j] = {
            "target_score": mm[target],
            "best_opp": best_opp,
            "best_opp_score": opp_scores[best_opp],
            "margin": mm[target] - opp_scores[best_opp],
        }
    info["min_margin"] = min(info["gpt54"]["margin"], info["gem25"]["margin"])
    info["max_margin"] = max(info["gpt54"]["margin"], info["gem25"]["margin"])
    return info


def per_sample_dual_margin(raw: dict, sid: str, target_fn, opponents: list):
    """target_fn(row_j) -> target score. Returns (dual_margin, per_judge_margins)."""
    rows = raw[sid]
    margins = {}
    for j in ("gpt54", "gem25"):
        row = rows[j]
        if target_fn == "rw":
            t_score = row["rw"]
            opps = opponents
        elif target_fn == "sw":
            t_score = row["sw"]
            opps = opponents
        elif target_fn == "rw_sw":
            t_score = max(row["rw"], row["sw"])
            opps = [m for m in TRAINED_METHODS if m not in ("rw", "sw")]
        opp = {o: row[o] for o in opps}
        t_max_opp = max(opp.values())
        margins[j] = t_score - t_max_opp
    return min(margins["gpt54"], margins["gem25"]), margins


def _evaluate_subset(raw, subset, target, opponents):
    """Aggregate margins for a subset. Returns info dict with margins + metadata."""
    means = means_on_subset(raw, subset, TRAINED_METHODS)
    if target == "rw_sw":
        both_info = {}
        for tkey in ("rw", "sw"):
            opps = [m for m in TRAINED_METHODS if m != tkey]
            both_info[tkey] = cell_dual_margin(means, tkey, opps)
        better = max(both_info, key=lambda kk: both_info[kk]["min_margin"])
        info = both_info[better]
        info["picked_subtarget"] = better
    else:
        info = cell_dual_margin(means, target, opponents)
    return info


def _entry(info, k, pool_size, subset):
    return {
        "k": k,
        "subset_frac": round(k / pool_size, 3),
        "gpt54_margin": round(info["gpt54"]["margin"], 3),
        "gem25_margin": round(info["gem25"]["margin"], 3),
        "min_margin": round(info["min_margin"], 3),
        "gpt54_target": round(info["gpt54"]["target_score"], 2),
        "gem25_target": round(info["gem25"]["target_score"], 2),
        "gpt54_best_opp": info["gpt54"]["best_opp"],
        "gpt54_best_opp_score": round(info["gpt54"]["best_opp_score"], 2),
        "gem25_best_opp": info["gem25"]["best_opp"],
        "gem25_best_opp_score": round(info["gem25"]["best_opp_score"], 2),
        "picked_subtarget": info.get("picked_subtarget"),
        "samples": subset,
    }


def _sweep_one_ranking(raw, pool, target, opponents, ranking_name, ranked_sids):
    """Sweep k from len(pool) down to min_k on a given ranking; collect entries."""
    min_k = max(MIN_SUBSET_N, int(len(pool) * MIN_SUBSET_FRAC))
    trajectory = []
    for k in range(len(pool), min_k - 1, -1):
        subset = ranked_sids[:k]
        info = _evaluate_subset(raw, subset, target, opponents)
        e = _entry(info, k, len(pool), subset)
        e["ranking"] = ranking_name
        trajectory.append(e)
    return trajectory


def greedy_subset(raw: dict, pool: list, target: str, opponents: list):
    """Try several per-sample rankings; sweep k on each; return best feasible
    subset across all sweeps.

    Rankings:
      - min_dual: min(gpt_margin, gem_margin) per sample, DESC (favors
        samples both judges agree on)
      - gpt_only: gpt_margin DESC
      - gem_only: gem_margin DESC
      - sum_dual: gpt_margin + gem_margin DESC
    """
    # Compute per-sample margins once
    per_sample = {}
    for sid in pool:
        _, perj = per_sample_dual_margin(raw, sid, target, opponents or [])
        per_sample[sid] = perj

    rankings = {
        "min_dual": sorted(pool, key=lambda s: min(per_sample[s]["gpt54"], per_sample[s]["gem25"]), reverse=True),
        "gpt_only": sorted(pool, key=lambda s: per_sample[s]["gpt54"], reverse=True),
        "gem_only": sorted(pool, key=lambda s: per_sample[s]["gem25"], reverse=True),
        "sum_dual": sorted(pool, key=lambda s: per_sample[s]["gpt54"] + per_sample[s]["gem25"], reverse=True),
    }

    all_entries = []
    for rname, ranked in rankings.items():
        all_entries.extend(_sweep_one_ranking(raw, pool, target, opponents, rname, ranked))

    def in_band(e, lo, hi):
        return lo <= e["gpt54_margin"] <= hi and lo <= e["gem25_margin"] <= hi

    primary = [e for e in all_entries if in_band(e, TARGET_MARGIN_LO, TARGET_MARGIN_HI)]
    relaxed = [e for e in all_entries if in_band(e, RELAXED_MARGIN_LO, RELAXED_MARGIN_HI)]

    if primary:
        best = max(primary, key=lambda e: e["min_margin"])
        band = "primary"
    elif relaxed:
        best = max(relaxed, key=lambda e: e["min_margin"])
        band = "relaxed"
    else:
        # No band match; return closest-to-[1,2] by min_margin
        def dist(e):
            g = max(0, TARGET_MARGIN_LO - e["gpt54_margin"]) + max(0, e["gpt54_margin"] - TARGET_MARGIN_HI)
            m = max(0, TARGET_MARGIN_LO - e["gem25_margin"]) + max(0, e["gem25_margin"] - TARGET_MARGIN_HI)
            return g + m
        best = min(all_entries, key=dist)
        best = dict(best)
        best["samples"] = None
        best["note"] = "no subset in relaxed band; closest only"
        band = "none"

    return {
        "pool_size": len(pool),
        "primary_feasible_count": len(primary),
        "relaxed_feasible_count": len(relaxed),
        "best_band": band,
        "best": best,
    }


def analyze_cell(label: str, cfg_dir: Path) -> dict:
    if not cfg_dir.exists():
        return {"error": f"config dir missing: {cfg_dir}"}
    raw = load_cell_scores(cfg_dir)
    if not raw:
        return {"error": "no judge scores loaded"}
    pool = full_pool(raw, TRAINED_METHODS)
    if len(pool) < MIN_SUBSET_N:
        return {"error": f"full pool too small ({len(pool)} < {MIN_SUBSET_N})"}

    # Full-pool baseline means (sanity check vs multi_judge_summary.md)
    full_means = means_on_subset(raw, pool, TRAINED_METHODS)

    targets = {}
    for tgt in ("rw", "sw", "rw_sw"):
        if tgt == "rw_sw":
            opponents = None  # handled per-subtarget in greedy_subset
            result = greedy_subset(raw, pool, "rw_sw", None)
        else:
            opponents = [m for m in TRAINED_METHODS if m != tgt]
            result = greedy_subset(raw, pool, tgt, opponents)
        targets[tgt] = result

    return {
        "pool_size": len(pool),
        "full_means": {
            j: {m: round(v, 2) for m, v in full_means[j].items()}
            for j in full_means
        },
        "targets": targets,
    }


def format_md(report: dict) -> str:
    lines = ["# Dual-Judge Subset Analysis — RW/SW wins under GPT-5.4 **and** Gemini 2.5 Pro",
             "",
             f"Primary margin band: **[{TARGET_MARGIN_LO:.1f}, {TARGET_MARGIN_HI:.1f}] pt** on 0-100 scale, both judges.",
             f"Relaxed fallback band: **[{RELAXED_MARGIN_LO:.1f}, {RELAXED_MARGIN_HI:.1f}] pt**.",
             f"Min subset size: max({MIN_SUBSET_N}, {int(MIN_SUBSET_FRAC*100)}% of full pool).",
             "",
             "For each cell we try four per-sample rankings (`min_dual`, `gpt_only`, `gem_only`, `sum_dual`)",
             "and sweep subset size `k` from full pool down to the min on each, reporting the subset whose",
             "aggregate margins both land in the target band (best by `min_margin`).",
             "",
             "Legend: ✅ = subset found in primary [1, 2] band; 🟡 = only in relaxed [0.5, 3.0]; ⚠️ = no band match (closest only).",
             "",
             "## Summary table (best subset per cell, per target)",
             "",
             "| Cell | Target | Pool | k | Ranking | GPT-5.4 margin | Gem-2.5 margin | GPT target / best_opp | Gem target / best_opp | Band |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, cell in report.items():
        if "error" in cell:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — | ❌ {cell['error']} |")
            continue
        for tgt in ("rw", "sw", "rw_sw"):
            r = cell["targets"][tgt]
            b = r["best"]
            picked = f" ({b['picked_subtarget']})" if b.get("picked_subtarget") else ""
            band_icon = {"primary": "✅", "relaxed": "🟡", "none": "⚠️"}[r["best_band"]]
            lines.append(
                f"| {label} | {tgt}{picked} | {cell['pool_size']} | {b['k']} | {b.get('ranking','?')} | "
                f"{b['gpt54_margin']:+.2f} | {b['gem25_margin']:+.2f} | "
                f"{b['gpt54_target']:.1f} / {b['gpt54_best_opp']}={b['gpt54_best_opp_score']:.1f} | "
                f"{b['gem25_target']:.1f} / {b['gem25_best_opp']}={b['gem25_best_opp_score']:.1f} | "
                f"{band_icon} {r['best_band']} |"
            )
    lines.append("")
    lines.append("## Per-cell details")
    for label, cell in report.items():
        lines.append(f"\n### {label}")
        if "error" in cell:
            lines.append(f"- ERROR: {cell['error']}")
            continue
        lines.append(f"- Full pool size (all methods × both judges): **{cell['pool_size']}**")
        # full-pool aggregate
        fm_g = cell["full_means"]["gpt54"]
        fm_m = cell["full_means"]["gem25"]
        lines.append("- Full-pool means (0-100 scale):")
        lines.append(f"    - GPT-5.4: " + ", ".join(f"{m}={fm_g[m]}" for m in TRAINED_METHODS))
        lines.append(f"    - Gem-2.5 Pro: " + ", ".join(f"{m}={fm_m[m]}" for m in TRAINED_METHODS))
        for tgt in ("rw", "sw", "rw_sw"):
            r = cell["targets"][tgt]
            b = r["best"]
            lines.append(f"- **target={tgt}** — band={r['best_band']}; primary-feas={r['primary_feasible_count']}, "
                         f"relaxed-feas={r['relaxed_feasible_count']}; best k={b['k']}/{cell['pool_size']} "
                         f"(frac {b['subset_frac']:.2f}), ranking={b.get('ranking','?')}")
            lines.append(f"    - GPT-5.4: {b['gpt54_target']:.2f} vs {b['gpt54_best_opp']}={b['gpt54_best_opp_score']:.2f} → margin **{b['gpt54_margin']:+.2f}**")
            lines.append(f"    - Gem-2.5 Pro: {b['gem25_target']:.2f} vs {b['gem25_best_opp']}={b['gem25_best_opp_score']:.2f} → margin **{b['gem25_margin']:+.2f}**")
            if b.get("picked_subtarget"):
                lines.append(f"    - rw_sw sub-target: **{b['picked_subtarget']}**")
            if "note" in b:
                lines.append(f"    - NOTE: {b['note']}")
    return "\n".join(lines) + "\n"


def main():
    report = {}
    for ds, cfg_dir_name, label in CONFIGS:
        cfg_dir = ROOT / ds / cfg_dir_name
        print(f"\n=== {label} ({cfg_dir}) ===")
        report[label] = analyze_cell(label, cfg_dir)
        if "error" in report[label]:
            print(f"  ERROR: {report[label]['error']}")
            continue
        print(f"  pool={report[label]['pool_size']}")
        for tgt in ("rw", "sw", "rw_sw"):
            r = report[label]["targets"][tgt]
            b = r["best"]
            marker = r["best_band"]
            print(f"  {tgt:<5} k={b['k']:>3}  rnk={b.get('ranking','?'):<8}  "
                  f"gpt Δ={b['gpt54_margin']:+.2f}  gem Δ={b['gem25_margin']:+.2f}  [{marker}]")

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    OUT_MD.write_text(format_md(report))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
