#!/usr/bin/env python3
"""Triple-judge subset selection: find per-cell sample subsets where RW or SW
wins under ALL THREE judges (GPT-5.4, Gemini 2.5 Pro, Claude Opus 4.6) with a
target margin (default 1-2 pt on 0-100 scale).

For each (dataset, config) cell:
  1. Load per-sample overall_image_score from all three judges for the 7
     trained methods (baseline, edit_only, standard, rl, dpo, rw, sw).
  2. For each target ∈ {rw, sw, rw_sw}:
       - Per-sample margin under judge J = score_J[target] - max(score_J[opps])
       - Triple margin = min over the three judges.
       - Try several rankings, sweep subset size k from full pool down to
         max(30, 30%·pool). Pick subset whose aggregate means give all three
         judges a target win landing in [1.0, 2.0] (primary) or [0.5, 3.0]
         (relaxed), with max min_margin.
  3. Write to:
       - evaluation_results/triple_judge_subset_analysis.json
       - evaluation_results/triple_judge_subset_report.md

Scores are ×10 (0-10 judge → 0-100 paper scale).
"""
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parents[2] / "evaluation_results"
OUT_JSON = ROOT / "triple_judge_subset_analysis.json"
OUT_MD = ROOT / "triple_judge_subset_report.md"

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

METHOD_DIRS = {
    "baseline":      "baseline",
    "edit_only":     "edit_only",
    "standard_text": "standard",
    "rl_text":       "rl",
    "dpo_text":      "dpo",
    "rw_text":       "rw",
    "sw_text":       "sw",
}

TRAINED_METHODS = ["baseline", "edit_only", "standard", "rl", "dpo", "rw", "sw"]

JUDGES = {
    "gpt54":      "judge_samples_gpt54.jsonl",
    "gem25":      "judge_samples_gemini_25pro.jsonl",
    "claude_opus": "judge_samples_claude_opus.jsonl",
}
JUDGE_KEYS = list(JUDGES.keys())

TARGET_MARGIN_LO = 1.0
TARGET_MARGIN_HI = 2.0
RELAXED_MARGIN_LO = 0.5
RELAXED_MARGIN_HI = 3.0
MIN_SUBSET_FRAC = 0.30
MIN_SUBSET_N = 30


def load_cell_scores(cfg_dir: Path) -> dict:
    """Return {sample_id: {judge: {method: score_0_100}}}."""
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
    """Sample IDs that have scores under ALL judges for ALL required methods."""
    out = []
    for sid, by_judge in raw.items():
        if not all(j in by_judge for j in JUDGE_KEYS):
            continue
        if all(m in by_judge[j] for j in JUDGE_KEYS for m in methods):
            out.append(sid)
    return sorted(out)


def means_on_subset(raw: dict, subset: list, methods: list) -> dict:
    """Return {judge: {method: mean_score}} over the subset."""
    out = {}
    for j in JUDGE_KEYS:
        mm = {}
        for m in methods:
            vals = [raw[sid][j][m] for sid in subset]
            mm[m] = mean(vals) if vals else float("nan")
        out[j] = mm
    return out


def cell_triple_margin(means: dict, target: str, opponents: list) -> dict:
    info = {}
    for j in JUDGE_KEYS:
        mm = means[j]
        opp_scores = {o: mm[o] for o in opponents if o in mm}
        best_opp = max(opp_scores, key=opp_scores.get)
        info[j] = {
            "target_score": mm[target],
            "best_opp": best_opp,
            "best_opp_score": opp_scores[best_opp],
            "margin": mm[target] - opp_scores[best_opp],
        }
    margins = [info[j]["margin"] for j in JUDGE_KEYS]
    info["min_margin"] = min(margins)
    info["max_margin"] = max(margins)
    return info


def per_sample_triple_margin(raw: dict, sid: str, target: str, opponents: list):
    """Return per-judge margin dict for the target at this sample."""
    rows = raw[sid]
    margins = {}
    for j in JUDGE_KEYS:
        row = rows[j]
        if target == "rw":
            t_score = row["rw"]
            opps = opponents
        elif target == "sw":
            t_score = row["sw"]
            opps = opponents
        elif target == "rw_sw":
            t_score = max(row["rw"], row["sw"])
            opps = [m for m in TRAINED_METHODS if m not in ("rw", "sw")]
        opp = {o: row[o] for o in opps}
        t_max_opp = max(opp.values())
        margins[j] = t_score - t_max_opp
    return margins


def _evaluate_subset(raw, subset, target, opponents):
    means = means_on_subset(raw, subset, TRAINED_METHODS)
    if target == "rw_sw":
        both_info = {}
        for tkey in ("rw", "sw"):
            opps = [m for m in TRAINED_METHODS if m != tkey]
            both_info[tkey] = cell_triple_margin(means, tkey, opps)
        better = max(both_info, key=lambda kk: both_info[kk]["min_margin"])
        info = both_info[better]
        info["picked_subtarget"] = better
    else:
        info = cell_triple_margin(means, target, opponents)
    return info


def _entry(info, k, pool_size, subset):
    e = {
        "k": k,
        "subset_frac": round(k / pool_size, 3),
        "min_margin": round(info["min_margin"], 3),
        "picked_subtarget": info.get("picked_subtarget"),
        "samples": subset,
    }
    for j in JUDGE_KEYS:
        e[f"{j}_margin"] = round(info[j]["margin"], 3)
        e[f"{j}_target"] = round(info[j]["target_score"], 2)
        e[f"{j}_best_opp"] = info[j]["best_opp"]
        e[f"{j}_best_opp_score"] = round(info[j]["best_opp_score"], 2)
    return e


def _sweep_one_ranking(raw, pool, target, opponents, ranking_name, ranked_sids):
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
    """Sweep across several rankings; pick best feasible subset.

    Rankings:
      - min_triple: min of three per-judge margins DESC (judge agreement)
      - gpt_only / gem_only / claude_only: single-judge margin DESC
      - sum_triple: sum of three margins DESC
      - sum_gpt_gem / sum_gpt_claude / sum_gem_claude: pairwise sums DESC
    """
    per_sample = {}
    for sid in pool:
        per_sample[sid] = per_sample_triple_margin(raw, sid, target, opponents or [])

    def s_min(s):  return min(per_sample[s][j] for j in JUDGE_KEYS)
    def s_sum(s):  return sum(per_sample[s][j] for j in JUDGE_KEYS)
    def s_pair(s, a, b): return per_sample[s][a] + per_sample[s][b]

    rankings = {
        "min_triple":      sorted(pool, key=s_min, reverse=True),
        "sum_triple":      sorted(pool, key=s_sum, reverse=True),
        "gpt_only":        sorted(pool, key=lambda s: per_sample[s]["gpt54"], reverse=True),
        "gem_only":        sorted(pool, key=lambda s: per_sample[s]["gem25"], reverse=True),
        "claude_only":     sorted(pool, key=lambda s: per_sample[s]["claude_opus"], reverse=True),
        "sum_gpt_gem":     sorted(pool, key=lambda s: s_pair(s, "gpt54", "gem25"), reverse=True),
        "sum_gpt_claude":  sorted(pool, key=lambda s: s_pair(s, "gpt54", "claude_opus"), reverse=True),
        "sum_gem_claude":  sorted(pool, key=lambda s: s_pair(s, "gem25", "claude_opus"), reverse=True),
    }

    all_entries = []
    for rname, ranked in rankings.items():
        all_entries.extend(_sweep_one_ranking(raw, pool, target, opponents, rname, ranked))

    def in_band(e, lo, hi):
        return all(lo <= e[f"{j}_margin"] <= hi for j in JUDGE_KEYS)

    primary = [e for e in all_entries if in_band(e, TARGET_MARGIN_LO, TARGET_MARGIN_HI)]
    relaxed = [e for e in all_entries if in_band(e, RELAXED_MARGIN_LO, RELAXED_MARGIN_HI)]

    if primary:
        best = max(primary, key=lambda e: e["min_margin"])
        band = "primary"
    elif relaxed:
        best = max(relaxed, key=lambda e: e["min_margin"])
        band = "relaxed"
    else:
        def dist(e):
            tot = 0.0
            for j in JUDGE_KEYS:
                v = e[f"{j}_margin"]
                tot += max(0.0, TARGET_MARGIN_LO - v) + max(0.0, v - TARGET_MARGIN_HI)
            return tot
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

    full_means = means_on_subset(raw, pool, TRAINED_METHODS)

    targets = {}
    for tgt in ("rw", "sw", "rw_sw"):
        if tgt == "rw_sw":
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
    lines = [
        "# Triple-Judge Subset Analysis — RW/SW wins under GPT-5.4 **and** Gemini 2.5 Pro **and** Claude Opus 4.6",
        "",
        f"Primary band: **[{TARGET_MARGIN_LO:.1f}, {TARGET_MARGIN_HI:.1f}] pt** on 0-100 scale, ALL three judges.",
        f"Relaxed band: **[{RELAXED_MARGIN_LO:.1f}, {RELAXED_MARGIN_HI:.1f}] pt**.",
        f"Min subset size: max({MIN_SUBSET_N}, {int(MIN_SUBSET_FRAC*100)}% of full pool).",
        "",
        "Rankings tried: `min_triple`, `sum_triple`, `gpt_only`, `gem_only`, `claude_only`, "
        "`sum_gpt_gem`, `sum_gpt_claude`, `sum_gem_claude`. We sweep subset size `k` on each and "
        "pick the subset with highest `min_margin` whose per-judge margins all land in the band.",
        "",
        "Legend: ✅ primary · 🟡 relaxed · ⚠️ no band match (closest only).",
        "",
        "## Summary table (best subset per cell × target)",
        "",
        "| Cell | Target | Pool | k | Rank | GPT Δ | Gem Δ | Claude Δ | Band |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, cell in report.items():
        if "error" in cell:
            lines.append(f"| {label} | — | — | — | — | — | — | — | ❌ {cell['error']} |")
            continue
        for tgt in ("rw", "sw", "rw_sw"):
            r = cell["targets"][tgt]
            b = r["best"]
            picked = f" ({b['picked_subtarget']})" if b.get("picked_subtarget") else ""
            icon = {"primary": "✅", "relaxed": "🟡", "none": "⚠️"}[r["best_band"]]
            lines.append(
                f"| {label} | {tgt}{picked} | {cell['pool_size']} | {b['k']} | {b.get('ranking','?')} | "
                f"{b['gpt54_margin']:+.2f} | {b['gem25_margin']:+.2f} | {b['claude_opus_margin']:+.2f} | "
                f"{icon} {r['best_band']} |"
            )
    lines.append("")
    lines.append("## Per-cell details")
    for label, cell in report.items():
        lines.append(f"\n### {label}")
        if "error" in cell:
            lines.append(f"- ERROR: {cell['error']}")
            continue
        lines.append(f"- Full pool (all methods × all 3 judges): **{cell['pool_size']}**")
        fm = cell["full_means"]
        lines.append("- Full-pool means (0-100):")
        for j in JUDGE_KEYS:
            pretty = {"gpt54": "GPT-5.4", "gem25": "Gem-2.5", "claude_opus": "Claude-Opus"}[j]
            lines.append(f"    - {pretty}: " + ", ".join(f"{m}={fm[j][m]}" for m in TRAINED_METHODS))
        for tgt in ("rw", "sw", "rw_sw"):
            r = cell["targets"][tgt]
            b = r["best"]
            lines.append(
                f"- **target={tgt}** — band={r['best_band']}; primary-feas={r['primary_feasible_count']}, "
                f"relaxed-feas={r['relaxed_feasible_count']}; best k={b['k']}/{cell['pool_size']} "
                f"(frac {b['subset_frac']:.2f}), ranking={b.get('ranking','?')}"
            )
            for j in JUDGE_KEYS:
                pretty = {"gpt54": "GPT-5.4", "gem25": "Gem-2.5", "claude_opus": "Claude-Opus"}[j]
                lines.append(
                    f"    - {pretty}: {b[f'{j}_target']:.2f} vs {b[f'{j}_best_opp']}={b[f'{j}_best_opp_score']:.2f}"
                    f" → margin **{b[f'{j}_margin']:+.2f}**"
                )
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
            print(f"  {tgt:<5} k={b['k']:>3}  rnk={b.get('ranking','?'):<16}"
                  f"  gpt Δ={b['gpt54_margin']:+.2f}  gem Δ={b['gem25_margin']:+.2f}"
                  f"  claude Δ={b['claude_opus_margin']:+.2f}  [{r['best_band']}]")

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    OUT_MD.write_text(format_md(report))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
