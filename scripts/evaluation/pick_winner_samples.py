#!/usr/bin/env python3
"""Pick per-sample "winner" samples for composite figures.

For each (dataset, config) and each target method in {rw, sw, dpo, rl}, rank
samples by the margin:
    target_method_score - max(other_methods_shown_in_composite)

Other methods shown in composite: baseline, edit_only, standard, rl, dpo, rw, sw.
Gemini 2.5 is also scored so we can require target > gemini25 as a tiebreaker
(we want examples where the method beats the frontier zero-shot).

Scores: GPT-5.4 image judge overall_image_score (0-10) from judge_samples_gpt54.jsonl.

Output: JSON with top candidates per figure/config.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2] / "evaluation_results"

# (dataset, config_dir_name, human_label)
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

# dir-name : short method key shown in composite. All configs (text & vision)
# use the same "_text" suffixed dirs on disk.
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
VISION_METHOD_DIRS = METHOD_DIRS


def load_scores(config_dir: Path, method_dirs: dict) -> dict:
    """Return {sample_id: {method_key: score}}."""
    scores = defaultdict(dict)
    for dname, key in method_dirs.items():
        jsonl = config_dir / dname / "judge_samples_gpt54.jsonl"
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
                if sid is not None and isinstance(s, (int, float)):
                    scores[sid][key] = float(s)
    return scores


def rank_for_method(scores: dict, target: str, opponents: list, require_beat_g25=True):
    """Return sorted list of (margin, sample_id, target_score, best_opp_key, best_opp_score, g25)."""
    out = []
    for sid, m in scores.items():
        if target not in m:
            continue
        opp_scores = {k: m[k] for k in opponents if k in m}
        if not opp_scores:
            continue
        best_opp_key = max(opp_scores, key=opp_scores.get)
        best_opp = opp_scores[best_opp_key]
        g25 = m.get("gemini25")
        if require_beat_g25 and (g25 is None or m[target] <= g25):
            continue
        margin = m[target] - best_opp
        out.append((margin, sid, m[target], best_opp_key, best_opp, g25))
    out.sort(reverse=True)
    return out


def main():
    report = {}
    # For "rw_sw" figure we want the BEST of rw or sw to lead; rank by
    # max(rw, sw) - max(other_non_rw_sw).
    for ds, cfg_dir, label in CONFIGS:
        is_vision = "vision" in cfg_dir
        method_dirs = VISION_METHOD_DIRS if is_vision else METHOD_DIRS
        scores = load_scores(ROOT / ds / cfg_dir, method_dirs)
        if not scores:
            report[label] = {"error": "no scores loaded"}
            continue

        # per-method rankings
        trained = ["baseline", "edit_only", "standard", "rl", "dpo", "rw", "sw"]
        rankings = {}
        for tgt in ["rw", "sw", "dpo", "rl"]:
            others = [m for m in trained if m != tgt]
            rankings[tgt] = rank_for_method(scores, tgt, others)[:10]

        # rw_sw combined: for each sample, score_rs = max(rw, sw), winner=which one
        rs = []
        for sid, m in scores.items():
            if "rw" not in m and "sw" not in m:
                continue
            rs_pick = "rw" if m.get("rw", -1) >= m.get("sw", -1) else "sw"
            rs_score = m[rs_pick]
            opp_keys = ["baseline", "edit_only", "standard", "rl", "dpo"]
            opp_scores = {k: m[k] for k in opp_keys if k in m}
            if not opp_scores:
                continue
            best_opp_key = max(opp_scores, key=opp_scores.get)
            best_opp = opp_scores[best_opp_key]
            g25 = m.get("gemini25")
            if g25 is None or rs_score <= g25:
                continue
            margin = rs_score - best_opp
            rs.append((margin, sid, rs_pick, rs_score, best_opp_key, best_opp, g25))
        rs.sort(reverse=True)
        rankings["rw_sw"] = rs[:10]

        report[label] = rankings

    out_path = Path(__file__).resolve().parent / "winner_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    # Print compact summary
    for label, rk in report.items():
        print(f"\n=== {label} ===")
        if "error" in rk:
            print(f"  {rk['error']}")
            continue
        for tgt, top in rk.items():
            if not top:
                print(f"  {tgt}: (no candidates)")
                continue
            best = top[0]
            if tgt == "rw_sw":
                margin, sid, pick, score, opp, opp_s, g25 = best
                print(f"  {tgt}: Δ={margin:+.2f}  {pick}={score:.2f}  {opp}={opp_s:.2f}  g25={g25:.2f}  {sid}")
            else:
                margin, sid, score, opp, opp_s, g25 = best
                print(f"  {tgt}: Δ={margin:+.2f}  {tgt}={score:.2f}  {opp}={opp_s:.2f}  g25={g25:.2f}  {sid}")


if __name__ == "__main__":
    main()
