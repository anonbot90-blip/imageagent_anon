#!/usr/bin/env python3
"""Replace 3 selected samples whose Edit-Only column was missing.

For each affected pick, scan winner_audit.json candidates (same figure +
same cfg ideally) for the next highest-margin sample that (a) has all 9
predicted_edit.png files on disk, (b) does not overlap with any other
already-selected sample_id, and (c) preserves the theme-diversity rule
within the figure.

Writes: updated figure_selections.json + a small log.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/"scripts/evaluation/winner_audit.json"
SEL = ROOT/"scripts/evaluation/figure_selections.json"
EVAL = ROOT/"evaluation_results"

CFG2DIR = {
    "simple_text_4b":    ("simple",  "text_parallel_cot_4b_trajectory"),
    "simple_text_8b":    ("simple",  "text_parallel_cot_8b_trajectory"),
    "simple_vision_4b":  ("simple",  "vision_parallel_cot_4b_trajectory"),
    "simple_vision_8b":  ("simple",  "vision_parallel_cot_8b_trajectory"),
    "normal_text_4b":    ("normal",  "text_parallel_cot_4b_trajectory"),
    "normal_text_8b":    ("normal",  "text_parallel_cot_8b_trajectory"),
    "normal_vision_4b":  ("normal",  "vision_parallel_cot_4b_trajectory"),
    "normal_vision_8b":  ("normal",  "vision_parallel_cot_8b_trajectory"),
    "complex_text_4b":   ("complex", "text_parallel_cot_4b_trajectory"),
    "complex_text_8b":   ("complex", "text_parallel_cot_8b_trajectory"),
    "complex_vision_4b": ("complex", "vision_parallel_cot_4b_trajectory"),
    "complex_vision_8b": ("complex", "vision_parallel_cot_8b_trajectory"),
}
METHOD_DIRS = ["baseline","edit_only","standard_text","rl_text","dpo_text","rw_text","sw_text","gemini25"]


def has_all_nine(cfg_label, sid):
    ds,c = CFG2DIR[cfg_label]
    d = EVAL/ds/c
    if not (d/"baseline/samples"/sid/"original.png").exists():
        return False
    for m in METHOD_DIRS:
        if not (d/m/"samples"/sid/"predicted_edit.png").exists():
            return False
    return True


def theme_key(sid):
    parts = sid.split("_")
    if parts and parts[0] == "image":
        parts = parts[1:]
    if parts:
        parts = parts[1:]
    while parts and (parts[0] == "v2" or (parts[0].startswith("l") and parts[0][1:].isdigit()) or parts[0].isdigit()):
        parts = parts[1:]
    while parts and parts[-1] in {"multi","complex","dual","triple"}:
        parts = parts[:-1]
    return "_".join(parts) if parts else sid


def candidates_for(audit, tgt_key, cfg_label):
    rk = audit.get(cfg_label, {})
    top = rk.get(tgt_key, [])
    out = []
    for entry in top:
        if tgt_key == "rw_sw":
            margin, sid, which, score, opp, opp_s, g25 = entry
            meta = {"method_picked": which}
        else:
            margin, sid, score, opp, opp_s, g25 = entry
            meta = {"method_picked": tgt_key}
        out.append({
            "cfg": cfg_label,
            "sample_id": sid,
            "margin": float(margin),
            "score": float(score),
            "opp": opp,
            "opp_score": float(opp_s),
            "g25": float(g25),
            "theme": theme_key(sid),
            **meta,
        })
    return out


def main():
    audit = json.loads(AUDIT.read_text())
    sel = json.loads(SEL.read_text())

    # Gather global used_ids and per-figure used themes
    figs = ["teaser","rw_sw","dpo","rl"]
    used_ids = set()
    per_fig_themes = {f: set() for f in figs}
    per_fig_cfgs  = {f: set() for f in figs}
    for f in figs:
        for p in sel[f]:
            used_ids.add(p["sample_id"])
            per_fig_themes[f].add(p["theme"])
            per_fig_cfgs[f].add(p["cfg"])

    # Identify affected: any pick where edit_only is missing on disk
    affected = []  # (fig, idx)
    for f in figs:
        for i, p in enumerate(sel[f]):
            if not has_all_nine(p["cfg"], p["sample_id"]):
                affected.append((f, i))
    print("Affected picks:")
    for f,i in affected:
        p = sel[f][i]
        print(f"  {f}[{i}]  {p['cfg']}  {p['sample_id']}")

    # Target key used to rank inside audit for each figure
    fig2tgt = {"teaser":"sw", "rw_sw":"rw_sw", "dpo":"dpo", "rl":"rl"}

    for f, i in affected:
        old = sel[f][i]
        # Remove the old entry's bookkeeping
        used_ids.discard(old["sample_id"])
        per_fig_themes[f].discard(old["theme"])
        per_fig_cfgs[f].discard(old["cfg"])

        # First try same cfg; if none, try other cfgs
        candidates = candidates_for(audit, fig2tgt[f], old["cfg"])
        replacement = None
        # preference 1: same cfg
        for c in candidates:
            if c["sample_id"] in used_ids: continue
            if c["theme"] in per_fig_themes[f]: continue
            if not has_all_nine(c["cfg"], c["sample_id"]): continue
            replacement = c; break
        # preference 2: any other cfg not already used within this figure
        if replacement is None:
            all_cfgs = [k for k in audit.keys() if k != old["cfg"] and k not in per_fig_cfgs[f]]
            # rank by margin within each cfg's top-of-list, choose global best
            pool = []
            for cfg in all_cfgs:
                pool.extend(candidates_for(audit, fig2tgt[f], cfg))
            pool.sort(key=lambda d: d["margin"], reverse=True)
            for c in pool:
                if c["sample_id"] in used_ids: continue
                if c["theme"] in per_fig_themes[f]: continue
                if c["cfg"] in per_fig_cfgs[f]: continue
                if not has_all_nine(c["cfg"], c["sample_id"]): continue
                replacement = c; break

        if replacement is None:
            raise RuntimeError(f"No viable replacement for {f}[{i}] {old['cfg']}")

        print(f"REPLACE {f}[{i}]:")
        print(f"  old: {old['cfg']} / {old['sample_id']}")
        print(f"  new: {replacement['cfg']} / {replacement['sample_id']} "
              f"(Δ={replacement['margin']:+.2f}, score={replacement['score']:.2f}, g25={replacement['g25']:.2f})")
        sel[f][i] = replacement
        used_ids.add(replacement["sample_id"])
        per_fig_themes[f].add(replacement["theme"])
        per_fig_cfgs[f].add(replacement["cfg"])

    SEL.write_text(json.dumps(sel, indent=2, default=str))
    print(f"\nWrote updated {SEL}")


if __name__ == "__main__":
    main()
