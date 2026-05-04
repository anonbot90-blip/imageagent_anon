#!/usr/bin/env python3
"""Select per-figure samples with diversity constraints from winner_audit.json.

Rules:
 - teaser: 4 SW-winning samples spanning different configs
 - rw_sw : 5 samples (best of RW or SW), spanning different configs, no overlap w/ teaser
 - dpo   : 5 DPO-winning samples, no overlap w/ prior picks
 - rl    : 5 RL-winning samples, no overlap w/ prior picks

For each figure we pick the highest-margin sample per config, then prune to
required count while maximizing config + theme diversity (no two samples from
the same config, no two samples sharing a theme keyword).
"""
import json
from pathlib import Path
from collections import OrderedDict

AUDIT = Path(__file__).resolve().parent / "winner_audit.json"
OUT   = Path(__file__).resolve().parent / "figure_selections.json"


def theme_key(sample_id: str) -> str:
    """Extract a short theme slug from sample_id."""
    # e.g., image_ad598802_3_modern_office → modern_office
    # e.g., image_72818451_248_classic_library → classic_library
    # e.g., image_0af6d348_v2_l3_0149_futuristic_weather_conditions_multi → weather_conditions (drop _v2_* hash/priority prefix)
    parts = sample_id.split("_")
    # Drop leading "image" + hash token
    if parts and parts[0] == "image":
        parts = parts[1:]
    if parts:
        parts = parts[1:]  # drop hash
    # Drop v2 / l{n} / 4-digit priority tokens
    while parts and (parts[0] == "v2" or (parts[0].startswith("l") and parts[0][1:].isdigit()) or parts[0].isdigit()):
        parts = parts[1:]
    # Drop trailing complexity tag
    trailing = {"multi", "complex", "dual", "triple"}
    while parts and parts[-1] in trailing:
        parts = parts[:-1]
    return "_".join(parts) if parts else sample_id


def pick(audit, tgt, used_sids, used_themes, used_configs, k, prefer_big_margin=True):
    """Return list of dicts with config, sample_id, score, etc."""
    cands = []
    for cfg_label, rk in audit.items():
        if "error" in rk:
            continue
        top = rk.get(tgt, [])
        for entry in top:
            # rw_sw entries are tuples with different arity
            if tgt == "rw_sw":
                margin, sid, which, score, opp, opp_s, g25 = entry
                meta = {"method_picked": which}
            else:
                margin, sid, score, opp, opp_s, g25 = entry
                meta = {"method_picked": tgt}
            if sid in used_sids:
                continue
            if cfg_label in used_configs:
                continue
            th = theme_key(sid)
            if th in used_themes:
                continue
            cands.append({
                "cfg": cfg_label,
                "sample_id": sid,
                "margin": margin,
                "score": score,
                "opp": opp,
                "opp_score": opp_s,
                "g25": g25,
                "theme": th,
                **meta,
            })
            break  # only top-of-config per cfg
    # sort by margin desc
    cands.sort(key=lambda d: d["margin"], reverse=True)
    picked = []
    local_cfg, local_th = set(), set()
    for c in cands:
        if c["cfg"] in local_cfg or c["theme"] in local_th:
            continue
        picked.append(c)
        local_cfg.add(c["cfg"])
        local_th.add(c["theme"])
        if len(picked) >= k:
            break
    return picked


def main():
    with open(AUDIT) as f:
        audit = json.load(f)

    used_sids, used_themes, used_configs = set(), set(), set()

    # 1. Teaser: 4 SW winners. Allow theme/config reuse across DPO/RL only
    # (teaser+rw_sw are same story -> don't reuse between them).
    teaser = pick(audit, "sw", used_sids, used_themes, used_configs, k=4)
    for t in teaser:
        used_sids.add(t["sample_id"])
        used_themes.add(t["theme"])
        used_configs.add(t["cfg"])

    # 2. rw_sw (5) — share story w/ teaser so also avoid those picks.
    rw_sw = pick(audit, "rw_sw", used_sids, used_themes, used_configs, k=5)
    for t in rw_sw:
        used_sids.add(t["sample_id"])
        used_themes.add(t["theme"])
        used_configs.add(t["cfg"])

    # For DPO and RL, reset the config constraint (different story, fine to
    # reuse configs); keep sample_id uniqueness only.
    # Actually: visual_comparisons.tex describes rows by config, so keep config-unique WITHIN a figure.
    # So we reset used_configs here so DPO/RL can draw from any config (incl. those claimed by teaser/rw_sw).

    # 3. DPO (5) — only deduplicate sample IDs across figures.
    dpo = pick(audit, "dpo", used_sids, set(), set(), k=5)
    for t in dpo:
        used_sids.add(t["sample_id"])

    # 4. RL (5) — same policy.
    rl = pick(audit, "rl", used_sids, set(), set(), k=5)
    for t in rl:
        used_sids.add(t["sample_id"])

    out = OrderedDict([
        ("teaser", teaser),
        ("rw_sw", rw_sw),
        ("dpo", dpo),
        ("rl", rl),
    ])

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {OUT}")

    for fig, items in out.items():
        print(f"\n=== {fig} ({len(items)} picks) ===")
        for c in items:
            print(f"  {c['cfg']:<20}  Δ={c['margin']:+.2f}  pick={c.get('method_picked','-'):<3}  {c['sample_id']}")


if __name__ == "__main__":
    main()
