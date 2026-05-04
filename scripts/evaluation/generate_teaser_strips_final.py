#!/usr/bin/env python
"""Generate 10-column teaser strips for the 9 feasible cells in
``evaluation_results_final/sample_mappings/``.

Columns (in order): Original, Baseline, Edit-Only, Standard, RL, DPO, RW, SW, GPT-4o, Gemini 2.5.

Output: ``latex/neurips2026/img/teaser_final/<winner>_<theme>_<cell>.jpg``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO / "evaluation_results"
MAPPINGS = REPO / "evaluation_results_final" / "sample_mappings"
OUT_DIR = REPO / "latex" / "neurips2026" / "img" / "teaser_final"

# cell -> (dataset_dir, modality, size)
# modality is 'text' or 'vision' (used for _text/_vision method suffixes + dir name)
CELLS = [
    ("simple_text_4b", "simple", "text", "4b"),
    ("normal_text_4b", "normal", "text", "4b"),
    ("normal_text_8b", "normal", "text", "8b"),
    ("normal_vision_4b", "normal", "vision", "4b"),
    ("normal_vision_8b", "normal", "vision", "8b"),
    ("complex_text_4b", "complex", "text", "4b"),
    ("complex_text_8b", "complex", "text", "8b"),
    ("complex_vision_4b", "complex", "vision", "4b"),
    ("complex_vision_8b", "complex", "vision", "8b"),
]

# Methods in column order, templated on modality suffix
METHOD_TEMPLATES = [
    "baseline",
    "edit_only",
    "standard_{mod}",
    "rl_{mod}",
    "dpo_{mod}",
    "rw_{mod}",
    "sw_{mod}",
    "gpt4o",
    "gemini25",
]


def trajectory_dir(dataset: str, modality: str, size: str) -> Path:
    return EVAL_ROOT / dataset / f"{modality}_parallel_cot_{size}_trajectory"


def methods_for(modality: str) -> list[str]:
    # All trajectory dirs use the "_text" method suffix (kept for historical
    # reasons even in vision trajectory dirs), so ignore `modality` here.
    return [m.format(mod="text") for m in METHOD_TEMPLATES]


def sample_has_all(traj: Path, methods: list[str], sid: str) -> bool:
    """A sample is usable iff every method has predicted_edit.png and at least
    one has original.png."""
    has_original = False
    for m in methods:
        p = traj / m / "samples" / sid / "predicted_edit.png"
        if not p.exists():
            return False
        if (traj / m / "samples" / sid / "original.png").exists():
            has_original = True
    return has_original


def pick_sample(best_json: Path, traj: Path, methods: list[str]) -> str | None:
    data = json.loads(best_json.read_text())
    for sid in data.get("sample_ids", []):
        if sample_has_all(traj, methods, sid):
            return sid
    return None


def load_original(traj: Path, methods: list[str], sid: str) -> Image.Image:
    for m in methods:
        p = traj / m / "samples" / sid / "original.png"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"no original.png for {sid}")


def compose_strip(images: list[Image.Image], target_height: int = 512) -> Image.Image:
    """Compose images side-by-side at target_height, preserving aspect, pad to square-ish cells."""
    # Use a square cell (target_height x target_height) so the final strip has
    # a clean 10:1-ish aspect and matches teaser_g25 proportions.
    cell = target_height
    rendered = []
    for img in images:
        ar = img.width / img.height
        if ar >= 1:
            w = cell
            h = int(cell / ar)
        else:
            h = cell
            w = int(cell * ar)
        thumb = img.resize((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (cell, cell), (255, 255, 255))
        canvas.paste(thumb, ((cell - w) // 2, (cell - h) // 2))
        rendered.append(canvas)
    total_w = cell * len(rendered)
    strip = Image.new("RGB", (total_w, cell), (255, 255, 255))
    for i, r in enumerate(rendered):
        strip.paste(r, (i * cell, 0))
    return strip


def theme_of(sample_id: str) -> str:
    """image_<hash>_<idx>_<theme...> -> <theme...>"""
    parts = sample_id.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return "sample"


def run_cell(cell: str, dataset: str, modality: str, size: str, best_per_cell: dict) -> Path | None:
    info = best_per_cell.get(cell)
    if info is None:
        print(f"[skip] {cell}: not in best_per_cell.json")
        return None
    subtarget = info.get("picked_subtarget") or info.get("target")
    winner_col = subtarget  # rw or sw
    traj = trajectory_dir(dataset, modality, size)
    methods = methods_for(modality)
    best_json = MAPPINGS / cell / "best.json"
    if not best_json.exists():
        print(f"[skip] {cell}: missing {best_json}")
        return None
    sid = pick_sample(best_json, traj, methods)
    if sid is None:
        print(f"[FAIL] {cell}: no sample has all 10 methods populated under {traj}")
        return None
    print(f"[pick] {cell}: {sid} (winner={winner_col})")
    imgs = [load_original(traj, methods, sid)]
    for m in methods:
        p = traj / m / "samples" / sid / "predicted_edit.png"
        imgs.append(Image.open(p).convert("RGB"))
    strip = compose_strip(imgs, target_height=512)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{winner_col}_{theme_of(sid)}_{cell}.jpg"
    strip.save(out, "JPEG", quality=90)
    print(f"[save] {out} ({strip.width}x{strip.height})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="restrict to these cell names")
    args = ap.parse_args()
    bpc_path = MAPPINGS / "best_per_cell.json"
    best_per_cell = json.loads(bpc_path.read_text())
    targets = [c for c in CELLS if not args.only or c[0] in args.only]
    results: list[Path] = []
    for cell, ds, mod, sz in targets:
        try:
            r = run_cell(cell, ds, mod, sz, best_per_cell)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[ERR ] {cell}: {e}", file=sys.stderr)
    print(f"\nWrote {len(results)} strips to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
