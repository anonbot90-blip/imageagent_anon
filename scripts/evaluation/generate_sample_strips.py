#!/usr/bin/env python
"""Generate one 10-column teaser strip per sample ID for every feasible cell
under ``evaluation_results_final/sample_mappings/``.

Each strip has:
  - column labels on top (Original, Baseline, Edit-Only, Standard, RL, DPO,
    RW, SW, GPT-4o, Gemini 2.5)
  - image row (512x512 per cell, white-padded to square)
  - prompt text below (parsed from ``instruction.txt``, with the
    ``style_transformation_mode`` prefix stripped)

Output: ``evaluation_results_final/sample_mappings/<cell>/strips/<sample_id>.jpg``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO / "evaluation_results"
MAPPINGS = REPO / "evaluation_results_final" / "sample_mappings"

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

# (method_subdir, column_label)
METHODS: list[tuple[str, str]] = [
    ("baseline",      "Baseline"),
    ("edit_only",     "Edit-Only"),
    ("standard_text", "Standard"),
    ("rl_text",       "RL"),
    ("dpo_text",      "DPO"),
    ("rw_text",       "RW"),
    ("sw_text",       "SW"),
    ("gpt4o",         "GPT-4o"),
    ("gemini25",      "Gemini 2.5"),
]

COLUMN_LABELS = ["Original"] + [label for _, label in METHODS]

# Font setup (prefer DejaVu, fall back to default).
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_fonts(cell_size: int) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    """Return (label_font, prompt_font). ICML-style: 48pt labels, 42pt prompt,
    both bold. Sized proportionally to cell_size (== 512 matches ICML exactly)."""
    label_pt = max(20, cell_size * 48 // 512)   # 48 at 512
    prompt_pt = max(18, cell_size * 42 // 512)  # 42 at 512
    try:
        return (
            ImageFont.truetype(FONT_PATH_BOLD, label_pt),
            ImageFont.truetype(FONT_PATH_BOLD, prompt_pt),
        )
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def trajectory_dir(dataset: str, modality: str, size: str) -> Path:
    return EVAL_ROOT / dataset / f"{modality}_parallel_cot_{size}_trajectory"


def sample_has_all(traj: Path, sid: str) -> tuple[bool, str]:
    has_original = False
    for m, _ in METHODS:
        p = traj / m / "samples" / sid / "predicted_edit.png"
        if not p.exists():
            return False, f"missing {m}/predicted_edit.png"
        if (traj / m / "samples" / sid / "original.png").exists():
            has_original = True
    if not has_original:
        return False, "no original.png anywhere"
    return True, ""


def load_original(traj: Path, sid: str) -> Image.Image:
    for m, _ in METHODS:
        p = traj / m / "samples" / sid / "original.png"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(sid)


def read_instruction(traj: Path, sid: str) -> str:
    """Parse the styling instruction. Prefers ``predicted_plan.json`` →
    ``overall_instruction`` (human-readable form) like the ICML pipeline;
    falls back to ``instruction.txt`` with prefix/trailer stripping."""
    # Prefer the plan JSON (clean text, no style_transformation_mode prefix)
    plan_candidates = [
        traj / "baseline" / "samples" / sid / "predicted_plan.json",
        traj / "sw_text"  / "samples" / sid / "predicted_plan.json",
        traj / "rw_text"  / "samples" / sid / "predicted_plan.json",
        traj / "gpt4o"    / "samples" / sid / "predicted_plan.json",
        traj / "gemini25" / "samples" / sid / "predicted_plan.json",
    ]
    text = ""
    for p in plan_candidates:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                cand = d.get("overall_instruction") or d.get("user_prompt") or ""
                if cand.strip():
                    text = cand.strip()
                    break
            except Exception:
                pass
    # Fallback to instruction.txt
    if not text:
        txt_candidates = [
            traj / "baseline" / "samples" / sid / "instruction.txt",
            traj / "gemini25" / "samples" / sid / "instruction.txt",
            traj / "gpt4o"    / "samples" / sid / "instruction.txt",
            traj / "sw_text"  / "samples" / sid / "instruction.txt",
        ]
        for c in txt_candidates:
            if c.exists():
                text = c.read_text(errors="ignore").strip()
                if text:
                    break
    # Normalise: strip the "style_transformation_mode" token and the
    # ubiquitous "Maintain high quality..." trailer.
    text = re.sub(r"^\s*style_transformation_mode\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s*Maintain (?:high |photorealistic )?quality[^.]*\.?\s*$", ".", text)
    text = text.replace("\n", " ").strip()
    return text


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Greedy word-wrap to max_width px."""
    words = text.split()
    lines: list[str] = []
    line = ""
    for w in words:
        trial = (line + " " + w).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def compose_strip(
    images: list[Image.Image],
    labels: list[str],
    prompt: str,
    cell_size: int = 512,
) -> Image.Image:
    """ICML-style: 48pt bold labels on top, 512x512 cells with 20px gaps,
    42pt bold prompt (up to 4 wrapped lines) on the bottom."""
    n = len(images)
    label_h = 85        # ICML: matches 48pt label space
    gap = 20            # ICML: inter-image gap
    pad_top = 20
    pad_prompt = 15

    # Prepare square cells
    cells: list[Image.Image] = []
    for img in images:
        ar = img.width / img.height
        if ar >= 1:
            w = cell_size
            h = int(cell_size / ar)
        else:
            h = cell_size
            w = int(cell_size * ar)
        thumb = img.resize((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        canvas.paste(thumb, ((cell_size - w) // 2, (cell_size - h) // 2))
        cells.append(canvas)

    total_w = cell_size * n + gap * (n - 1)
    label_font, prompt_font = load_fonts(cell_size)

    # Compute prompt lines up-front so we know how tall the footer needs to be.
    tmp = Image.new("RGB", (10, 10))
    tmp_draw = ImageDraw.Draw(tmp)
    prompt_lines: list[str] = []
    line_h = 52  # ICML fixed leading (close to 42pt * 1.24)
    prompt_h = 20
    if prompt:
        max_prompt_w = total_w - 2 * pad_prompt
        prompt_lines = wrap_text(tmp_draw, prompt, prompt_font, max_prompt_w)
        prompt_lines = prompt_lines[:4]  # ICML: up to 4 lines
        prompt_h = pad_prompt + len(prompt_lines) * line_h + pad_prompt

    total_h = pad_top + label_h + cell_size + prompt_h
    out = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)

    # Column labels (centered above each cell, accounting for gaps)
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=label_font)
        tw = bbox[2] - bbox[0]
        col_x = i * (cell_size + gap)
        x = col_x + (cell_size - tw) // 2
        y = pad_top
        draw.text((x, y), label, fill=(0, 0, 0), font=label_font)

    # Image row
    row_y = pad_top + label_h
    for i, c in enumerate(cells):
        out.paste(c, (i * (cell_size + gap), row_y))

    # Prompt footer (bold, left-aligned)
    if prompt_lines:
        y = row_y + cell_size + pad_prompt
        for line in prompt_lines:
            draw.text((pad_prompt, y), line, fill=(0, 0, 0), font=prompt_font)
            y += line_h

    return out


def run_sample(traj: Path, sid: str, out_dir: Path, overwrite: bool) -> str:
    out = out_dir / f"{sid}.jpg"
    if not overwrite and out.exists():
        return "skip"
    ok, why = sample_has_all(traj, sid)
    if not ok:
        return f"missing:{why}"
    try:
        imgs = [load_original(traj, sid)]
        for m, _ in METHODS:
            imgs.append(
                Image.open(traj / m / "samples" / sid / "predicted_edit.png").convert("RGB")
            )
        prompt = read_instruction(traj, sid)
        strip = compose_strip(imgs, COLUMN_LABELS, prompt, cell_size=512)
        out.parent.mkdir(parents=True, exist_ok=True)
        strip.save(out, "JPEG", quality=88)
        return "ok"
    except Exception as e:
        return f"err:{e}"


def run_cell(cell: str, dataset: str, modality: str, size: str, overwrite: bool) -> dict:
    traj = trajectory_dir(dataset, modality, size)
    best_json = MAPPINGS / cell / "best.json"
    if not best_json.exists():
        return {"cell": cell, "status": "no_best_json"}
    data = json.loads(best_json.read_text())
    sample_ids = data.get("sample_ids", [])
    out_dir = MAPPINGS / cell / "strips"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "missing": 0, "err": 0}
    t0 = time.time()
    for i, sid in enumerate(sample_ids, 1):
        r = run_sample(traj, sid, out_dir, overwrite)
        if r == "ok":
            counts["ok"] += 1
        elif r == "skip":
            counts["skip"] += 1
        elif r.startswith("missing"):
            counts["missing"] += 1
        else:
            counts["err"] += 1
            print(f"  [{cell}] {sid}: {r}", file=sys.stderr)
        if i % 25 == 0 or i == len(sample_ids):
            dt = time.time() - t0
            print(f"  [{cell}] {i}/{len(sample_ids)}  ok={counts['ok']}  missing={counts['missing']}  err={counts['err']}  ({dt:.1f}s)")
    return {"cell": cell, "status": "done", "total": len(sample_ids), **counts, "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="restrict to these cell names")
    ap.add_argument("--skip-existing", action="store_true", help="do not overwrite existing strips")
    args = ap.parse_args()
    overwrite = not args.skip_existing
    targets = [c for c in CELLS if not args.only or c[0] in args.only]
    summary: list[dict] = []
    for cell, ds, mod, sz in targets:
        print(f"\n=== {cell} (dataset={ds}, modality={mod}, size={sz}) ===")
        summary.append(run_cell(cell, ds, mod, sz, overwrite))
    print("\n=== SUMMARY ===")
    for s in summary:
        print(s)
    written = sum(s.get("ok", 0) for s in summary)
    total = sum(s.get("total", 0) for s in summary)
    print(f"\nWrote {written} strips across {len(summary)} cells (pool of {total} sample IDs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
