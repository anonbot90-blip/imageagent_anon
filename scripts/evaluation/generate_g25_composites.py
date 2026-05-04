#!/usr/bin/env python3
"""Generate 9-way composite comparison images with Gemini 2.5 as the 9th column.

Reads figure_selections.json and builds:
  latex/img/teaser_g25/<slug>.jpg                    (4 teaser picks)
  latex/img/appendix/rw_sw_g25/<slug>.jpg            (5 rw_sw picks)
  latex/img/appendix/dpo_g25/<slug>.jpg              (5 dpo picks)
  latex/img/appendix/rl_g25/<slug>.jpg               (5 rl picks)

Columns: Original, Baseline, Edit-Only, Standard, RL, DPO, RW, SW, Gemini 2.5.
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = ROOT / "scripts/evaluation/figure_selections.json"
EVAL_ROOT = ROOT / "evaluation_results"
IMG_ROOT = ROOT / "latex/img"

# cfg label -> (dataset, config dir)
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

LABELS = ["Original", "Baseline", "Edit-Only", "Standard",
          "RL", "DPO", "RW", "SW", "Gemini 2.5"]
METHOD_DIRS = ["baseline", "edit_only", "standard_text",
               "rl_text", "dpo_text", "rw_text", "sw_text", "gemini25"]

TARGET_H = 512
GAP = 20
LABEL_H = 85
PROMPT_H = 240
MAX_CHARS_PER_LINE = 90
MAX_PROMPT_LINES = 4


def load_fonts():
    try:
        fl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        fp = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    except Exception:
        fl = ImageFont.load_default()
        fp = ImageFont.load_default()
    return fl, fp


def read_prompt(sample_dir: Path) -> str:
    """Prefer overall_instruction from predicted_plan.json; fall back to edit_prompt.txt."""
    plan = sample_dir / "predicted_plan.json"
    if plan.exists():
        try:
            d = json.loads(plan.read_text())
            oi = d.get("overall_instruction")
            if oi:
                return oi.strip()
        except Exception:
            pass
    ep = sample_dir / "edit_prompt.txt"
    if ep.exists():
        txt = ep.read_text().strip()
        # strip "Editing Instruction:" prefix + method token if present
        if txt.startswith("Editing Instruction:"):
            txt = txt[len("Editing Instruction:"):].strip()
        # drop leading tool token (e.g., "style_transformation_mode ")
        if " " in txt:
            head, rest = txt.split(" ", 1)
            if head.endswith("_mode") or head.endswith("_transformation"):
                txt = rest
        return txt
    return ""


def wrap_lines(text: str):
    words = text.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        if cur_len + len(w) + 1 <= MAX_CHARS_PER_LINE:
            cur.append(w)
            cur_len += len(w) + 1
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return lines[:MAX_PROMPT_LINES]


def build_composite(cfg_label: str, sample_id: str, out_path: Path) -> bool:
    ds, cfg = CFG2DIR[cfg_label]
    cfg_dir = EVAL_ROOT / ds / cfg

    # resolve per-column image paths
    paths = [cfg_dir / "baseline" / "samples" / sample_id / "original.png"]
    for m in METHOD_DIRS:
        paths.append(cfg_dir / m / "samples" / sample_id / "predicted_edit.png")

    # load and resize
    imgs = []
    for p, lbl in zip(paths, LABELS):
        if p.exists():
            im = Image.open(p).convert("RGB")
            ar = im.width / im.height
            im = im.resize((int(TARGET_H * ar), TARGET_H), Image.LANCZOS)
            imgs.append(im)
        else:
            print(f"  MISSING {lbl}: {p}")
            ph = Image.new("RGB", (TARGET_H, TARGET_H), (200, 200, 200))
            d = ImageDraw.Draw(ph)
            d.text((20, 20), "Missing", fill=(100, 100, 100))
            imgs.append(ph)

    prompt = read_prompt(cfg_dir / "baseline" / "samples" / sample_id)

    total_w = sum(im.width for im in imgs) + GAP * (len(imgs) - 1)
    prompt_h = PROMPT_H if prompt else 20
    total_h = TARGET_H + LABEL_H + prompt_h
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fl, fp = load_fonts()

    x = 0
    for im, lbl in zip(imgs, LABELS):
        canvas.paste(im, (x, LABEL_H))
        bb = draw.textbbox((0, 0), lbl, font=fl)
        tw = bb[2] - bb[0]
        draw.text((x + (im.width - tw) // 2, 20), lbl, fill="black", font=fl)
        x += im.width + GAP

    if prompt:
        py = TARGET_H + LABEL_H + 15
        for i, line in enumerate(wrap_lines(prompt)):
            draw.text((15, py + i * 52), line, fill="black", font=fp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # save JPEG directly
    canvas.save(out_path, "JPEG", quality=92)
    return True


def slug_for(category: str, cfg_label: str, sample_id: str) -> str:
    # preserve semantic theme for easy LaTeX reference
    theme_tokens = sample_id.split("_")
    # drop image + hash
    if theme_tokens and theme_tokens[0] == "image":
        theme_tokens = theme_tokens[1:]
    if theme_tokens:
        theme_tokens = theme_tokens[1:]
    # drop v2 / lN / 4-digit
    while theme_tokens and (theme_tokens[0] == "v2" or
                            (theme_tokens[0].startswith("l") and theme_tokens[0][1:].isdigit()) or
                            theme_tokens[0].isdigit()):
        theme_tokens = theme_tokens[1:]
    # drop trailing dual/triple/multi/complex
    while theme_tokens and theme_tokens[-1] in {"multi", "complex", "dual", "triple"}:
        theme_tokens = theme_tokens[:-1]
    theme = "_".join(theme_tokens) if theme_tokens else "sample"
    return f"{category}_{theme}_{cfg_label}"


def main():
    data = json.loads(SELECTIONS.read_text())

    tasks = []
    for item in data["teaser"]:
        slug = slug_for("sw", item["cfg"], item["sample_id"])
        tasks.append(("teaser_g25", slug, item))
    for item in data["rw_sw"]:
        slug = slug_for("rw_sw", item["cfg"], item["sample_id"])
        tasks.append(("appendix/rw_sw_g25", slug, item))
    for item in data["dpo"]:
        slug = slug_for("dpo", item["cfg"], item["sample_id"])
        tasks.append(("appendix/dpo_g25", slug, item))
    for item in data["rl"]:
        slug = slug_for("rl", item["cfg"], item["sample_id"])
        tasks.append(("appendix/rl_g25", slug, item))

    manifest = []
    ok = 0
    for outdir, slug, item in tasks:
        out_path = IMG_ROOT / outdir / f"{slug}.jpg"
        print(f"→ {out_path.relative_to(ROOT)}  [{item['cfg']} / {item['sample_id']}]")
        try:
            build_composite(item["cfg"], item["sample_id"], out_path)
            ok += 1
            manifest.append({
                "slug": slug,
                "out_path": str(out_path.relative_to(ROOT)),
                "cfg": item["cfg"],
                "sample_id": item["sample_id"],
                "method_picked": item.get("method_picked"),
                "margin": item.get("margin"),
                "score": item.get("score"),
                "theme": item.get("theme"),
                "opp": item.get("opp"),
                "opp_score": item.get("opp_score"),
                "g25": item.get("g25"),
                "category": outdir.split("/")[-1],
            })
        except Exception as e:
            print(f"  FAILED: {e}")

    man_path = ROOT / "scripts/evaluation/g25_composite_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote manifest: {man_path}")
    print(f"Composites generated: {ok}/{len(tasks)}")
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
