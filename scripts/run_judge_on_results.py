#!/usr/bin/env python3
"""
Retroactive multi-judge evaluation script.

Reads existing evaluation results (original.png + predicted_edit.png + instruction.txt
+ predicted_plan.json) and runs the specified judge on all samples.

Writes aggregate summary ONLY — does not touch existing result files.

Output per baseline dir:
  judge_summary_<judge_id>.json

Usage:
    # Judge a single experiment across all its baselines
    python scripts/run_judge_on_results.py \\
        --results-dir evaluation_results/simple/text_parallel_cot_4b_trajectory \\
        --judge gpt54

    # Judge all experiments under simple/normal/complex
    python scripts/run_judge_on_results.py \\
        --results-base evaluation_results \\
        --datasets simple normal complex \\
        --judge gemini

    # Skip if summary already exists
    python scripts/run_judge_on_results.py ... --skip-existing
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.evaluation.judge_registry import get_judge, get_action_judge, VALID_JUDGES

BASELINE_DIRS = ["baseline", "edit_only", "standard_text", "rl_text", "rw_text",
                 "dpo_text", "sw_text", "gpt4o", "gemini25"]


def find_sample_dirs(baseline_dir: Path) -> List[Path]:
    """Find all per-sample directories under a baseline dir."""
    samples_dir = baseline_dir / "samples"
    if samples_dir.exists():
        return sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    # Fallback: sample dirs directly in baseline_dir
    return sorted([d for d in baseline_dir.iterdir()
                   if d.is_dir() and (d / "original.png").exists()])


def load_sample(sample_dir: Path):
    """Load original image, predicted edit, instruction, and plan from a sample dir."""
    orig_path = sample_dir / "original.png"
    edit_path = sample_dir / "predicted_edit.png"
    instr_path = sample_dir / "instruction.txt"
    plan_path = sample_dir / "predicted_plan.json"

    if not orig_path.exists() or not edit_path.exists():
        return None

    original = Image.open(orig_path).convert("RGB")
    predicted = Image.open(edit_path).convert("RGB")

    instruction = ""
    if instr_path.exists():
        raw = instr_path.read_text().strip()
        # Strip mode prefix if present (e.g. "style_transformation_mode Transform to ...")
        parts = raw.split(" ", 1)
        if len(parts) == 2 and "_mode" in parts[0]:
            instruction = parts[1]
        else:
            instruction = raw

    plan = None
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text())
        except Exception:
            pass

    return {
        "original": original,
        "predicted": predicted,
        "instruction": instruction,
        "plan": plan,
        "sample_id": sample_dir.name,
    }


def run_judge_on_baseline(baseline_dir: Path, judge_id: str, skip_existing: bool = True,
                          rate_limit_delay: float = 0.3, output_as: str = None) -> Optional[Dict]:
    """
    Run judge on all samples in a baseline dir.
    Saves per-sample results incrementally to judge_samples_<output_as>.jsonl.
    Resumes from where it left off if interrupted.
    Returns aggregate summary dict, or None if skipped.

    output_as: if set, write files as this judge_id instead of judge_id
               (e.g. run proxy_claude but write to claude_opus files)
    """
    out_id = output_as or judge_id
    summary_file = baseline_dir / f"judge_summary_{out_id}.json"
    samples_file = baseline_dir / f"judge_samples_{out_id}.jsonl"

    if skip_existing and summary_file.exists():
        print(f"  ⏭️  Skipping {baseline_dir.name} — {summary_file.name} already exists")
        return None

    sample_dirs = find_sample_dirs(baseline_dir)
    if not sample_dirs:
        print(f"  ⚠️  No samples found in {baseline_dir}")
        return None

    # Load already-judged sample IDs for resume
    judged_ids = set()
    if samples_file.exists():
        with open(samples_file) as f:
            for line in f:
                try:
                    judged_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass

    pending = [d for d in sample_dirs if d.name not in judged_ids]

    if judged_ids:
        print(f"  ⏩ Resuming {baseline_dir.name}: {len(judged_ids)} done, {len(pending)} remaining")
    else:
        print(f"  📊 Judging {len(sample_dirs)} samples in {baseline_dir.name} with {judge_id}...")

    if pending:
        image_judge = get_judge(judge_id)
        action_judge = get_action_judge(judge_id)
        errors = 0

        with open(samples_file, "a") as f_out:
            for sample_dir in tqdm(pending, desc=f"  {baseline_dir.name}", leave=False):
                sample = load_sample(sample_dir)
                if sample is None:
                    errors += 1
                    continue

                entry = {"sample_id": sample_dir.name}

                # Image judge
                try:
                    img_scores = image_judge.judge_single_edit(
                        sample["original"], sample["predicted"], sample["instruction"]
                    )
                    if "error" not in img_scores:
                        entry["image_scores"] = img_scores
                except Exception as e:
                    print(f"    ⚠️  Image judge failed for {sample['sample_id']}: {e}")
                    errors += 1

                # Action plan judge (only if plan exists)
                if sample["plan"] is not None:
                    try:
                        act_scores = action_judge.judge_action_plan(
                            sample["original"], sample["instruction"], sample["plan"]
                        )
                        if "error" not in act_scores:
                            entry["action_scores"] = act_scores
                    except Exception as e:
                        print(f"    ⚠️  Action judge failed for {sample['sample_id']}: {e}")

                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()

                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)

    # Aggregate from complete jsonl (covers both resumed + newly judged)
    image_scores_all = []
    action_scores_all = []
    with open(samples_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "image_scores" in entry:
                    image_scores_all.append(entry["image_scores"])
                if "action_scores" in entry:
                    action_scores_all.append(entry["action_scores"])
            except Exception:
                pass

    def avg_scores(score_list, keys):
        if not score_list:
            return {}
        return {k: round(sum(s.get(k, 0) for s in score_list) / len(score_list), 4) for k in keys}

    img_keys = ["instruction_following", "visual_quality", "transformation_strength",
                "coherence", "semantic_accuracy", "technical_execution", "overall_image_score"]
    act_keys = ["relevance", "theme_style_focus", "completeness", "efficiency", "correctness",
                "reasoning_conciseness", "reasoning_completeness", "reasoning_specificity",
                "overall_action_quality", "overall_reasoning_quality", "overall_score"]

    summary = {
        "judge": out_id,
        "baseline": baseline_dir.name,
        "n_samples": len(sample_dirs),
        "n_image_judged": len(image_scores_all),
        "n_action_judged": len(action_scores_all),
        "n_errors": errors if pending else 0,
        "image_scores": avg_scores(image_scores_all, img_keys),
        "action_scores": avg_scores(action_scores_all, act_keys),
    }

    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"  ✅ Written: {summary_file}")
    print(f"     Image overall: {summary['image_scores'].get('overall_image_score', 'N/A'):.3f}  "
          f"Action overall: {summary['action_scores'].get('overall_score', 'N/A')}")

    return summary


def run_judge_on_experiment(experiment_dir: Path, judge_id: str, skip_existing: bool,
                             rate_limit_delay: float, output_as: str = None,
                             method: str = None, skip_combined: bool = False):
    """Run judge on all baseline subdirs of an experiment dir.

    method: if set, only run that one baseline subdir (for per-method parallelism).
    skip_combined: if True, skip writing judge_combined_<id>.json (avoids race
                   conditions when multiple method runs execute in parallel).
    """
    out_id = output_as or judge_id
    print(f"\n🔬 Experiment: {experiment_dir.name}")
    targets = [method] if method else BASELINE_DIRS
    results = {}
    for bname in targets:
        bdir = experiment_dir / bname
        if bdir.exists():
            result = run_judge_on_baseline(bdir, judge_id, skip_existing, rate_limit_delay, output_as)
            if result:
                results[bname] = result

    # Write combined summary — merge with existing to preserve prior runs
    if results and not skip_combined:
        combined_file = experiment_dir / f"judge_combined_{out_id}.json"
        existing = {}
        if combined_file.exists():
            try:
                existing = json.loads(combined_file.read_text())
            except Exception:
                pass
        existing.update(results)
        combined_file.write_text(json.dumps(existing, indent=2))
        print(f"\n📋 Combined summary: {combined_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Retroactive judge on existing evaluation results")
    parser.add_argument("--judge", required=True, choices=VALID_JUDGES,
                        help="Judge model to use")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Single experiment dir (e.g. evaluation_results/simple/text_parallel_cot_4b_trajectory)")
    parser.add_argument("--results-base", type=str, default=None,
                        help="Base results dir — combine with --datasets to scan multiple")
    parser.add_argument("--datasets", nargs="+", default=["simple", "normal", "complex"],
                        help="Dataset subdirs to scan under --results-base (default: simple normal complex)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip if judge_summary_<id>.json already exists (default: True)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-run even if summary already exists")
    parser.add_argument("--rate-limit-delay", type=float, default=0.3,
                        help="Seconds between API calls (default: 0.3)")
    parser.add_argument("--output-as", type=str, default=None,
                        help="Write output files under this judge ID instead of --judge "
                             "(e.g. --judge proxy_claude --output-as claude_opus)")
    parser.add_argument("--method", type=str, default=None, choices=BASELINE_DIRS,
                        help="Only run this one baseline subdir (for per-method parallelism)")
    parser.add_argument("--skip-combined", action="store_true",
                        help="Skip writing judge_combined_<id>.json (use when running "
                             "multiple --method screens in parallel; run one final combine pass after)")
    args = parser.parse_args()

    project_root_path = Path(__file__).parent.parent

    if args.results_dir:
        experiment_dir = Path(args.results_dir)
        if not experiment_dir.is_absolute():
            experiment_dir = project_root_path / experiment_dir
        run_judge_on_experiment(experiment_dir, args.judge, args.skip_existing,
                                args.rate_limit_delay, args.output_as,
                                method=args.method, skip_combined=args.skip_combined)

    elif args.results_base:
        base = Path(args.results_base)
        if not base.is_absolute():
            base = project_root_path / base

        for dataset in args.datasets:
            dataset_dir = base / dataset
            if not dataset_dir.exists():
                print(f"⚠️  Dataset dir not found: {dataset_dir}")
                continue
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}")
            print(f"{'='*60}")
            for exp_dir in sorted(dataset_dir.iterdir()):
                if exp_dir.is_dir():
                    run_judge_on_experiment(exp_dir, args.judge, args.skip_existing,
                                            args.rate_limit_delay, args.output_as,
                                            method=args.method, skip_combined=args.skip_combined)
    else:
        parser.error("Provide either --results-dir or --results-base")


if __name__ == "__main__":
    main()
