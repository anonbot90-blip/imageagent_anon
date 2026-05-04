#!/bin/bash
# Combine per-method gemini_31fl summaries into experiment-level combined JSONs.
# Run after all `gem31fl_*` screens have finished (launch_judge_gemini31fl.sh).

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate img-agent

python - <<'PY'
import json
from pathlib import Path

ROOT = Path("./evaluation_results")
JUDGE_ID = "gemini_31fl"
METHODS = ["baseline", "edit_only", "standard_text", "rl_text", "rw_text",
           "sw_text", "dpo_text", "gpt4o", "gemini25"]

count_combined = 0
for ds in ["simple", "normal", "complex"]:
    ds_dir = ROOT / ds
    if not ds_dir.exists():
        continue
    for exp_dir in sorted(ds_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        combined = {}
        for m in METHODS:
            summary = exp_dir / m / f"judge_summary_{JUDGE_ID}.json"
            if summary.exists():
                try:
                    combined[m] = json.loads(summary.read_text())
                except Exception as e:
                    print(f"  ⚠️  {summary}: {e}")
        if combined:
            out = exp_dir / f"judge_combined_{JUDGE_ID}.json"
            out.write_text(json.dumps(combined, indent=2))
            print(f"✅ {out.relative_to(ROOT)}  ({len(combined)} methods)")
            count_combined += 1
print(f"\nTotal combined files written: {count_combined}")
PY
