"""Phase 1 entrypoint: Qwen2-VL-2B on a DocVQA subset, no pruning.

This is the reference point every later method is measured against.

    python baseline_eval.py --n 25     # quick latency check first
    python baseline_eval.py            # full 250-example run

Writes:
  results/baseline_docvqa.csv   per-example rows (inspect failures here)
  results/summary.csv           one appended row; the master table for the
                                final accuracy-vs-token-count plot
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from doqap.data.docvqa import load_docvqa_subset
from doqap.eval.runner import run_eval
from doqap.models.qwen_vl import QwenVL

RESULTS = Path("results")

# Locked schema — every phase appends a row with exactly these columns.
SUMMARY_COLS = [
    "pruning_method", "dataset", "keep_ratio", "n",
    "avg_image_tokens", "anls", "avg_latency_s", "timestamp",
]


def append_summary(row: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "summary.csv"
    df = pd.DataFrame([row], columns=SUMMARY_COLS)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="subset size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = load_docvqa_subset(n=args.n, seed=args.seed)
    wrapper = QwenVL()

    df = run_eval(wrapper, examples)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "baseline_docvqa.csv", index=False)

    summary = {
        "pruning_method": "none",
        "dataset": "docvqa",
        "keep_ratio": 1.0,
        "n": len(df),
        "avg_image_tokens": round(df["n_image_tokens"].mean(), 1),
        "anls": round(df["anls"].mean(), 4),
        "avg_latency_s": round(df["latency_s"].mean(), 3),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    append_summary(summary)

    print("\n=== Phase 1 baseline ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
