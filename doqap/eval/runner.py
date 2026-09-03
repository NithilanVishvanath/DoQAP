"""The evaluation loop.

Deliberately dumb: iterate examples, ask the wrapper for an answer, score it,
collect rows. Baseline-shaped for now (one call to wrapper.answer per example);
Phase 2 will refactor this when pruning hooks actually need to plug in.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from doqap.data.docvqa import DocVQAExample
from doqap.eval.metrics import anls


def run_eval(wrapper, examples: list[DocVQAExample]) -> pd.DataFrame:
    """Run the wrapper over every example, return one row per example.

    A per-example exception is recorded as a wrong answer (anls=0) rather than
    aborting the run — an 8-minute sweep shouldn't die on a single bad decode,
    and an honest baseline counts a crash as a miss, not a skip.
    """
    rows = []
    n_errors = 0

    for ex in tqdm(examples, desc="eval"):
        try:
            r = wrapper.answer(ex.image, ex.question)
            pred, n_tok, latency = r.text, r.n_image_tokens, r.latency_s
        except Exception as e:  # noqa: BLE001 - want everything, logged below
            n_errors += 1
            pred, n_tok, latency = f"<ERROR: {type(e).__name__}: {e}>", 0, 0.0

        rows.append(
            {
                "question_id": ex.question_id,
                "question": ex.question,
                "pred": pred,
                "golds": " | ".join(ex.answers),  # flat string so the CSV round-trips
                "anls": anls(pred, ex.answers),
                "n_image_tokens": n_tok,
                "latency_s": latency,
            }
        )

    if n_errors:
        print(f"WARNING: {n_errors}/{len(examples)} examples errored (scored as wrong)")

    return pd.DataFrame(rows)
