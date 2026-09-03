"""ANLS — DocVQA's official metric.

Average Normalized Levenshtein Similarity: smooth partial credit for
near-miss answers, but a hard 0.5 floor so a genuinely wrong answer scores
zero rather than accumulating letter-overlap credit. That floor is what
makes ANLS an accuracy proxy and not just a string-distance average.
"""

from __future__ import annotations

import Levenshtein

ANLS_THRESHOLD = 0.5  # official DocVQA value; below this, score is 0


def _normalize(s: str) -> str:
    # Official DocVQA normalization: lowercase + strip surrounding whitespace.
    # Deliberately NOT stripping punctuation — keeps us comparable to the
    # literature. The terse-answer prompt is what keeps predictions clean.
    return s.lower().strip()


def anls(pred: str, golds: list[str]) -> float:
    """ANLS for a single question against its set of acceptable answers."""
    p = _normalize(pred)

    best = 0.0
    for gold in golds:
        g = _normalize(gold)

        if not p and not g:
            # both empty -> identical; avoid 0/0 in the ratio below
            sim = 1.0
        else:
            dist = Levenshtein.distance(p, g)
            sim = 1.0 - dist / max(len(p), len(g))

        best = max(best, sim)

    # threshold: only "at least half right" answers keep their score
    return best if best >= ANLS_THRESHOLD else 0.0


def mean_anls(preds: list[str], golds_list: list[list[str]]) -> float:
    """Dataset-level ANLS = mean of per-question scores."""
    if not preds:
        return 0.0
    scores = [anls(p, gs) for p, gs in zip(preds, golds_list, strict=True)]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    # sanity checks: python -m doqap.eval.metrics
    cases = [
        ("100001", ["100001"], 1.0),                       # exact
        ("100001.", ["100001"], 6 / 7),                    # trailing period dings it
        ("taken by mouth", ["TAKEN BY MOUTH", "Taken by mouth"], 1.0),  # case-insensitive
        ("the invoice number is 100001", ["100001"], 0.0), # verbose -> below threshold
        ("", ["something"], 0.0),                          # empty pred
        ("dog", ["cat"], 0.0),                             # wrong, sim=0.33 < 0.5
    ]
    for pred, golds, expected in cases:
        got = anls(pred, golds)
        ok = "OK " if abs(got - expected) < 1e-6 else "FAIL"
        print(f"{ok} anls({pred!r}, {golds}) = {got:.4f}  (expected {expected:.4f})")
