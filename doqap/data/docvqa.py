"""DocVQA subset loader.

This file knows about *data only* — no resizing, no tokenization, no model
concerns. It hands back a fixed, reproducible list of examples that every
pruning method in every later phase will be scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from PIL.Image import Image


@dataclass
class DocVQAExample:
    question_id: int
    question: str
    image: Image
    answers: list[str]  # multiple human-provided answers; ANLS scores vs the best match


def load_docvqa_subset(n: int = 250, seed: int = 0) -> list[DocVQAExample]:
    """Return a deterministic n-example slice of DocVQA validation.

    We shuffle-then-take rather than head-slice because the raw dataset is
    grouped by source document: a head slice would over-represent a few
    documents and question types. A seeded shuffle gives a representative
    mix, and pinning the seed means method A at 50% keep and method B at
    50% keep are compared on the *identical* 250 examples — otherwise the
    accuracy-vs-token-count curves aren't comparable.
    """
    # config "DocVQA" (the other config in this repo is InfographicVQA).
    # validation is the only split with public ground-truth answers.
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

    # datasets' own shuffle is a seeded permutation of the index — cheap,
    # doesn't materialise the images until we actually access rows below.
    ds = ds.shuffle(seed=seed).select(range(n))

    return [
        DocVQAExample(
            question_id=row["questionId"],
            question=row["question"],
            image=row["image"],  # already a PIL image, RGB
            answers=row["answers"],
        )
        for row in ds
    ]


if __name__ == "__main__":
    # quick smoke test: python -m doqap.data.docvqa
    xs = load_docvqa_subset(n=5)
    for x in xs:
        print(f"[{x.question_id}] {x.image.size}  Q: {x.question}")
        print(f"    answers: {x.answers}")
