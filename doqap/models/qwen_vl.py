"""Qwen2-VL-2B wrapper.

One job: take (image, question) -> (answer text, image-token count, latency).
The image-token count is the x-axis of every plot in this project, so it is
read straight from the processor's grid metadata rather than estimated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from PIL.Image import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# DocVQA's metric (ANLS) is edit distance to short human answers like "100001".
# Left to itself the instruct model replies "The invoice number is 100001.",
# which scores near zero despite being correct. This instruction pins the
# output format so the metric measures reading ability, not verbosity.
TERSE_INSTRUCTION = (
    "Answer using a single word or short phrase copied from the document. "
    "Do not write a full sentence."
)

# The 5060 has 8 GB. Qwen2-VL's vision encoder runs full self-attention over
# every patch of the image at once (no windowing until Qwen2.5-VL), so a native
# ~2 MP scan (=20k patches / 5k tokens) OOMs the encoder. Capping here to 1280
# tokens keeps the whole pipeline on-GPU with headroom for the attention-output
# hooks Phases 2 & 4 add. Consequence: "100% keep" is this 1280-token image,
# NOT native resolution — every pruning curve is measured against this, not
# against published Qwen2-VL DocVQA numbers. See notes.md.
MAX_IMAGE_TOKENS = 1280
MAX_PIXELS = MAX_IMAGE_TOKENS * 28 * 28  # 28 px = one post-merge token


@dataclass
class AnswerResult:
    text: str
    n_image_tokens: int
    latency_s: float


class QwenVL:
    def __init__(self, device: str = "cuda"):
        self.device = device
        # bfloat16: 2B params fit comfortably on the 5060; bf16 keeps the
        # dynamic range of fp32 (unlike fp16) so no loss-scaling worries.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            # sdpa is fast but does NOT expose attention weights. Phases 2 & 4
            # need per-head attention (query->image); switch to "eager" there.
            attn_implementation="sdpa",
        ).to(device).eval()

        # max_pixels cap (see MAX_PIXELS note) — forced by 8 GB VRAM. Every
        # method in every phase prunes down from this same starting image, so
        # the keep-ratio comparison stays internally valid.
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)

    @torch.inference_mode()
    def answer(self, image: Image, question: str, max_new_tokens: int = 32) -> AnswerResult:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}\n\n{TERSE_INSTRUCTION}"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt"
        ).to(self.device)

        # image_grid_thw = [[t, h, w]] in *patch* units (t=1 for a still image).
        # The LLM sees one token per 2x2 patch block (spatial merge), so the
        # token count is the patch count divided by merge_size**2.
        t, h, w = inputs["image_grid_thw"][0].tolist()
        merge = self.processor.image_processor.merge_size
        n_image_tokens = (t * h * w) // (merge * merge)

        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - start

        # generate() returns prompt + completion; keep only the new tokens.
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        return AnswerResult(text=text, n_image_tokens=n_image_tokens, latency_s=latency_s)


if __name__ == "__main__":
    # smoke test: python -m doqap.models.qwen_vl
    from doqap.data.docvqa import load_docvqa_subset

    vl = QwenVL()
    for ex in load_docvqa_subset(n=3):
        r = vl.answer(ex.image, ex.question)
        print(f"Q: {ex.question}")
        print(f"  pred: {r.text!r}  gold: {ex.answers}")
        print(f"  image_tokens={r.n_image_tokens}  latency={r.latency_s:.2f}s")
