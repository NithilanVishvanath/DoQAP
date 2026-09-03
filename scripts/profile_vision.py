"""Profile ONE vision-encoder forward + ONE LLM prefill, print top CUDA ops,
and A/B the attention implementation. Run: python -m scripts.profile_vision"""
from __future__ import annotations

import time

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from doqap.data.docvqa import load_docvqa_subset
from doqap.models.qwen_vl import MODEL_ID, MAX_PIXELS, TERSE_INSTRUCTION


def build_inputs(processor):
    ex = load_docvqa_subset(n=1)[0]
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": f"{ex.question}\n\n{TERSE_INSTRUCTION}"}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=[ex.image], return_tensors="pt").to("cuda")


def run(attn_impl):
    print(f"\n===== attn_implementation = {attn_impl} =====")
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
    ).to("cuda").eval()

    # what did the vision tower actually pick up?
    vis = model.visual if hasattr(model, "visual") else model.model.visual
    print("vision config attn_implementation:",
          getattr(vis.config, "_attn_implementation", "?"))

    inputs = build_inputs(processor)
    pv = inputs["pixel_values"].to(next(vis.parameters()).dtype)
    grid = inputs["image_grid_thw"]
    n_patch = grid.prod().item()
    print(f"patches through ViT: {n_patch}")

    with torch.inference_mode():
        for _ in range(2):  # warmup
            vis(pv, grid_thw=grid)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        vis(pv, grid_thw=grid)
        torch.cuda.synchronize()
        print(f"vision encoder: {(time.perf_counter()-t0)*1e3:.1f} ms")

        t0 = time.perf_counter()
        model.generate(**inputs, do_sample=False, max_new_tokens=1)
        torch.cuda.synchronize()
        print(f"prefill (gen 1 tok): {(time.perf_counter()-t0)*1e3:.1f} ms")

        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            vis(pv, grid_thw=grid)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("torch", torch.__version__)
    for impl in ["sdpa", "eager", "flash_attention_2"]:
        try:
            run(impl)
        except Exception as e:
            print(f"{impl}: FAILED -> {type(e).__name__}: {e}")
