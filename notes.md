# DoQAP — running learning log

## Phase 1 — baseline (no pruning)

### Setup decisions
- **Model**: Qwen/Qwen2-VL-2B-Instruct, bf16, `attn_implementation="sdpa"`.
  - sdpa works and is fast (~700 ms vision, ~1.3 s prefill after warmup).
  - eager is ~4x slower (materializes the 4960x4960 vision attention). flash-attn2
    not installable on Windows.
  - Phases 2 & 4 need attention weights out of the model → will switch to eager
    (or a hook) there and eat the slowdown.
- **Resolution**: `max_pixels = 1280 * 28 * 28` → image tokens capped at 1280.
  - Forced by 8 GB VRAM: Qwen2-VL's ViT runs full attention over *every* patch
    (no windowing until Qwen2.5-VL), so a native ~2 MP scan = 20k patches OOMs.
  - **Consequence**: "100% keep" = this 1280-token image, NOT native resolution.
    Every pruning curve is measured against this, not against published
    Qwen2-VL DocVQA numbers. The keep-ratio comparison between methods is still
    internally valid — all methods prune down from the same starting image.
  - TODO: sanity-check baseline ANLS at 1024 vs 1280 once we have a number; if
    1280 is so low the model can't read the docs, there's no headroom to show
    pruning differences.
- **Prompt**: append "Answer using a single word or short phrase copied from the
  document. Do not write a full sentence." Without this the instruct model says
  "The invoice number is 100001." which ANLS scores as 0 despite being correct.
- **Metric**: ANLS, official normalization (lowercase + strip whitespace only,
  no punctuation removal). 0.5 threshold.
- **Subset**: 250 examples, seeded shuffle (seed=0), DocVQA validation split.
  Same 250 for every method in every phase.

### How Qwen2-VL turns an image into tokens (Phase 1 checkpoint concept)
1. Patchify into 14x14 px patches. 1400x1400 img → 100x100 = 10,000 patches.
2. ViT (32 layers) runs over ALL patches with full self-attention.
3. Spatial merge: each 2x2 block of patches → 1 token for the LLM. 10,000 → 2,500.
4. Dynamic resolution: image is pre-resized so total pixels ∈ [min_pixels,
   max_pixels] and divisible by 28. `max_pixels` is the knob that caps token count.
   `n_image_tokens = (grid_h * grid_w) / merge_size**2`, read from `image_grid_thw`.

### Diagnostics done
- GPU is healthy: 37 TFLOP/s bf16, holds that rate even at 8.5/8.5 GB used
  (so slowdowns are not shared-memory paging or bad Blackwell kernels).
- torch 2.11.0+cu128 stable (not nightly).
- `scripts/profile_vision.py` kept — A/Bs attention impls, profiles the ViT.

### Results
- _(pending: run `python baseline_eval.py --n 25` then the full run)_
