# Gemma 4 dense 31B NVFP4-AWQ: development checkpoint

Target: [AEON standard 31B NVFP4](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4)
at revision `c9f11f3cbeb8efc93d5afcb5dc1b33abee1f09ba`.
The profile has 60 layers, hidden size 5,376, intermediate size 21,504,
32 query heads, 16 local and four global KV heads, head dimensions 256/512,
and a 1,024-token sliding window. Global attention uses raw K for V, with
separate normalization. There is no PLE or shared KV. Native context is
262,144 tokens. The native vision encoder has width 1,152 and 27 layers;
visual attention blocks are bidirectional. The model has no audio encoder.
This deployment uses the main model alone, with drafting disabled.

Source reuses the dense engine, public HF model/mask/RoPE components and packed
projection operators. Publisher identities, complete headers, streamed payloads,
AWQ input scaling and native buffer initialization remain checked. META model
construction avoids allocating a full BF16 parameter mirror.

The broad CPU integration gate passed 1,331 checks with 39 optional/backend skips.
The final publisher-authority correction passed 62 focused checks, and the
MATH-SDPA allocation-bound correction passed 111. Pinned HF 5.17 constructor
warnings about removal of the `device` argument in 5.18 remain visible.
The first installed-image GPU load rejected a nonfinite FP8 scale in
`model.language_model.layers.10.mlp.down_proj.weight_scale`. The file matches
the pinned publisher SHA256. The publisher's patched loader zeros NaN scale
bytes, discarding the corresponding 16-weight blocks. That workaround is not
enabled here, and its quality impact has not been established. Full loading,
numerical correctness, quality, media, capacity and installed HTTP qualification
remain blocked or pending. The candidate retired without OOM; production
remains healthy at its existing 131,072-token context.

The initial diagnostic command, following the candidate image's protected
CUDA entrypoint and allocator adapter, is:

```sh
serve /models/hf/hub/models--AEON-7--Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4/snapshots/c9f11f3cbeb8efc93d5afcb5dc1b33abee1f09ba \
  --backend cuda --precision checkpoint --no-drafts --parallel 1 \
  --name gemma4-31b --host 127.0.0.1 --port 8889 --context 8192 \
  --kv-dtype bf16 --vision --thinking --thinking-budget 0
```

Use a new whole-wheel image containing this source. Bind `/models/hf` read-only
and a separate writable kernel cache. The initial test uses a 40 GiB Torch
allocator ceiling, matching `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=40`, with normal
capacity admission and `TENSORFOLD_MEMORY_RESERVE_GIB=16`. The estimate for its
8,192-token serving owners is about 30.09 GiB; actual runtime use remains to be
measured. The serving process uses the existing sealed SDK.

The 8,192-token diagnostic does not qualify the requested 131,072-token
production context. Longer contexts require actual capacity evidence, including
growing KV, retained prefix snapshots and prompt-attention workspace. Native
262,144-token metadata is retained. Existing E4B production and rollback artifacts
remain available during validation. No 31B performance winner is claimed.
