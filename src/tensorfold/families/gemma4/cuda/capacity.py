"""Header-only admission for original Gemma affine/NVFP4 weights and DFlash (BF16 or private FP8 copies).

The serving grant uses the shared CUDA allocator policy: on a unified GPU,
system MemAvailable and its configured host floor govern the single pool.
"""

from __future__ import annotations

import math
from pathlib import Path


def admit(model_dir, context, context_explicit, *, drafter_dir=None, vision=False,
          max_rows=16, streams=1, keep=1, drafter_bits=16):
    import torch
    from tensorfold.checkpoint_metadata import read_config
    from tensorfold.families import quant_method
    from tensorfold.cuda.capacity import Geometry, Weights, admit as common, headers
    from .state import (GLOBAL_CHUNK, SEQUENTIAL_ROWS, capacity_bytes, companion_rings, sliding_slots,
                        widest_forward)
    from .weights import Config, tensor_contract, validate_tensor_metadata

    raw = read_config(model_dir)
    entries = headers(model_dir)
    media_geometry = None
    native = quant_method(raw) == "compressed-tensors"
    if native:
        from .nvfp4_weights import CT_PARTS, ct_tensor_contract, validate_headers
        cfg = validate_headers(raw, entries, vision=vision)
        matrices, plain = ct_tensor_contract(cfg)
        names = set(plain)
        names.update(name + '.' + suffix for name in matrices for suffix in CT_PARTS)
        if vision:
            names.update(name for name in entries if name.startswith(('model.vision_tower.', 'model.embed_vision.')))
    else:
        cfg = Config.from_dict(raw)
        where = {name: (None, None, None, info['dtype'], info['shape']) for name, info in entries.items()}
        validate_tensor_metadata(cfg, raw, where)
        matrices, plain = tensor_contract(cfg)
        names = set(plain)
        for name in matrices:
            names.update(name + suffix for suffix in ('.weight', '.scales', '.biases'))
        if vision:
            from .vision import projector_spec, vision_shapes, vision_weight_bytes
            vision_weight_bytes(raw, entries)
            projector_spec(raw, entries)
            names.update(vision_shapes(raw))
            names.update('embed_vision.embedding_projection' + suffix
                         for suffix in ('.weight', '.scales', '.biases'))
    if vision:
        from .vision import processor_geometry
        media_geometry = processor_geometry(model_dir)

    from tensorfold.cuda.capacity import itemsize
    sizes = {name: math.prod(entries[name]['shape']) * itemsize(entries[name], name) for name in names}
    # Dense Q4 output rows are padded to 128 in the native lossless layout.
    padding = 0
    for name in (() if native else names):
        info = entries[name]
        if name.endswith('.weight') and info['dtype'] in ('U32', 'I32') and len(info['shape']) == 2:
            from .weights import quant_spec
            bits, _ = quant_spec(raw, name[:-7])
            if bits == 4:
                rows = info['shape'][0]
                group_bytes = sum(sizes[name[:-7] + suffix] for suffix in ('.weight', '.scales', '.biases'))
                padding += (-(rows) % 128) * (group_bytes // rows)
    resident = sum(sizes.values()) + padding
    staging = 3 * max(sizes.values(), default=0)

    draft = None
    if drafter_dir is not None:
        from ..dflash_metadata import checked_headers
        draft, draft_entries = checked_headers(Path(drafter_dir))
        if (draft.hidden, draft.target_layers, draft.vocab) != (cfg.hidden, cfg.layers, cfg.vocab):
            raise ValueError('Gemma DFlash companion dimensions disagree with the target')
        bf16 = {name: math.prod(info['shape']) * itemsize(info, name) for name, info in draft_entries.items()}
        staging = max(staging, 3 * max(bf16.values()))
        if drafter_bits == 8:
            # Private FP8 copies (tensorfold.cuda.requant) of the fused projections ([k|v|q], [gate|up]; norms stay
            # BF16) and of the target's tied head for proposals. Loading holds every BF16 source on the device
            # until its copy exists: staging covers them plus the largest copy's own transient.
            from tensorfold.cuda.requant import resident_bytes, staging_bytes
            shapes = list(draft.fused_shapes().values())
            matrices = {name for name, info in draft_entries.items() if len(info['shape']) == 2}
            resident += sum(size for name, size in bf16.items() if name not in matrices)
            resident += sum(resident_bytes('fp8', n, k) for n, k in shapes)
            resident += resident_bytes('fp8', cfg.vocab, cfg.hidden)
            staging = max(staging, sum(bf16.values())
                          + max(staging_bytes('fp8', n, k) for n, k in [*shapes, (cfg.vocab, cfg.hidden)]))
        else:
            resident += sum(bf16.values())

    longest = max(media_geometry.image_soft_tokens, media_geometry.video_soft_tokens) if vision else 0
    rows = widest_forward(max_rows, longest)

    def workspace(slots):
        # One layer's simultaneous activation/route tensors and retained taps.
        scratch = rows * (32 * cfg.hidden + 16 * cfg.top_k * cfg.moe_width) * 2
        scratch += max_rows * cfg.vocab * 4
        # Attention chunk partials (FP32 o plus m and l): every sliding forward,
        # and full attention below the sequential row count.
        group = cfg.heads // cfg.kv_heads
        scratch += rows * cfg.heads * sliding_slots(cfg.window, rows, group, longest) * (cfg.head_dim + 2) * 4
        chunks = -(-min(slots, cfg.max_position) // GLOBAL_CHUNK)
        scratch += (SEQUENTIAL_ROWS - 1) * cfg.heads * chunks * (cfg.global_head_dim + 2) * 4
        if vision:
            v = raw['vision_config']
            patches = max(media_geometry.image_patches, media_geometry.video_patches)
            scratch += 3 * v['num_attention_heads'] * patches**2 * 4
            scratch += 16 * patches * (v['hidden_size'] + v['intermediate_size']) * 4
        return scratch

    def geometry(slots):
        target = capacity_bytes(cfg, min(slots, cfg.max_position), rows)
        companion = 0
        if draft is not None:
            # DraftState K and V (the engine's proposal block is at most max_rows).
            companion = sum(companion_rings(draft.layer_types, draft.sliding_window, max_rows, rows,
                                            min(slots, cfg.max_position))) * draft.kv_heads * draft.head_dim * 4
        return (streams + keep) * (target + companion) + streams * workspace(slots)

    return common(model_dir, context, context_explicit, torch, Geometry(geometry, 0),
                  lambda name, info: (0, 0),
                  weight_estimator=lambda _: (Weights(resident, staging), staging))
