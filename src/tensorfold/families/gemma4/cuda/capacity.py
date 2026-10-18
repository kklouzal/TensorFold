"""Header-only admission for original Gemma affine/NVFP4 weights and BF16 DFlash.

The serving grant uses the shared CUDA allocator policy: on a unified GPU,
system MemAvailable and its configured host floor govern the single pool.
"""

from __future__ import annotations

import math
from pathlib import Path


GRAPH_SPAN = 16      # graphs.Graphs' smallest full-attention grid (chunks); larger grids double up to the context's


def graph_workspace(cfg, max_rows: int, slots: int, taps: int) -> int:
    """Retained bytes of graphs.Graphs: each (rows, grid, taps) graph's static outputs plus one activation bound.

    Outputs per row: FP32-bounded logits, the final hidden row and ``taps`` tap rows. The graphs share one private
    pool, so their transient activations need one forward's bound at ``max_rows``.
    """
    chunk = 512                                    # state.GLOBAL_CHUNK
    limit = -(-min(slots, cfg.max_position) // chunk)
    grids = len({min(limit, max(GRAPH_SPAN, 1 << k)) for k in range(limit.bit_length() + 1)})
    row = cfg.vocab * 4 + 2 * cfg.hidden * (1 + taps)
    outputs = grids * 2 * (max_rows * (max_rows + 1) // 2) * row
    return outputs + max_rows * ((32 * cfg.hidden + 16 * cfg.top_k * cfg.moe_width) * 2 + cfg.vocab * 4)


def expert_workspace(cfg, rows: int) -> int:
    """Retained bytes of nvfp4_experts.Workspace for forwards of at most ``rows`` rows.

    One PlanBuffers (picks, grouped plan, BF16 slot outputs) and the bank's
    ExpertsA4.run_plan scratch (pair codes/scales for two projections, down
    codes/scales, shared-input row plan) per power-of-two bucket.
    """
    total, bucket = 0, 16
    maximum = 1 << max(4, (rows - 1).bit_length())
    d, width, experts = cfg.hidden, cfg.moe_width, cfg.experts
    while bucket <= maximum:
        pairs = bucket * cfg.top_k
        wide = pairs > 1024
        total += 8 * pairs + 12 * (min(pairs, experts) + pairs // 16) + 8 + 2 * pairs * d
        total += 4 * (pairs if wide else 1) + 4 * (((pairs + 1023) // 1024) * experts if wide else 1)
        total += pairs * (d + d // 8 + width // 2 + width // 16) + 12 * bucket
        bucket *= 2
    return total


def admit(model_dir, context, context_explicit, *, drafter_dir=None, vision=False,
          max_rows=16, streams=1, keep=1):
    import torch
    from tensorfold.checkpoint_metadata import read_config
    from tensorfold.families import quant_method
    from tensorfold.cuda.capacity import Geometry, Weights, admit as common, headers
    from .state import capacity_bytes
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
        draft_size = sum(math.prod(info['shape']) * itemsize(info, name)
                         for name, info in draft_entries.items())
        resident += draft_size
        staging = max(staging, 3 * max(math.prod(info['shape']) * itemsize(info, name)
                                    for name, info in draft_entries.items()))

    def workspace(slots):
        rows = max(128, max_rows, media_geometry.image_soft_tokens if vision else 0,
                   media_geometry.video_soft_tokens if vision else 0)
        # One layer's simultaneous activation/route tensors and retained taps.
        scratch = rows * (32 * cfg.hidden + 16 * cfg.top_k * cfg.moe_width) * 2
        scratch += max_rows * cfg.vocab * 4
        if native:
            scratch += expert_workspace(cfg, rows)
        from .weights import TAP_LAYERS
        scratch += graph_workspace(cfg, max_rows, slots, len([i for i in TAP_LAYERS if i < cfg.layers]))
        # Growth copies preserve the old allocation until its replacement is ready.
        largest = max(heads * dim for heads, dim in
                      (cfg.attention_geometry(i) for i in range(cfg.layers)))
        scratch += (slots + 512) * largest * 4
        if vision:
            v = raw['vision_config']
            patches = max(media_geometry.image_patches, media_geometry.video_patches)
            scratch += 3 * v['num_attention_heads'] * patches**2 * 4
            scratch += 16 * patches * (v['hidden_size'] + v['intermediate_size']) * 4
        return scratch

    def geometry(slots):
        target = capacity_bytes(cfg, min(slots, cfg.max_position), max_rows)
        companion = 0
        if draft is not None:
            companion = sum((min(slots, draft.sliding_window - 1) if kind == 'sliding_attention' else slots)
                            * draft.kv_heads * draft.head_dim * 4 for kind in draft.layer_types)
            # Context projection/append may retain old and new K/V concurrently.
            companion *= 2
        return (streams + keep) * (target + companion) + streams * workspace(slots)

    return common(model_dir, context, context_explicit, torch, Geometry(geometry, 0),
                  lambda name, info: (0, 0),
                  weight_estimator=lambda _: (Weights(resident, staging), staging))
