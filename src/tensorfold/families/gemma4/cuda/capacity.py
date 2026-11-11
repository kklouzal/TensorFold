"""Header-only admission for original Gemma affine/NVFP4 weights and DFlash (BF16 or private FP8 copies).

The serving grant uses the shared CUDA allocator policy: on a unified GPU,
system MemAvailable and its configured host floor govern the single pool.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


GRAPH_SPAN = 16      # graphs.Graphs' smallest full-attention grid (chunks); larger grids double up to the context's
DFLASH_FULL_CHUNK = 1024


def check_dflash_graph(enabled, bits, rows, kind):
    if type(enabled) is not bool:
        raise ValueError('dflash_graph must be a bool')
    if enabled and (kind != 'dflash' or type(bits) is not int or bits != 8
                    or type(rows) is not int or not 2 <= rows <= 16):
        raise ValueError('dflash_graph requires an enabled FP8 DFlash block of2..16 rows')



def check_dflash_head_a8(enabled, bits, rows, kind, dflash_graph=False):
    if type(enabled) is not bool:
        raise ValueError('dflash_head_native_a8 must be a bool')
    if enabled and (kind != 'dflash' or type(bits) is not int or bits != 8
                    or type(rows) is not int or rows != 5 or dflash_graph):
        raise ValueError('native A8 proposal head requires FP8 DFlash depth4 with dflash_graph disabled')


def check_dflash_head_w4a16(enabled, bits, kind, dflash_graph=False, native_a8=False):
    if type(enabled) is not bool:
        raise ValueError('dflash_head_w4a16 must be a bool')
    if enabled and (kind != 'dflash' or type(bits) is not int or bits != 8 or dflash_graph or native_a8):
        raise ValueError('W4A16 private head requires FP8 DFlash without draft graph/native A8')


def dflash_head_w4a16_memory(hidden: int, vocab: int, rows: int) -> dict[str, int]:
    """Separate CPU quantization and GPU upload/packing owners; no source alias credit."""
    from tensorfold.cuda.requant import NVFP4_CHUNK_ROWS, resident_bytes, staging_bytes
    from tensorfold.cuda.direct_read import ALIGN
    if any(type(value) is not int or value < 1 for value in (hidden, vocab, rows)) or hidden % 64:
        raise ValueError('W4A16 head requires positive dimensions and hidden multiple64')
    resident = resident_bytes('nvfp4', vocab, hidden)
    cpu_fields = vocab * hidden // 2 + vocab * hidden // 16 + 4
    source = 2 * vocab * hidden
    # Reader._to_host may hold aligned backing plus a full clone for an
    # arbitrary valid offset. The backing is < source+3ALIGN; no offset restriction.
    read_peak = 2 * source + 3 * ALIGN
    cpu_peak = source + cpu_fields + min(vocab, NVFP4_CHUNK_ROWS) * hidden * 19
    host_peak = max(read_peak, cpu_peak)
    # Existing staging bound contains the final packed destination: subtract it
    # once when adding to final resident weights. CPU source/temps retire first.
    gpu_peak = staging_bytes('nvfp4', vocab, hidden)
    return {'resident_bytes': resident, 'cpu_read_peak_bytes': read_peak,
            'cpu_quantization_peak_bytes': cpu_peak, 'host_loading_peak_bytes': host_peak,
            'cpu_upload_bytes': cpu_fields, 'gpu_packing_peak_bytes': gpu_peak,
            'device_extra_bytes': max(0, gpu_peak - resident),
            'unified_extra_bytes': max(0, host_peak - resident, cpu_fields + gpu_peak - resident),
            'logits_softcap_bytes': 3 * rows * vocab * 4}


def dflash_head_a8_workspace(hidden: int, vocab: int, rows: int) -> dict[str, int]:
    """Per-call eager owners, without allocator reuse credit.

    Head output and two live softcap intermediates coexist. Private proposal
    logits/quantizer owners retire after readback, before target verification;
    their arithmetic workspace takes a maximum with target transient work.
    """
    if any(type(value) is not int or value < 1 for value in (hidden, vocab, rows)):
        raise ValueError('native A8 head workspace requires positive integer dimensions')
    return {'activation_bytes': rows * (3 * hidden + 4),
            'logits_softcap_bytes': 3 * rows * vocab * 4}


def dflash_graph_workspace(cfg, rows: int, slots: int) -> int:
    """One replaceable logits graph's conservative tensor-payload estimate.

    Sum all allocation requests without transient reuse credit, including the
    fresh owning return. This is not physical graph-pool/allocator residency.
    Native FP8 lane projections have no activation quantizer or expert plan;
    optional noncluster split-K scratch is charged even on cluster-capable GPUs.
    """
    from .state import sliding_slots

    check_dflash_graph(True, 8, rows, 'dflash')
    if type(slots) is not int or slots < 1:
        raise ValueError('DFlash graph context must be a positive integer')
    length = min(rows, slots)
    total = rows * 4  # fixed owner IDs
    if length < 2:
        return total
    total += length * cfg.hidden * 2 + length * 8  # embedding and block start/end
    q = cfg.heads * cfg.head_dim
    kvq = (2 * cfg.kv_heads + cfg.heads) * cfg.head_dim

    def split_scratch(n):
        # qmm.split_k stops immediately once ceil(N/64)>=192. Otherwise
        # charge its largest eight-way scratch, without relying on clusters.
        return 0 if n >= 192 * 64 else 8 * length * n * 4

    for kind, causal in zip(cfg.layer_types, cfg.causal_layers):
        splits = (sliding_slots(cfg.sliding_window, length, cfg.heads // cfg.kv_heads,
                                0 if causal else length) if kind == 'sliding_attention'
                  else -(-slots // DFLASH_FULL_CHUNK))
        total += length * (6 * cfg.hidden + kvq + 2 * q + 4 * cfg.width) * 2
        total += length * cfg.heads * splits * (cfg.head_dim + 2) * 4
        total += sum(split_scratch(n) for n in (kvq, cfg.hidden, 2 * cfg.width, cfg.hidden))
    total += (length - 1) * cfg.hidden * 2
    total += 5 * (length - 1) * cfg.vocab * 4  # head, divide/tanh/multiply, owning clone
    total += split_scratch(cfg.vocab)  # overbounds the head's length-1 rows
    return total


def graph_workspace(cfg, max_rows: int, slots: int, taps: int) -> int:
    """Retained bytes of graphs.Graphs: each (rows, grid, taps) graph's static outputs plus one activation bound.

    Outputs per row: FP32-bounded logits, the final hidden row and ``taps`` tap rows. The graphs share one private
    pool, so their transient activations need one forward's bound at ``max_rows``.
    """
    from .state import GLOBAL_CHUNK

    limit = -(-min(slots, cfg.max_position) // GLOBAL_CHUNK)
    grids = len({min(limit, max(GRAPH_SPAN, 1 << k)) for k in range(limit.bit_length() + 1)})
    row = cfg.vocab * 4 + 2 * cfg.hidden * (1 + taps)
    outputs = grids * 2 * (max_rows * (max_rows + 1) // 2) * row
    return outputs + max_rows * ((32 * cfg.hidden + 16 * cfg.top_k * cfg.moe_width) * 2 + cfg.vocab * 4)


def expert_workspace(cfg, rows: int) -> int:
    """Retained bytes of nvfp4_experts.Workspace for forwards of at most ``rows`` rows.

    One PlanBuffers (picks, grouped plan, BF16 slot outputs) and the bank's
    ExpertsA4.run_plan scratch (plan_scratch_bytes_a4) and shared-input
    [rows, 3] row plan per power-of-two bucket.
    """
    from tensorfold.cuda.nvfp4.host_layout import plan_scratch_bytes_a4

    total, bucket = 0, 16
    maximum = 1 << max(4, (rows - 1).bit_length())
    d, width, experts = cfg.hidden, cfg.moe_width, cfg.experts
    while bucket <= maximum:
        pairs = bucket * cfg.top_k
        wide = pairs > 1024
        total += 8 * pairs + 12 * (min(pairs, experts) + pairs // 16) + 8 + 2 * pairs * d
        total += 4 * (pairs if wide else 1) + 4 * (((pairs + 1023) // 1024) * experts if wide else 1)
        total += plan_scratch_bytes_a4(pairs, width, d) + 12 * bucket
        bucket *= 2
    return total



def companion_kind(directory):
    """Select a declared companion family; each reader validates its schema."""
    from tensorfold.checkpoint_metadata import read_config
    raw = read_config(directory)
    if raw.get('model_type') == 'gemma4_assistant' or raw.get('architectures') == ['Gemma4AssistantForCausalLM']:
        return 'sparkulator'
    return 'dflash'  # existing strict DFlash metadata reader refuses other schemas



def archive_retention_charge(model_dir, drafter_dir=None):
    """Price the actual profile's group names without importing Torch.

    The authoritative Core price depends on names/end-group, not window
    extents. Unit extent1 represents a present window for pricing only; the
    Archive configures the actual allocated ring extents once constructed.
    """
    from tensorfold.checkpoint_metadata import read_config
    from tensorfold.engine.prefix_blocks import retention_config_charge
    from .weights import Config
    cfg = Config.from_dict(read_config(model_dir))
    windows = {}
    if any(not cfg.sliding(i) for i in range(cfg.layers)):
        windows['target_full'] = None
    if any(cfg.sliding(i) for i in range(cfg.layers)):
        windows['target_window'] = 1
    if drafter_dir is not None:
        if companion_kind(drafter_dir) == 'sparkulator':
            windows['hidden'] = 1
        else:
            from ..dflash_metadata import DraftConfig
            draft = DraftConfig.from_dict(read_config(drafter_dir))
            if 'full_attention' in draft.layer_types:
                windows['draft_full'] = None
            if 'sliding_attention' in draft.layer_types:
                windows['draft_window'] = 1
    return retention_config_charge(windows, require_end_group='hidden')


def admit(model_dir, context, context_explicit, *, drafter_dir=None, vision=False,
          max_rows=16, streams=1, keep=1, drafter_bits=16, prepare_global_prompt=False, lossless_head=False, spark_global_gqa=False, spark_sampled_q=False, global_trig=False, delta4_head=False, global_cache="none", global_cache_prompt=False, dflash_graph=False, dflash_head_native_a8=False, dflash_head_w4a16=False, prefix_cache_bytes=None):
    if prefix_cache_bytes is not None and (type(prefix_cache_bytes) is not int or not 0 <= prefix_cache_bytes <= sys.maxsize):
        raise ValueError("prefix_cache_bytes must be native nonnegative bytes within the host size range")
    if prefix_cache_bytes is not None and prefix_cache_bytes > 0:
        from tensorfold.engine.prefix_blocks import validate_prefix_budget
        validate_prefix_budget(prefix_cache_bytes)
        validate_prefix_budget(prefix_cache_bytes, retention_bytes=archive_retention_charge(model_dir, drafter_dir))
    import torch
    from tensorfold.checkpoint_metadata import read_config
    from tensorfold.families import quant_method
    from tensorfold.cuda.capacity import Geometry, Weights, admit as common, headers
    from .state import (GLOBAL_CHUNK, SEQUENTIAL_ROWS, capacity_bytes, check_global_cache, companion_rings, derived_capacity_bytes, sliding_slots,
                        widest_forward)
    from .weights import Config, tensor_contract, validate_tensor_metadata

    check_global_cache(global_cache, prompt=global_cache_prompt,
                       prepared_prompt=prepare_global_prompt, trig=global_trig)
    check_dflash_graph(dflash_graph, drafter_bits, max_rows, 'dflash' if drafter_dir is not None else None)
    check_dflash_head_a8(dflash_head_native_a8, drafter_bits, max_rows,
                         'dflash' if drafter_dir is not None else None, dflash_graph)
    check_dflash_head_w4a16(dflash_head_w4a16, drafter_bits, 'dflash' if drafter_dir is not None else None,
                           dflash_graph, dflash_head_native_a8)
    if type(global_trig) is not bool:
        raise ValueError("global_trig must be a bool")
    if type(spark_sampled_q) is not bool:
        raise ValueError("spark_sampled_q must be a bool")
    if type(spark_global_gqa) is not bool:
        raise ValueError("spark_global_gqa must be a bool")
    if spark_global_gqa:
        from ..sparkulator_metadata import validate_global_gqa_torch
        validate_global_gqa_torch(torch)
    if type(lossless_head) is not bool:
        raise ValueError("lossless_head must be a bool")
    if type(delta4_head) is not bool:
        raise ValueError('delta4_head must be a bool')
    if lossless_head and max_rows > 16:
        raise ValueError("lossless target head verification currently admits at most16 rows")
    if type(prepare_global_prompt) is not bool:
        raise ValueError("prepare_global_prompt must be a bool")
    raw = read_config(model_dir)
    entries = headers(model_dir)
    media_geometry = None
    method = quant_method(raw)
    native = method == "compressed-tensors"
    if delta4_head and not native:
        raise ValueError('Delta4 tied table is a CT-only investigation candidate')
    modelopt = method == "modelopt"
    if dflash_head_w4a16:
        info = entries.get('model.language_model.embed_tokens.weight')
        if not (native or modelopt) or info is None or info['dtype'] != 'BF16' or len(info['shape']) != 2:
            raise ValueError('W4A16 private head requires an original BF16 CT/ModelOpt checkpoint head')
    bounds = None
    if global_cache != "none" and not native:
        raise ValueError('persistent global cache requires native CT arithmetic')
    if global_trig and not native:
        raise ValueError('global trig requires CT arithmetic')
    if lossless_head and not modelopt:
        raise ValueError("lossless_head is a ModelOpt-only investigation candidate")
    if prepare_global_prompt and not (native or modelopt):
        raise ValueError("prepared prompt KV requires CT or ModelOpt arithmetic")
    if native:
        from .nvfp4_weights import CT_PARTS, ct_tensor_contract, validate_headers
        cfg = validate_headers(raw, entries, vision=vision)
        matrices, plain = ct_tensor_contract(cfg)
        names = set(plain)
        names.update(name + '.' + suffix for name in matrices for suffix in CT_PARTS)
        if vision:
            names.update(name for name in entries if name.startswith(('model.vision_tower.', 'model.embed_vision.')))
    elif modelopt:
        from .modelopt_weights import validate_headers, storage_bounds
        cfg = validate_headers(raw, entries, vision=vision)
        bounds = storage_bounds(cfg, entries, vision=vision,
                                **({"lossless_head": True} if lossless_head else {}))
        names = set()  # Mixed native/prepared storage is authoritative in bounds.
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
    if global_trig:
        from .attention import ROTATED, check_geometry
        check_geometry(cfg)
        if all(cfg.sliding(i) for i in range(cfg.layers)):
            raise ValueError('global trig requires at least one full-attention layer')
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
    host_staging = staging
    delta4_bounds = None
    delta4_cpu_peak = 0
    delta4_uma = False
    if delta4_head:
        from tensorfold.cuda.b16_delta4 import header_bounds
        from tensorfold.cuda.capacity import admit_host_commitment, unified

        delta4_bounds = header_bounds((cfg.vocab, cfg.hidden))
        resident += delta4_bounds['max_valid_encoded_bytes'] - delta4_bounds['raw_source_bytes']
        # Do not replace a data-dependent escape bound with the AEON census.
        # CPU source+encoded/temporaries and reader staging coexist until the
        # upload fence; source retirement receives no speculative reuse credit.
        delta4_cpu_peak = (delta4_bounds['raw_source_bytes'] + delta4_bounds['max_valid_encoded_bytes']
                          + delta4_bounds['pack_temporary_bytes'] + 2 * (64 << 20) + 2 * 4096)
        host_staging = max(host_staging, delta4_cpu_peak)
        delta4_uma = unified(torch)
        admit_host_commitment(delta4_cpu_peak)
    if bounds is not None:
        from tensorfold.cuda.capacity import admit_host_commitment, unified
        resident = bounds["resident"]
        host_staging = bounds["host_staging"]
        staging = bounds["device_staging"]
        if unified(torch):
            staging += host_staging
        # Check live host allowance without charging the shared UMA pool twice.
        admit_host_commitment(host_staging)

    if dflash_head_native_a8:
        if (cfg.hidden, cfg.vocab) != (2816, 262144):
            raise ValueError('native A8 proposal head requires hidden2816/vocabulary262144')
        if tuple(torch.cuda.get_device_capability()) != (12, 1):
            raise ValueError('native A8 proposal head prototype requires SM121')

    head_w4_work = None
    draft = spark = None
    kind = companion_kind(drafter_dir) if drafter_dir is not None else None
    check_dflash_graph(dflash_graph, drafter_bits, max_rows, kind)
    check_dflash_head_a8(dflash_head_native_a8, drafter_bits, max_rows, kind, dflash_graph)
    check_dflash_head_w4a16(dflash_head_w4a16, drafter_bits, kind, dflash_graph, dflash_head_native_a8)
    if spark_sampled_q and (kind != 'sparkulator' or not modelopt):
        raise ValueError('Spark sampled q requires the actual Sparkulator/ModelOpt pair')
    if spark_sampled_q and not 2 <= max_rows <= 16:
        raise ValueError('Spark sampled q verification currently admits2..16 rows')
    if spark_global_gqa and (kind != 'sparkulator' or not modelopt):
        raise ValueError('Spark global GQA requires the actual Sparkulator/ModelOpt pair')
    if spark_global_gqa and (type(context) is not int or not 0 < context <= 131072):
        raise ValueError('Spark global GQA requires an explicit context up to131072')
    if kind == "sparkulator":
        from ..sparkulator_metadata import checked_headers
        spark, _ = checked_headers(Path(drafter_dir))
        if drafter_bits != 16:
            raise ValueError("Sparkulator baseline requires bits16")
        if (cfg.hidden, cfg.layers, cfg.vocab, cfg.kv_heads, cfg.global_kv_heads, cfg.head_dim,
                cfg.global_head_dim, cfg.window) != (2816, 30, 262144, 8, 2, 256, 512, 1024):
            raise ValueError("Sparkulator main target cache geometry differs")
        if cfg.max_position > spark.max_position:
            raise ValueError("Sparkulator context cannot cover the main native window")
        head_resident, head_peak = spark.memory_bytes(drafter_bits)
        resident += head_resident
        head_stage = head_peak - head_resident
        from tensorfold.cuda.capacity import admit_host_commitment, unified
        if unified(torch):
            head_stage += spark.host_loading_bytes
        staging = max(staging, head_stage)
        host_staging = max(host_staging, spark.host_loading_bytes)
        admit_host_commitment(host_staging)
    elif drafter_dir is not None:
        from ..dflash_metadata import checked_headers
        draft, draft_entries = checked_headers(Path(drafter_dir))
        if (draft.hidden, draft.target_layers, draft.vocab) != (cfg.hidden, cfg.layers, cfg.vocab):
            raise ValueError('Gemma DFlash companion dimensions disagree with the target')
        if modelopt:
            from .weights import TAP_LAYERS
            if draft.tap_layers != tuple(i for i in TAP_LAYERS if i < cfg.layers):
                raise ValueError('Gemma DFlash target taps disagree with the ModelOpt target')
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
            if dflash_head_w4a16:
                info = entries['model.language_model.embed_tokens.weight']
                if tuple(info['shape']) != (cfg.vocab, cfg.hidden):
                    raise ValueError('W4A16 original head shape differs from target metadata')
                from tensorfold.cuda.capacity import admit_host_commitment, unified
                head_w4_work = dflash_head_w4a16_memory(cfg.hidden, cfg.vocab, max_rows - 1)
                resident += head_w4_work['resident_bytes']
                # Transformer FP8 conversion and CPU head quantization/packing
                # are successive construction phases, not a cross-product max.
                staging = max(staging, sum(bf16.values()) + max(staging_bytes('fp8', n, k) for n, k in shapes),
                              head_w4_work['unified_extra_bytes'] if unified(torch) else head_w4_work['device_extra_bytes'])
                host_staging = max(host_staging, head_w4_work['host_loading_peak_bytes'])
                admit_host_commitment(head_w4_work['host_loading_peak_bytes'])
            else:
                resident += resident_bytes('fp8', cfg.vocab, cfg.hidden)
                extra_head_chunk = 0
                if lossless_head or delta4_head:
                    from tensorfold.cuda.requant import CHUNK_ROWS
                    head_rows = min(CHUNK_ROWS, cfg.vocab, cfg.max_position)
                    # NEW lookup IDs8 + rawBF16 K*2 + returned FP8 K + F32scale4.
                    # Retain existing complete quantization/fragment/source bound;
                    # claim no released-source/allocator reuse credit.
                    extra_head_chunk = head_rows * (3 * cfg.hidden + 12)
                staging = max(staging, sum(bf16.values())
                              + max(staging_bytes('fp8', n, k) for n, k in [*shapes, (cfg.vocab, cfg.hidden)])
                              + extra_head_chunk)
        else:
            resident += sum(bf16.values())

    longest = max(media_geometry.image_soft_tokens, media_geometry.video_soft_tokens) if vision else 0
    rows = widest_forward(max_rows, longest)
    from tensorfold.cuda.sampling import effective_workspace
    from tensorfold.cuda.speculative import chain_workspace
    # Every request policy is supported after admission, including full-support top-k0 and tied-boundary repair.
    # Producer logits are already in workspace/graph geometry; charge only the sampler's additional owners here.
    # Without a companion the engine retains sample_rows and never enters sampled rejection verification.
    sampling = (effective_workspace(streams * max_rows, cfg.vocab, top_k=0)
                if draft is not None or spark is not None else {"device_bytes": 0, "host_bytes": 0})
    proposals = (chain_workspace(max_rows - 1, cfg.vocab, streams=streams, penalized=True)
                 if draft is not None or spark is not None else {"device_bytes": 0, "host_bytes": 0})
    from tensorfold.engine.penalties import bias_workspace
    bias = bias_workspace(max_rows, cfg.vocab, streams=streams)

    q_extra = {"device_bytes": 0, "host_bytes": 0}
    retained_q = 0
    if spark_sampled_q:
        # Same existing recurrent helper and immutable record as E4B. Spark's
        # explicit q policy caps support at256; target sampling/p is unchanged.
        from .e2b_engine import assistant_q_host_bytes
        q_extra = effective_workspace(1, cfg.vocab, top_k=min(cfg.vocab, 256))
        retained_q = assistant_q_host_bytes(max_rows - 1, min(cfg.vocab, 256), proposal_policy=True)

    head_a8_work = dflash_head_a8_workspace(cfg.hidden, cfg.vocab, max_rows - 1) if dflash_head_native_a8 else None

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
        if native or modelopt:
            scratch += expert_workspace(cfg, rows)
        if dflash_head_native_a8:
            scratch = max(scratch, head_a8_work['logits_softcap_bytes']) + head_a8_work['activation_bytes']
        if head_w4_work is not None:
            scratch = max(scratch, head_w4_work['logits_softcap_bytes'])
        from .weights import TAP_LAYERS
        scratch += graph_workspace(cfg, max_rows, slots, 0 if spark is not None else
                                   len([i for i in TAP_LAYERS if i < cfg.layers]))
        if spark is not None:
            scratch += spark.workspace_bytes(min(slots, spark.max_position),
                                             **({"global_gqa": True} if spark_global_gqa else {}))
        if dflash_graph:
            scratch += dflash_graph_workspace(draft, max_rows, min(slots, cfg.max_position))
        if vision:
            v = raw['vision_config']
            patches = max(media_geometry.image_patches, media_geometry.video_patches)
            scratch += 3 * v['num_attention_heads'] * patches**2 * 4
            scratch += 16 * patches * (v['hidden_size'] + v['intermediate_size']) * 4
        return scratch

    def state_bytes(slots):
        target = capacity_bytes(cfg,min(slots,cfg.max_position),rows)
        companion = 0
        if draft is not None:
            companion = sum(companion_rings(draft.layer_types,draft.sliding_window,max_rows,rows,
                                            min(slots,cfg.max_position))) * draft.kv_heads * draft.head_dim * 4
        spark_live = spark.state_bytes(min(slots,spark.max_position)) if spark is not None else 0
        spark_snapshot = spark.snapshot_bytes(min(slots,spark.max_position)) if spark is not None else 0
        return target+companion,spark_live,spark_snapshot

    def geometry(slots):
        live,spark_live,spark_snapshot = state_bytes(slots)
        # Working states are shared once per live stream. Archive credit is
        # separate and replaces, rather than adds to, the legacy snapshots.
        archive = keep*(live+spark_snapshot) if prefix_cache_bytes is None else prefix_cache_bytes
        retained = streams*(live+spark_live)+archive
        prepared = (slots * cfg.global_kv_heads * cfg.global_head_dim * 4
                    if prepare_global_prompt else 0)
        trig = slots * ROTATED * 8 if global_trig else 0
        derived = streams * derived_capacity_bytes(cfg,min(slots,cfg.max_position),global_cache)
        return retained+streams*workspace(slots)+prepared+trig+derived

    if bounds is None and spark is None:
        host_staging = max(host_staging, staging) if delta4_head else staging
    if delta4_uma:
        # Charge the CPU encoding owners once to UMA device staging only
        # after later companion staging is known. Host accounting retains its
        # independent CPU bound rather than importing that UMA sum again.
        staging += delta4_cpu_peak
    # Selecting explicit archive bytes also selects a fixed logical window:
    # an oversized cache grant must reject, never buy room by shrinking it.
    admitted_context = context
    if prefix_cache_bytes is not None and (context is None or context == 0):
        # The shared planner treats zero as native, but its explicit nonfit
        # check needs the resolved positive native window rather than zero.
        admitted_context = cfg.max_position
    admitted_explicit = context_explicit if prefix_cache_bytes is None else True
    result = common(model_dir, admitted_context, admitted_explicit, torch, Geometry(geometry, 0, maximum_slots=131072 if spark_global_gqa else None),
                  lambda name, info: (0, 0),
                  weight_estimator=lambda _: (Weights(resident, staging), host_staging),
                  device_workspace_bytes=sampling["device_bytes"] + proposals["device_bytes"]
                                         + q_extra["device_bytes"] + bias["device_bytes"],
                  host_workspace_bytes=sampling["host_bytes"] + proposals["host_bytes"]
                                       + q_extra["host_bytes"] + retained_q + bias["host_bytes"])
    live,_spark_live,spark_snapshot = state_bytes(result['context_window'])
    result['prefix_cache_default_bytes'] = keep*(live+spark_snapshot)
    result['prefix_cache_bytes'] = result['prefix_cache_default_bytes'] if prefix_cache_bytes is None else prefix_cache_bytes
    result['logit_bias_workspace'] = bias
    if head_w4_work is not None:
        result['dflash_head_w4a16'] = {'memory': head_w4_work, 'owning_logits_dtype': 'float32',
                                    'source': 'original-owned-CPU-BF16', 'unqualified_prototype': True}
    if dflash_head_native_a8:
        result['dflash_head_native_a8'] = {
            'max_rows': max_rows - 1, 'workspace': head_a8_work,
            'owning_logits_dtype': 'float32', 'unqualified_prototype': True}
    if dflash_graph:
        result['dflash_graph'] = {'entries': 1, 'max_rows': max_rows,
                                 'payload_workspace_bytes': dflash_graph_workspace(
                                     draft, max_rows, result['context_window']),
                                 'unqualified_prototype': True}

    if lossless_head:
        result['lossless_head'] = bounds['lossless_head']
    if delta4_head:
        result['delta4_head'] = delta4_bounds
    if spark_global_gqa:
        result['spark_global_gqa'] = {'scope': 'optional forced-efficient global attention only',
                                      'context': result['context_window'], 'unqualified_prototype': True}
    if spark_sampled_q:
        result['spark_sampled_q'] = {'proposal_support_max': min(cfg.vocab, 256),
                                     'retained_q_host_bytes': retained_q,
                                     'one_step_workspace': q_extra, 'unqualified_prototype': True}
    return result
