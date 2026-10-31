"""Shared-target GGUF MTP import with an explicit QSA numerical contract.

The converter fuses [fc_embedding | fc_hidden] and stores already-centred RMS
gammas. The runtime splits that projection without changing any target weights.
The canonical companion's two vocabulary copies are descriptive only: the
existing target owns embedding and output execution. Explicit older shared-core
controls retain their recorded zero compression entry under this QSA profile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.direct_read import Reader

from ..gguf import GGUFCheckpoint, decode_rows, resolve_gguf, token_id_identity
from .gguf_weights import (
    ConcatPlan, DECODE_BYTES, ExpertPlan, GGUFDense,
    MATMUL_ROWS, Planner, STAGING_BYTES, materializer,
)
from .weight_types import AttnW, HC, LayerW, MoEW, MTPW

POLICY = "tensorfold.mtp-qsa4-2048-v1"


def model_plan(target, directory, *, tp=(0, 1)):
    """Validate the entire companion before any payload IO or device allocation."""
    entry = resolve_gguf(directory)
    if entry is None:
        raise ValueError("shared GGUF MTP requires a GGUF companion")
    checkpoint = GGUFCheckpoint.open(entry)
    c = target.cfg
    rank, world = tp
    full = target.meta.get("full", c)
    if (type(world) is not int or world not in (1, 2, 4, 8)
            or type(rank) is not int or rank not in range(world)
            or full.heads % world or full.moe_width % world or full.shared_width % world
            or c.heads * world != full.heads or c.moe_width * world != full.moe_width
            or c.shared_width * world != full.shared_width):
        raise ValueError("MTP target and tensor-parallel partition disagree")
    if c.index_ratio != 4 or c.index_budget != 2048:
        raise ValueError("this MTP numerical profile declares QSA ratio four and budget 2048")
    metadata = checkpoint.metadata
    position_profile = target.meta.get("position_profile")
    mtp_rope = None if position_profile is None else position_profile.mtp.rope
    if mtp_rope is not None:
        sections = metadata.get("qwen4exp.rope.dimension_sections", (11, 11, 10))
        if (metadata.get("qwen4exp.context_length") != mtp_rope.native_context
                or type(metadata.get("qwen4exp.context_length")) is not int
                or type(sections) is not tuple or len(sections) < 3
                or any(type(n) is not int for n in sections)
                or sections[:3] != mtp_rope.mrope_section):
            raise ValueError("MTP companion positional reference/layout differs from its own resolved role")
    shared = metadata.get("qwen4exp.nextn_shared_target_tensors")
    vocabulary_copies = {name: checkpoint.tensors[name] for name in ("token_embd.weight", "output.weight")
                         if name in checkpoint.tensors}
    if vocabulary_copies:
        if (len(vocabulary_copies) != 2 or (shared is not None and shared is not False)
                or any(t.shape != (c.vocab, c.hidden) for t in vocabulary_copies.values())):
            raise ValueError("MTP companion vocabulary copies must describe both complete target matrices")
    elif shared is not True:
        raise ValueError("MTP companion without vocabulary copies must declare shared target tensors")
    if (type(metadata.get("qwen4exp.nextn_predict_layers")) is not int
            or metadata.get("qwen4exp.nextn_predict_layers") != 1
            or type(metadata.get("qwen4exp.block_count")) is not int
            or metadata.get("qwen4exp.block_count") != c.layers + 1):
        raise ValueError("MTP companion must contain one declared shared-target prediction block")
    expected = {"embedding_length": c.hidden, "attention.head_count": full.heads,
                "attention.head_count_kv": full.kv_heads, "attention.key_length": c.head_dim,
                "attention.value_length": c.head_dim, "expert_count": c.experts,
                "expert_used_count": c.top_k, "expert_feed_forward_length": full.moe_width,
                "expert_shared_feed_forward_length": full.shared_width,
                "hyper_connection.count": c.streams, "hyper_connection.low_rank": c.low,
                "attention.indexer.head_count": c.index_heads,
                "attention.indexer.key_length": c.index_dim,
                "attention.indexer.top_k": c.index_budget,
                "rope.dimension_count": c.rotary_dim if mtp_rope is None else mtp_rope.rotary_dim,
                "rope.freq_base": c.rope_theta if mtp_rope is None else mtp_rope.theta}
    for name, value in expected.items():
        actual = metadata.get("qwen4exp." + name)
        if isinstance(actual, bool) or (type(value) is int and type(actual) is not int) or actual != value:
            raise ValueError("MTP target geometry differs at " + name)
    if len(metadata.get("tokenizer.ggml.tokens", ())) != c.vocab:
        raise ValueError("MTP vocabulary must match the shared target token space")
    vocabulary = token_id_identity(metadata)
    if target.meta.get("gguf_token_id_identity") != vocabulary:
        raise ValueError("MTP token-ID mapping differs from the selected target")
    ratios = metadata.get("qwen4exp.attention.compress_ratios")
    if (not isinstance(ratios, tuple) or len(ratios) != c.layers + 1
            or any(type(n) is not int or n not in (0, 4) for n in ratios)):
        raise ValueError("MTP source compression metadata is not the declared QSA family")
    if vocabulary_copies and ratios[-1] != c.index_ratio:
        raise ValueError("full MTP companion must explicitly declare its QSA compression ratio")
    # GGUF serializes epsilon as FP32. Match the target under that representation.
    source_eps = metadata.get("qwen4exp.attention.layer_norm_rms_epsilon")
    if type(source_eps) is not float or np.float32(source_eps) != np.float32(c.eps):
        raise ValueError("MTP RMS epsilon differs from its target")
    global_kv = full.kv_heads
    if global_kv >= world:
        if global_kv % world:
            raise ValueError("MTP KV heads do not partition TP")
        local_kv, first_kv = global_kv // world, rank * (global_kv // world)
    else:
        if world % global_kv:
            raise ValueError("MTP KV replication groups do not partition TP")
        local_kv, first_kv = 1, rank // (world // global_kv)
    if c.kv_heads != local_kv:
        raise ValueError("MTP local KV head count differs from target")
    planner = Planner(checkpoint)
    base = f"blk.{c.layers}."
    d, hd, wide = c.hidden, c.head_dim, c.streams * c.hidden

    def matrix(name, shape, **kwargs):
        source = checkpoint.tensors.get(base + name)
        if source is None or source.shape != shape or source.ggml_type not in (0, 1, 8, 30):
            raise ValueError("MTP matrix shape/encoding differs at " + name)
        return planner.matrix(base + name, **kwargs)

    def array(name, shape, **kwargs):
        source = checkpoint.tensors.get(base + name)
        if source is None or source.shape != shape or source.ggml_type != 0:
            raise ValueError("MTP normalization/router shape or encoding differs at " + name)
        return planner.array(base + name, **kwargs)

    def hc(prefix, inject):
        down = matrix(prefix + "down.weight", (c.low, wide))
        if inject:
            down = ConcatPlan((down, matrix(prefix + "inject.weight", (c.streams, wide))))
        up = matrix(prefix + "up.weight", (wide, c.low))
        return HC(down, up, array(prefix + "norm.weight", (wide,)), inject, down, up)

    qrows = range(rank * c.heads * 2 * hd, (rank + 1) * c.heads * 2 * hd)
    kvrows = range(first_kv * hd, (first_kv + local_kv) * hd)
    projection = ConcatPlan((matrix("attn_q.weight", (full.heads * 2 * hd, d), rows=qrows),
                            matrix("attn_k.weight", (global_kv * hd, d), rows=kvrows),
                            matrix("attn_v.weight", (global_kv * hd, d), rows=kvrows),
                            matrix("indexer.q_proj.weight", (c.index_heads * c.index_dim, d)),
                            matrix("indexer.k_proj.weight", (c.index_dim, d))))
    attention = AttnW(projection, array("attn_q_norm.weight", (hd,)), array("attn_k_norm.weight", (hd,)),
                      array("indexer.q_norm.weight", (c.index_dim,)),
                      array("indexer.k_norm.weight", (c.index_dim,)),
                      matrix("attn_output.weight", (d, full.heads * hd),
                             columns=((rank * c.heads * hd, c.heads * hd),)))
    lo, width = rank * c.moe_width, c.moe_width
    expert_rows = [e * full.moe_width + i for e in range(c.experts) for i in range(lo, lo + width)]
    gate = matrix("ffn_gate_exps.weight", (c.experts, full.moe_width, d), rows=expert_rows, packed=True)
    up = matrix("ffn_up_exps.weight", (c.experts, full.moe_width, d), rows=expert_rows, packed=True)
    down = matrix("ffn_down_exps.weight", (c.experts, d, full.moe_width), columns=((lo, width),), packed=True)
    slo, swidth = rank * c.shared_width, c.shared_width
    shared = (matrix("ffn_gate_shexp.weight", (full.shared_width, d), rows=range(slo, slo + swidth)),
              matrix("ffn_up_shexp.weight", (full.shared_width, d), rows=range(slo, slo + swidth)),
              matrix("ffn_down_shexp.weight", (d, full.shared_width), columns=((slo, swidth),)))
    router = ConcatPlan((array("ffn_gate_inp.weight", (c.experts, d), transform="half_to_float"),
                         array("ffn_gate_inp_shexp.weight", (d,), transform="half_to_float")))
    layer = LayerW(c.layers, False, hc("hc_attn_", True), hc("hc_ffn_", True), None, attention,
                   MoEW(router, ExpertPlan(gate, up, down, shared, c.experts, width, d)))
    fusion_shape = (d, 2 * d)
    result = MTPW(array("nextn.enorm.weight", (d,)), array("nextn.hnorm.weight", (wide,)),
                  matrix("nextn.eh_proj.weight", fusion_shape, columns=((0, d),)),
                  matrix("nextn.eh_proj.weight", fusion_shape, columns=((d, d),)),
                  layer, hc("nextn.hc_head_", False))
    # Descriptor validation above does not create a plan for either copy. In
    # particular no materializer or payload reader can reach those source bytes.
    unused = set(checkpoint.tensors) - planner.used - vocabulary_copies.keys()
    if unused:
        raise ValueError("unmapped shared MTP tensors: " + ", ".join(sorted(unused)))
    return result, planner, {"policy": POLICY, "attention": "qsa", "compression_ratio": 4,
                             "complete_group_token_budget": 2048,
                             "source_final_compression_ratio": ratios[-1],
                             "source": str(Path(entry).resolve()), "shared_target_embedding_head": True,
                             "token_id_identity": vocabulary,
                             "source_tensor_count": len(checkpoint.tensors),
                             "source_payload_bytes": sum(t.nbytes for t in checkpoint.tensors.values()),
                             "imported_tensor_count": len(planner.used),
                             "imported_source_payload_bytes": sum(checkpoint.tensors[n].nbytes for n in planner.used),
                             "shared_target_tensors": {"embedding": "target.embed", "output": "target.head"},
                             "skipped_source_tensors": {
                                 name: {"shape": list(t.shape), "encoding": t.encoding, "bytes": t.nbytes,
                                        "runtime": "shared-target-alias"}
                                 for name, t in vocabulary_copies.items()},
                             "execution_dtype": "float16", "ple": False,
                             "rms_weights": "already-centred-gguf-gamma",
                             "fusion": "split-fp16-linears-and-per-hc-stream-add",
                             "qualification": "import controls only; QSA execution and quality pending"}


def byte_plan(target, directory, *, tp=(0, 1)):
    _, planner, receipt = model_plan(target, directory, tp=tp)
    resident = sum(m.resident_bytes for m in planner.matrices) + sum(a.resident_bytes for a in planner.arrays)
    router_host = max((a.transform_host_staging_bytes for a in planner.arrays), default=0)
    c = target.cfg
    return {"resident_bytes": resident, "staging_bytes": STAGING_BYTES,
            "host_staging_bytes": max(3 * STAGING_BYTES, router_host),
            "workspace_bytes": DECODE_BYTES + MATMUL_ROWS * c.streams * c.hidden * 4,
            "world": tp[1], "import": receipt}


def _cpu_dense(plan, reader, device):
    """Bounded host import control using the independent existing GGUF decoder."""
    if torch.device(device).type != "cpu":
        raise ValueError("CPU dense import control requires its explicit CPU device")
    result = torch.empty((plan.n, plan.k), dtype=torch.float16)
    limit = max(1, min(STAGING_BYTES // plan.source.row_bytes,
                       DECODE_BYTES // (4 * plan.source.shape[-1])))
    destination = 0
    for first, count in plan.row_ranges:
        for start in range(first, first + count, limit):
            rows = min(limit, first + count - start)
            raw = reader.read(plan.source.path, plan.source.offset + start * plan.source.row_bytes,
                              rows * plan.source.row_bytes).reshape(rows, plan.source.row_bytes)
            decoded = decode_rows(plan.source, raw.numpy())
            columns = [decoded[:, lo:lo + n] for lo, n in plan.columns]
            selected = columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)
            half = selected.astype(np.float16)
            if not np.all(np.isfinite(half)):
                raise ValueError("MTP decoded projection is not finite in FP16")
            result[destination:destination + rows].copy_(torch.from_numpy(half))
            destination += rows
    return GGUFDense(result)


def attach(target, directory, *, tp=(0, 1)):
    prototype, _, receipt = model_plan(target, directory, tp=tp)
    reader = Reader()
    device = target.device
    materialize = materializer(reader, device, dense_loader=_cpu_dense if torch.device(device).type == "cpu" else None)

    try:
        imported = materialize(prototype)
        reader.close()
    except BaseException as primary:
        try:
            reader.close()
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
        raise
    target.mtp = imported
    # The common caller may subsequently prepare a configured vocabulary subset.
    target.draft_head = target.draft_ids = None
    target.meta.update(mtp_checkpoint=receipt["source"], mtp_storage="shared-gguf-execution-fp16",
                       mtp_import=receipt)
    return target
