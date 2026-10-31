"""Exact replicated/sharded DFlash2 payloads and bounded source conversion.

Projection row/column shards follow the pinned SGLang reference. Feature FC,
both selector codebooks, grouped convolutions and all norms remain replicated.
Every retained tensor is FP16; temporary FP32 arithmetic is separately budgeted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, _metadata_object, _regular_stream, checkpoint_path, read_header

from .dflash2_config import DFlash2Config, count

SOURCE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
LOAD_CHUNK = 8 << 20


def source_stat(path):
    return _stat_identity(Path(path).stat())


def _stat_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _config_capture(path):
    """Parse/hash the same bounded opened JSON version, preserving strict parser rules."""
    with _regular_stream(path) as stream:
        before = _stat_identity(os.fstat(stream.fileno()))
        size = before[2]
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError("DFlash2 config metadata exceeds the existing JSON byte bound")
        raw = stream.read(size + 1)
        if len(raw) != size:
            raise ValueError("DFlash2 config changed during its bounded read")
        value = _metadata_object(raw, path)
        digest = hashlib.sha256(raw).hexdigest()
        if _stat_identity(os.fstat(stream.fileno())) != before or source_stat(path) != before:
            raise ValueError("DFlash2 config changed during parse/hash capture")
    return value, digest, before


def snapshot_budget(bytes_each, *, slots, keep, points):
    count(bytes_each, "snapshot bytes", 0)
    count(slots, "snapshot sequence slots")
    count(keep, "kept snapshot count", 0)
    count(points, "in-flight snapshot points per sequence", 0)
    versions = keep + slots * points
    return {"snapshot_bytes_each": bytes_each, "snapshot_versions": versions, "snapshot_bytes": versions * bytes_each}


def selector_host_budget(config, sequences):
    """Bound retained application payload/objects under the actual Python ABI.

    One sequence at a time converts its compact lattice to FP64 and creates
    immutable q rows; the batch's FP32 lattice/IDs and finished packets coexist.
    Runtime/library managed effects still require the separate measured peak.
    """
    import sys
    import numpy as np
    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow

    count(sequences, "proposal sequence count", 0)
    proposals, width = config.block - 1, config.top_k
    row = ProbabilityRow(np.arange(width, dtype=np.int64), np.ones(width, dtype=np.float64))
    row_bytes = sys.getsizeof(row) + sum(sys.getsizeof(value) + sys.getsizeof(value.base) for value in (row.ids, row.probs))
    packet = ChainPacket((0,) * proposals, (row,) * proposals, 0)
    packet_bytes = (sys.getsizeof(packet) + sys.getsizeof(packet.tokens) + sys.getsizeof(packet.q_rows)
                    + proposals * sys.getsizeof(config.vocab - 1) + sys.getsizeof(config.context)
                    + proposals * row_bytes)
    arrays = sequences * proposals * (width * 8 + width * width * 4)
    # The scalar walk's explicit score/order/probability/mask arrays, plus
    # their array headers and temporary token/q lists, never scale with B.
    transient = proposals * width * width * 8 + width * (16 * 8 + 4)
    transient += 20 * sys.getsizeof(np.empty(0)) + 2 * sys.getsizeof([None] * proposals)
    return {"host_selector_arrays_bytes": arrays, "host_immutable_q_packets_bytes": sequences * packet_bytes,
            "host_selector_transient_bytes": transient if sequences else 0,
            "host_workspace_bytes": arrays + sequences * packet_bytes + (transient if sequences else 0),
            "host_workspace_evidence": "application-array/object-construction-bound; managed peak qualification pending"}


def tensor_shapes(c):
    shapes = {"fc.weight": (c.hidden, len(c.taps) * c.hidden), "hidden_norm.weight": (c.hidden,),
              "norm.weight": (c.hidden,), "candidate_selector.hidden_projection.weight": (c.selector_rank, c.hidden),
              "candidate_selector.predecessor_codebook": (c.vocab, c.selector_rank),
              "candidate_selector.successor_codebook": (c.vocab, c.selector_rank)}
    for layer in range(c.layers):
        p = f"layers.{layer}."
        shapes.update({p + "input_layernorm.weight": (c.hidden,), p + "post_attention_layernorm.weight": (c.hidden,),
                       p + "self_attn.q_norm.weight": (c.head_dim,), p + "self_attn.k_norm.weight": (c.head_dim,),
                       p + "self_attn.q_proj.weight": (c.heads * c.head_dim, c.hidden),
                       p + "self_attn.k_proj.weight": (c.kv_heads * c.head_dim, c.hidden),
                       p + "self_attn.v_proj.weight": (c.kv_heads * c.head_dim, c.hidden),
                       p + "self_attn.o_proj.weight": (c.hidden, c.heads * c.head_dim),
                       p + "mlp.gate_proj.weight": (c.intermediate, c.hidden),
                       p + "mlp.up_proj.weight": (c.intermediate, c.hidden),
                       p + "mlp.down_proj.weight": (c.hidden, c.intermediate)})
        for kind in ("attention_conv", "mlp_conv"):
            shapes[p + kind + ".base_kernel"] = (2, c.conv_taps, c.hidden)
            shapes[p + kind + ".kernel_projection.weight"] = (2 * c.conv_taps * (c.hidden // c.group_size), c.hidden)
    return shapes


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple
    source_shape: tuple
    rows: tuple | None = None
    columns: tuple | None = None

    @property
    def nbytes(self):
        return math.prod(self.shape) * 2


def plan_weights(config, header, *, tp=(0, 8)):
    rank, world = tp
    local = config.partition(rank, world)
    shapes = tensor_shapes(config)
    actual = {name: row for name, row in header.items() if name != "__metadata__"}
    if actual.keys() != shapes.keys():
        raise ValueError("DFlash2 checkpoint tensor namespace differs from the declared architecture")
    plans = []
    for name, shape in shapes.items():
        row = actual[name]
        if row.get("shape") != list(shape) or row.get("dtype") not in SOURCE_BYTES:
            raise ValueError("DFlash2 tensor shape/dtype mismatch: " + name)
        rows = columns = None
        if name.endswith("self_attn.q_proj.weight"):
            width = local["heads"] * config.head_dim
            rows = rank * width, (rank + 1) * width
        elif name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
            width = local["kv_heads"] * config.head_dim
            rows = rank * width, (rank + 1) * width
        elif name.endswith("self_attn.o_proj.weight"):
            width = local["heads"] * config.head_dim
            columns = rank * width, (rank + 1) * width
        elif name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            width = local["intermediate"]
            rows = rank * width, (rank + 1) * width
        elif name.endswith("mlp.down_proj.weight"):
            width = local["intermediate"]
            columns = rank * width, (rank + 1) * width
        selected = list(shape)
        if rows is not None:
            selected[0] = rows[1] - rows[0]
        if columns is not None:
            selected[1] = columns[1] - columns[0]
        plans.append(TensorPlan(name, tuple(selected), shape, rows, columns))
    return tuple(plans)


@dataclass(frozen=True)
class DFlash2Plan:
    config: DFlash2Config
    path: Path
    header: dict
    tensors: tuple
    tp: tuple
    config_sha256: str
    source_sha256: str | None = None
    source_stat: tuple | None = None
    config_stat: tuple | None = None
    config_path: Path | None = None

    @classmethod
    def read(cls, directory, *, target=None, tp=(0, 8), filename="model.safetensors", block=4, expected_sha256=None):
        cfg_path = checkpoint_path(directory, "config.json")
        raw_config, config_sha256, config_version = _config_capture(cfg_path)
        config = DFlash2Config.from_dict(raw_config, block=block)
        if target is not None:
            config.check_target(target)
        path = checkpoint_path(directory, filename)
        before = source_stat(path)
        _, header = read_header(path, SOURCE_BYTES)
        if source_stat(path) != before:
            raise ValueError("DFlash2 tensor file changed during header planning")
        plans = plan_weights(config, header, tp=tp)
        if source_stat(cfg_path) != config_version:
            raise ValueError("DFlash2 config changed across tensor header admission")
        return cls(config, path, header, plans, tp, config_sha256, expected_sha256, before,
                   config_version, cfg_path)

    @property
    def resident_bytes(self):
        return sum(row.nbytes for row in self.tensors)

    def resources(self, *, slots, context, prefill_rows, execution_rows, keep, points):
        c = self.config
        for value, name in ((slots, "slots"), (prefill_rows, "prefill rows"), (execution_rows, "execution rows")):
            count(value, name)
        local = c.partition(*self.tp)
        if type(context) is not int or not 1 <= context <= c.context:
            raise ValueError("DFlash2 selected context exceeds the draft's absolute position support")
        state = c.state_bytes(tp=self.tp[1], context=context)
        # One bounded feature frame per preparation/verification buffer. Cache
        # materialization stages all layers before publishing a ring update.
        rows = max(prefill_rows, execution_rows)
        feature = (prefill_rows + execution_rows) * ((len(c.taps) + c.hc_streams) * c.hidden * 2
                                                    + (c.hidden // 256) * c.hc_streams * 4)
        # Commit copies the accepted subset of one frame before projection;
        # norm/RoPE temporaries remain bounded by those scheduled rows.
        conditioning = rows * (len(c.taps) * c.hidden * 2 + 24 * (c.hidden + local["kv_heads"] * c.head_dim))
        staged = max(rows * c.layers * local["kv_heads"] * c.head_dim * 4, state)
        attention = 12 * (min(context, c.window) + c.block) * local["kv_heads"] * c.head_dim
        attention += 8 * c.block * local["heads"] * (min(context, c.window) + c.block)
        draft_rows = c.block * min(slots, execution_rows // 2)
        frame = 16 * draft_rows * (c.hidden + local["intermediate"] + local["heads"] * c.head_dim)
        head = draft_rows * (c.vocab // self.tp[1] + int(c.vocab % self.tp[1] != 0)) * 16
        selector = draft_rows * c.top_k * (12 * c.selector_rank + 4 * c.top_k)
        capture_workspace = rows * c.hidden * (4 * (self.tp[1] + c.hc_streams + 1) + 2)
        return {"resident_bytes": self.resident_bytes, "state_bytes_each": state,
                "state_bytes": slots * state, **snapshot_budget(state, slots=slots, keep=keep, points=points),
                **selector_host_budget(c, draft_rows // c.block),
                "feature_frame_bytes": feature, "conditioning_stage_bytes": staged,
                "feature_capture_workspace_bytes": capture_workspace,
                "snapshot_reconstruction_parts_bytes": state,
                "conditioning_workspace_bytes": conditioning,
                "execution_workspace_bytes": feature + staged + max(conditioning, capture_workspace,
                                                                     attention + frame + head + selector),
                "loading_host_staging_bytes": 5 * LOAD_CHUNK, "target_shared_weight_bytes": 0,
                "draft_cache_policy": "physical-fp16-swa-ring-accepted-prefix-v1", "draft_window": c.window,
                "draft_block_rows": c.block, "actual_q": "conditional-selector-sparse16"}

    def serving_resources(self, *, slots, context, prefill_rows, execution_rows, keep, points, target_vocab):
        """One authoritative serving envelope, including full target p extraction.

        ``prefill_rows`` is the actual prompt-buffer extent, including shared
        expert rows when converged. Source target logits already belong to the
        target buffers; effective_workspace explicitly excludes those logits.
        Host q packets remain live while target probabilities are constructed.
        """
        from tensorfold.cuda.sampling import effective_workspace

        count(target_vocab, "target vocabulary")
        if target_vocab % self.tp[1]:
            raise ValueError("the initial DFlash2 target vocabulary must divide across TP8")
        resource = self.resources(slots=slots, context=context, prefill_rows=prefill_rows,
                                  execution_rows=execution_rows, keep=keep, points=points)
        probability = effective_workspace(execution_rows, target_vocab // self.tp[1], world=self.tp[1], top_k=0)
        return {**resource, "target_probability_device_bytes": probability["device_bytes"],
                "target_probability_host_bytes": probability["host_bytes"],
                "device_serving_bytes": resource["state_bytes"] + resource["snapshot_bytes"]
                    + resource["execution_workspace_bytes"] + probability["device_bytes"],
                "host_serving_bytes": resource["host_workspace_bytes"] + probability["host_bytes"]}


@dataclass
class DFlash2Weights:
    config: DFlash2Config
    tensors: dict
    tp: tuple
    identity: str
    shard_identity: str = ""

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def device_tensors(self):
        return tuple(self.tensors.values())


def load_shard(plan, *, device="cuda", chunk_bytes=LOAD_CHUNK, destination=None, source_authority=None):
    import torch
    from safetensors import safe_open

    count(chunk_bytes, "source load chunk bytes")
    if plan.config_path is not None and source_stat(plan.config_path) != plan.config_stat:
        raise ValueError("DFlash2 config changed after header admission")
    if plan.source_stat is not None and source_stat(plan.path) != plan.source_stat:
        raise ValueError("DFlash2 tensor file changed after header admission")
    if torch.device(device).type == "cuda" and plan.source_sha256 is None:
        raise ValueError("DFlash2 source SHA256 must be pinned before any device allocation")
    if chunk_bytes < max(math.prod(p.source_shape[1:]) * SOURCE_BYTES[plan.header[p.name]["dtype"]] for p in plan.tensors):
        raise ValueError("DFlash2 load chunk must hold one complete source row")
    if source_authority is not None:
        from tensorfold.engine.model_identity import ModelIdentity
        if type(source_authority) is not ModelIdentity:
            raise ValueError("DFlash2 hash reuse requires process-issued content authority")
        source_authority.verify_unchanged()
        matches = [item for item in source_authority.files if Path(item.path) == plan.path.resolve()]
        if not matches or any(item.sha256 != matches[0].sha256 for item in matches):
            raise ValueError("DFlash2 tensor file is absent from its issued input closure")
        source_hash = matches[0].sha256
    else:
        from tensorfold.engine.model_identity import capture_model_identity

        # The existing capture owner hashes only the opened extent, probes one
        # EOF byte, and validates descriptor/path versions before publication.
        # An append cannot extend preparation until a producer eventually stops.
        expected = plan.source_stat if plan.source_stat is not None else source_stat(plan.path)
        captured = capture_model_identity({"draft/weights": plan.path},
            runtime_identity={"loader": "dflash2-source-f16-v1"},
            authorize=lambda _name, path: path.resolve(strict=True),
            max_file_bytes=expected[2], max_total_bytes=expected[2])
        if captured.files[0].stat_identity != expected:
            raise ValueError("DFlash2 tensor file changed across source identity capture")
        source_hash = captured.files[0].sha256
    if plan.source_sha256 is not None and source_hash != plan.source_sha256:
        raise ValueError("DFlash2 tensor-file content differs from the pinned manifest")
    result = {} if destination is None else destination
    if result:
        raise ValueError("DFlash2 load destination must be a fresh owned tensor dictionary")
    identity = hashlib.sha256((plan.config.identity + "|" + source_hash + "|f16-f32acc-tp-v1|" + str(plan.tp[1])).encode())
    shard_identity = hashlib.sha256(str(plan.tp).encode())
    with safe_open(plan.path, framework="pt", device="cpu") as source:
        for tensor in plan.tensors:
            target = torch.empty(tensor.shape, dtype=torch.float16, device=device)
            result[tensor.name] = target  # journal the owned allocation before conversion
            shard_identity.update(tensor.name.encode())
            row_bytes = math.prod(tensor.source_shape[1:]) * SOURCE_BYTES[plan.header[tensor.name]["dtype"]]
            step = max(1, chunk_bytes // row_bytes)
            raw_slice = source.get_slice(tensor.name)
            begin, end = tensor.rows or (0, tensor.source_shape[0])
            for at in range(begin, end, step):
                raw = raw_slice[at:min(at + step, end)]
                if tensor.columns is not None:
                    raw = raw[:, tensor.columns[0]:tensor.columns[1]]
                if not bool(torch.isfinite(raw).all()) or bool((raw.abs().float() > 65504).any()):
                    raise ValueError("DFlash2 source is nonfinite or overflows FP16: " + tensor.name)
                shard_identity.update(raw.contiguous().view(torch.uint8).numpy().tobytes())
                target[at - begin:at - begin + raw.shape[0]].copy_(raw.to(torch.float16), non_blocking=False)
    if plan.source_stat is not None and source_stat(plan.path) != plan.source_stat:
        raise ValueError("DFlash2 tensor file changed during source loading")
    if plan.config_path is not None and source_stat(plan.config_path) != plan.config_stat:
        raise ValueError("DFlash2 config changed during source loading")
    if source_authority is not None:
        source_authority.verify_unchanged()
    return DFlash2Weights(plan.config, result, plan.tp, identity.hexdigest(), shard_identity.hexdigest())
