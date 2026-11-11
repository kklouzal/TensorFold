"""Private cold 1Cat qualification binding; no device operations or serving flags.

Receipts attest prepared, execution-eligible objects, not invocation counts,
physical admission, numeric quality or performance. The caller owns publication
and bounded host receipt storage outside measured request work.
"""

from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path


def create_capability(method, *, device_budget_bytes=None, global_host_bytes=None):
    """Return the actual typed selector; old-image baseline imports nothing new."""
    if method is None:
        if device_budget_bytes is not None or global_host_bytes is not None:
            raise ValueError("1Cat budgets require an explicit method")
        return None
    from tensorfold.families.qwen4_exp.cuda.onecat_experts import OneCatGU

    return OneCatGU(method, device_budget_bytes, global_host_bytes)


def _integer(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= sys.maxsize:
        raise ValueError("1Cat receipt requires bounded integer bytes/counts")
    return value


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return left.keys() == right.keys() and all(_same(left[k], v) for k, v in right.items())
    if type(right) in (tuple, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _origin(module):
    """Actual loaded module locator; deliberately no file hash or device query."""
    name, path = getattr(module, "__name__", None), getattr(module, "__file__", None)
    if type(name) is not str or not name or len(name) > 256 or type(path) is not str or not path or len(path) > 4096:
        raise ValueError("1Cat prepared module lacks its actual bounded origin")
    return {"module": name, "path": str(Path(path).resolve(strict=True))}


def _unrequested(engine):
    w = getattr(engine, "w", None)
    meta = {} if w is None else getattr(w, "meta", {})
    capacity = getattr(engine, "capacity_plan", {})
    plan = getattr(engine, "_owned_expert_plan", None)
    requested = getattr(engine, "_owned_expert_gate_up_requested", None)
    plans = (plan, meta.get("owned_expert_plan"))
    def selected(value):
        return type(value).__module__ in (
            "tensorfold.families.qwen4_exp.cuda.onecat_experts",
            "tensorfold.families.qwen4_exp.cuda.onecat_frame_experts")

    if (getattr(engine, "_onecat_owner", None) is not None
            or meta.get("onecat_experts") is not None or "onecat_gu_weights" in capacity
            or "onecat_weight_residency" in capacity.get("owned_expert_native", {})
            or any(getattr(p, "_onecat", False) or selected(getattr(p, "_gate_up", None)) for p in plans)
            or selected(requested)):
        raise ValueError("unrequested 1Cat owner/selection is active")
    if w is not None and any(getattr(getattr(getattr(layer, "moe", None), "experts", None), "_onecat_provider", None) is not None
                             for layer in getattr(w, "layers", ())):
        raise ValueError("unrequested 1Cat bank provider is active")
    return {}


def bound_receipt(engine, capability):
    """Bind the active frame to actual owners without device work or content hashes.

    Tensor shape/storage access is host metadata. The local storage pointer is
    only an alias observation, never an artifact or cross-process identity.
    """
    if capability is None:
        return _unrequested(engine)
    from tensorfold.families.qwen4_exp.cuda import onecat_experts as api
    from tensorfold.families.qwen4_exp.cuda import onecat_frame_experts as frame_api
    from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
    from tensorfold.cuda.sm70.onecat_gpu_packing import layout

    if type(capability) is not api.OneCatGU:
        raise TypeError("actual typed OneCatGU capability required")
    w, plan, owner = engine.w, engine._owned_expert_plan, engine._onecat_owner
    if (type(owner) is not frame_api.OneCatFrameOwner or owner is not w.meta.get("onecat_experts")
            or owner.spec is not capability or engine._owned_expert_gate_up_requested is not capability
            or type(plan) is not Plan or not plan._onecat or plan._gate_up is not capability
            or w.meta.get("owned_expert_plan") is not plan or engine.multi.w is not w
            or type(engine.tp) is not int or engine.tp != 8
            or type(engine.rank) is not int or not 0 <= engine.rank < 8
            or type(plan.partition) is not ExpertPartition
            or plan.partition != engine.expert_partition or plan.partition.rank != engine.rank
            or plan.partition.world != 8 or plan.partition.etp != 4
            or owner.complete is not True or owner.lease.enabled is not True
            or owner.lease.retiring is not False or owner.lease.closed is not False
            or owner.native_error is not None or w.mtp is not None
            or type(engine.depth) is not int or engine.depth != 0):
        raise ValueError("1Cat frame Engine/Plan/owner differs from its adopted request")
    if (len(w.layers) != 48 or len(owner.bound) != 48 or any(v is not True for v in owner.bound)
            or len(owner.descriptors) != 48 or set(owner.views) != {18, 21, 22, 23}
            or set(owner.tables) != {"empty", "iq2", "iq3"}):
        raise ValueError("1Cat frame requires all 48 complete target bindings")
    record = owner.record()
    if type(record) is not dict or set(record) != {
            "schema", "profile", "partition", "raw_bindings", "frame", "packing_scope", "derived_plane_hashes",
            "loading_phase_before", "loading_phase_after", "phase_scope", "closed", "enabled"}:
        raise ValueError("1Cat frame requires its actual owner schema, not retained planes")
    profile = record["profile"]
    expected = {"policy": frame_api.POLICY, "pin": api.PIN, "method": capability.method,
        "banks": 48, "decode_rows_limit": 32, "output_kind": "unweighted-slots",
        "additional_device_bytes": 117976064,
        "additional_device_scope": "persistent mutable execution frame; additive resident credit, not immutable weights",
        "frame_slab_bytes": 117964800, "shared_lookup_bytes": 11264, "packer_workspace_bytes": 0,
        "global_host_credit_bytes": capability.global_host_bytes,
        "host_policy": "metadata/codebook only; no host packed-bank cache",
        "packing_policy": "current-layer active valid owned experts; no host readback",
        "frame_ownership": "exclusive decoder worker/current stream; no reentry",
        "original_raw_retained": True, "baseline_grouped_gate_up_rows": 4,
        "tables": ("empty", "iq2", "iq3"), "format_banks": {"18": 17, "21": 10, "22": 20, "23": 1},
        "qualification": "private-unqualified", "selector": capability.identity,
        "gu_module": owner.gu.module.__name__, "down_module": owner.down.module.__name__,
        "packer_module": owner.packer.module.__name__}
    if (not _same(profile, expected) or not _same(engine.capacity_plan.get("onecat_gu_weights"), profile)
            or not _same(engine.multi.owned_expert_resources.get("onecat_weight_residency"), profile)
            or record["schema"] != "tensorfold.onecat-active-frame-owner-v1"
            or record["partition"] != plan.partition.identity()
            or record["enabled"] is not True or record["closed"] is not False
            or record["phase_scope"] != "whole raw weight load/frame allocation, not exact allocator overhead"
            or record["packing_scope"] != "active experts repacked per layer; inactive bytes are not initialized/observed"
            or record["derived_plane_hashes"] != "not recorded; no cold derived-bank content claim"):
        raise ValueError("1Cat frame profile/schema differs from requested/materialized bounds")
    if capability.device_budget_bytes < profile["additional_device_bytes"]:
        raise ValueError("1Cat frame exceeds caller device credit")
    frame = owner.frame
    pointer = frame.untyped_storage().data_ptr()
    _integer(pointer, positive=True)
    frame_facts = {"bytes": frame.numel(), "device": str(frame.device), "dtype": str(frame.dtype),
                   "storage_pointer_local_only": pointer}
    if (tuple(frame.shape) != (117964800,) or frame.storage_offset() != 0 or not frame.is_contiguous()
            or frame.numel() != 117964800 or str(frame.dtype) != "torch.uint8"
            or frame.device != owner.device or not _same(record["frame"], frame_facts)):
        raise ValueError("1Cat shared frame metadata differs from its actual storage")
    for kind, (gate, up) in owner.views.items():
        facts, offset = layout(kind), 0
        for output in (gate, up):
            if type(output) is not frame_api._View or output.kind != kind:
                raise ValueError("1Cat requires its actual per-format frame views")
            for tensor, shape in ((output.codes, facts["codes_shape"]), (output.scales, facts["scales_shape"])):
                if (tuple(tensor.shape) != shape or str(tensor.dtype) != "torch.uint8"
                        or tensor.device != owner.device or not tensor.is_contiguous()
                        or tensor.untyped_storage().data_ptr() != pointer or tensor.storage_offset() != offset):
                    raise ValueError("1Cat format views do not alias disjoint admitted frame intervals")
                offset += tensor.numel()
        if offset != facts["pair_plane_bytes"] or offset > frame.numel():
            raise ValueError("1Cat frame format extent differs")
    for group, size in (("empty", 0), ("iq2", 8192), ("iq3", 3072)):
        table = owner.tables[group]
        if (table.numel() != size or table.device != owner.device or str(table.dtype) != "torch.uint8"
                or not table.is_contiguous() or size and table.untyped_storage().data_ptr() == pointer):
            raise ValueError("1Cat lookup metadata differs")
    bindings = record["raw_bindings"]
    if type(bindings) is not tuple or len(bindings) != 48:
        raise ValueError("1Cat requires 48 ordered raw descriptor bindings")
    formats = {18: 0, 21: 0, 22: 0, 23: 0}
    for index, layer in enumerate(w.layers):
        bank, (pair, partition, kind) = layer.moe.experts, owner.descriptors[index]
        provider = getattr(bank, "_onecat_provider", None)
        if kind not in formats:
            raise ValueError("1Cat raw format is unsupported")
        gate, up = owner.views[kind]
        group = "iq2" if kind == 22 else "iq3" if kind in (18, 21) else "empty"
        if (type(provider) is not api._Bank or provider.spec is not capability or provider.lease is not owner.lease
                or provider.gu is not owner.gu or provider.down is not owner.down or provider.packer is not owner.packer
                or provider.gate_sink is not gate or provider.up_sink is not up
                or provider.partition != plan.partition or bank.partition != plan.partition or partition != plan.partition
                or provider.raw_gate is not bank.gate or provider.raw_up is not bank.up or provider.raw_down is not bank.down
                or provider.table is not owner.tables[group] or getattr(bank, "_owned_expert_plan", None) is not plan):
            raise ValueError("1Cat provider differs from raw bank/packer/shared frame")
        for raw in (bank.gate, bank.up):
            if (raw.ggml_type != kind or raw.column_offset != 0 or raw.n != 40960 or raw.k != 2560
                    or raw.data.device != owner.device or str(raw.data.dtype) != "torch.uint8"):
                raise ValueError("1Cat actual raw GU metadata differs")
        expected_binding = {"layer": index, "kind": kind,
            "gate_source_descriptor_sha256": hashlib.sha256(json.dumps(pair[0][0], separators=(",", ":")).encode()).hexdigest(),
            "up_source_descriptor_sha256": hashlib.sha256(json.dumps(pair[1][0], separators=(",", ":")).encode()).hexdigest(),
            "gate_raw_bytes": bank.gate.data.numel(), "up_raw_bytes": bank.up.data.numel(),
            "down_kind": bank.down.ggml_type, "down_column_offset": bank.down.column_offset,
            "source_scope": "descriptor hashes, not GPU content certificates; existing model authority owns raw content"}
        if not _same(bindings[index], expected_binding):
            raise ValueError("1Cat raw source descriptor binding differs")
        formats[kind] += 1
    if formats != {18: 17, 21: 10, 22: 20, 23: 1}:
        raise ValueError("1Cat actual provider format census differs")
    for field in ("loading_phase_before", "loading_phase_after"):
        phase = record[field]
        if type(phase) is not dict or set(phase) != {"free", "allocated", "reserved"}:
            raise ValueError("1Cat whole-load phase observation is missing")
        for value in phase.values():
            _integer(value)
    origins = {"owner": _origin(frame_api), "gate_up": _origin(owner.gu.module),
               "down": _origin(owner.down.module), "packer": _origin(owner.packer.module)}
    for field, prefix in (("gate_up", "tf_onecat_gu_"), ("down", "tf_gguf_"), ("packer", "tf_onecat_pack_")):
        if not origins[field]["module"].startswith(prefix):
            raise ValueError("1Cat frame native module locator differs")
    return {"schema": "tensorfold.private-onecat-frame-binding-v1", "rank": engine.rank, "world": engine.tp,
            "status": "prepared-execution-eligible", "record": deepcopy(record), "origins": origins,
            "storage_pointer_scope": "process-local alias observation; not persistent/cross-rank/content identity",
            "qualification": "private-unqualified; no invocation, physical-admission, quality or performance proof"}
