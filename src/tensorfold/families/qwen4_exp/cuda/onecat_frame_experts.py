"""Private active-expert GPU repacking into one reused GU execution frame.

Original raw weights/prefill remain unchanged. No all-layer derived-plane cache,
CPU bank repacking, extra Reader or per-step host transfer/readback is used.
The existing single decoder worker owns the frame until each current-stream
GU reader completes; another role/stream must not access it concurrently.
"""
import hashlib
import json

from .onecat_experts import OneCatOwner, _Bank, _describe, _key

POLICY = "owned-onecat-active-frame-current-down-v1"


class _View:
    def __init__(self, kind):
        self.kind = kind
        self.codes = self.scales = None


class OneCatFrameOwner(OneCatOwner):
    """Frame candidate reuses the completed C6 lifecycle, never its cold cache."""
    def __init__(self, spec, device):
        super().__init__(spec, device)
        self.packer = None
        self.frame = None
        self.views = {kind: (_View(kind), _View(kind)) for kind in (18, 21, 22, 23)}
        self.raw_bindings = [None] * 48

    def prepare_native(self):
        from tensorfold.cuda.sm70 import onecat_gpu_packing

        try:
            result = super().prepare_native()
            if self.spec.method != "tensorfold":
                self.packer = onecat_gpu_packing.prepare(self.device)
            return {**result, "policy": POLICY, "packer_module": None if self.packer is None else self.packer.module.__name__}
        except BaseException as error:
            self.native_error = error
            raise

    def forecast(self, prototype, staging_bytes):
        from tensorfold.cuda.sm70.onecat_gpu_packing import layout

        descriptors, _, _, groups = _describe(prototype, staging_bytes)
        frame_bytes = (0 if self.spec.method == "tensorfold" else
                       max(layout(kind)["pair_plane_bytes"] for kind in self.views))
        workspace = max(layout(kind)["workspace_bytes"] for _, _, kind in descriptors)
        if workspace:
            raise ValueError("frame policy requires the declared zero-workspace GPU producer ABI")
        if self.spec.method == "tensorfold":
            groups = set()
        lookup = (8192 if "iq2" in groups else 0) + (3072 if "iq3" in groups else 0)
        payload = frame_bytes + lookup
        if payload > self.spec.device_budget_bytes:
            raise MemoryError("GU frame/table bytes exceed explicit device credit")
        if self.spec.global_host_bytes < 8 * lookup:
            raise MemoryError("GLOBAL metadata/codebook credit cannot hold lookup payloads")
        if self.descriptors is not None:
            if descriptors != self.descriptors:
                raise ValueError("loaded raw GU descriptors differ from admitted frame sources")
            return self.forecast_record
        self.descriptors = descriptors
        self.forecast_record = {"policy": POLICY, "pin": "1b57260a834d40faa32bfc6ac18f050414a8f042",
            "method": self.spec.method, "down_method": self.spec.down_method,
            "iq2_source": self.spec.iq2_source, "prefill_method": self.spec.prefill_method,
            "down_format_policy": {"20": self.spec.down_method, "42": "tensorfold"},
            "mtp_provider": "tensorfold-fp32-mtp", "banks": 48, "decode_rows_limit": 32,
            "output_kind": "unweighted-slots", "additional_device_bytes": payload,
            "additional_device_scope": "persistent mutable execution frame; additive resident credit, not immutable weights",
            "frame_slab_bytes": frame_bytes, "shared_lookup_bytes": lookup, "packer_workspace_bytes": workspace,
            "global_host_credit_bytes": self.spec.global_host_bytes,
            "host_policy": "metadata/codebook only; no host packed-bank cache",
            "packing_policy": "once-per-prefill-layer-union; decode-per-panel",
            "frame_ownership": "exclusive decoder worker/current stream; no reentry",
            "original_raw_retained": True, "baseline_grouped_gate_up_rows": 4,
            "tables": sorted(groups),
            "format_banks": {str(kind): sum(item[2] == kind for item in descriptors) for kind in self.views},
            "qualification": "private-unqualified"}
        return self.forecast_record

    def begin_load(self, prototype, staging_bytes):
        import torch
        from tensorfold.cuda.sm70.onecat_gpu_packing import layout

        if self.packer is None and self.spec.method != "tensorfold":
            raise RuntimeError("GPU packer was not resolved before admission")
        # Parent only allocates the deduplicated lookup tensors and journals the
        # whole-load observation. Its virtual forecast dispatch uses this policy.
        super().begin_load(prototype, staging_bytes)
        self.frame = torch.empty(self.forecast_record["frame_slab_bytes"], dtype=torch.uint8, device=self.device)
        for kind, (gate, up) in self.views.items():
            if self.spec.method == "tensorfold":
                continue
            facts = layout(kind)
            offset = 0
            for output in (gate, up):
                code_shape, scale_shape = facts["codes_shape"], facts["scales_shape"]
                code_bytes, scale_bytes = code_shape[0] * code_shape[1], scale_shape[0] * scale_shape[1]
                output.codes = self.frame.narrow(0, offset, code_bytes).view(code_shape)
                offset += code_bytes
                output.scales = self.frame.narrow(0, offset, scale_bytes).view(scale_shape)
                offset += scale_bytes
            if offset > self.frame.numel():
                raise MemoryError("format views exceed the funded GU frame")

    def sink(self, plan):
        # Preserve the original loader, chunks and original raw H2D exactly.
        return None

    def bind_bank(self, plan, bank):
        for index, (pair, partition, kind) in enumerate(self.descriptors):
            if pair[0][0] == _key(plan.gate) and pair[1][0] == _key(plan.up):
                if self.bound[index] or bank.partition != partition or self.frame is None:
                    raise RuntimeError("raw frame bank lacks its once-only admitted placement")
                for raw in (bank.gate, bank.up):
                    if (raw.n != 40960 or raw.k != 2560 or raw.ggml_type != kind or raw.column_offset != 0
                            or raw.data.device != self.device or raw.data.numel() != raw.n * raw.data.shape[1]):
                        raise ValueError("actual raw GU bank differs from its frame descriptor")
                gate, up = self.views[kind]
                group = "iq2" if kind == 22 else "iq3" if kind in (18, 21) else "empty"
                if self.spec.method == "tensorfold":
                    group = "empty"
                # No reference back to this Owner: fixed weights can release
                # deterministically without an Owner/provider reference cycle.
                bank._onecat_provider = _Bank(self.spec, self.lease, self.gu, self.down,
                    gate, up, self.tables[group], partition, bank, packer=self.packer, q8_down=self.q8_down, raw_iq2_gu=self.raw_iq2_gu)
                self.raw_bindings[index] = {"layer": index, "kind": kind,
                    "gate_source_descriptor_sha256": hashlib.sha256(json.dumps(pair[0][0], separators=(",", ":")).encode()).hexdigest(),
                    "up_source_descriptor_sha256": hashlib.sha256(json.dumps(pair[1][0], separators=(",", ":")).encode()).hexdigest(),
                    "gate_raw_bytes": bank.gate.data.numel(), "up_raw_bytes": bank.up.data.numel(),
                    "down_kind": bank.down.ggml_type, "down_column_offset": bank.down.column_offset,
                    "source_scope": "descriptor hashes, not GPU content certificates; existing model authority owns raw content"}
                self.bound[index] = True
                return
        raise ValueError("raw bank was not declared by frame admission")

    def finish_load(self):
        import torch

        if (not self.started or self.complete or not all(self.bound) or self.frame is None
                or self.lease.retiring or self.lease.closed):
            raise RuntimeError("frame model load has incomplete raw bindings")
        torch.cuda.synchronize(self.device)
        actual = self.frame.numel() + sum(t.numel() for t in self.tables.values() if t is not None)
        if actual != self.forecast_record["additional_device_bytes"]:
            raise MemoryError("actual GU frame/table payload differs from its admission")
        self.phase_after = {"free": int(torch.cuda.mem_get_info(self.device)[0]),
                            "allocated": int(torch.cuda.memory_allocated(self.device)),
                            "reserved": int(torch.cuda.memory_reserved(self.device))}
        self.complete = True

    def profile(self):
        if not self.complete:
            raise RuntimeError("incomplete GPU frame has no positive profile")
        return {**self.forecast_record, "selector": list(self.spec.identity),
                "gu_module": self.gu.module.__name__, "down_module": self.down.module.__name__,
                "packer_module": None if self.packer is None else self.packer.module.__name__,
                "q8_down_module": None if self.q8_down is None else self.q8_down.module.__name__,
                "raw_iq2_gu_module": None if self.raw_iq2_gu is None else self.raw_iq2_gu.module.__name__}

    def record(self):
        return {"schema": "tensorfold.onecat-active-frame-owner-v1", "profile": self.profile(),
                "partition": self.descriptors[0][1].identity(), "raw_bindings": list(self.raw_bindings),
                "frame": {"bytes": self.frame.numel(), "device": str(self.frame.device),
                          "dtype": str(self.frame.dtype), "storage_pointer_local_only": self.frame.untyped_storage().data_ptr()},
                "packing_scope": "active experts repacked per layer; inactive bytes are not initialized/observed",
                "derived_plane_hashes": "not recorded; no cold derived-bank content claim",
                "loading_phase_before": self.phase_before, "loading_phase_after": self.phase_after,
                "phase_scope": "whole raw weight load/frame allocation, not exact allocator overhead",
                "closed": self.lease.closed, "enabled": self.lease.enabled}
