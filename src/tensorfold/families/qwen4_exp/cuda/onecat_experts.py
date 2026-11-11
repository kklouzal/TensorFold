"""Private, pre-admitted 1Cat GU composition with the current TF down path.

No public flag/default dispatch. The typed selector is for owned qualification
only. Original raw banks remain resident for the unchanged grouped4 prefill and
other-role path. Additional planes are ordinary weight residency, never borrowed
from routing scratch, State capacity, graph budgets or the physical floor.
"""
from dataclasses import dataclass

from tensorfold.cleanup import raise_failures
import hashlib
import json
import sys

from tensorfold.cuda.sm70.onecat_moe_gu import METHODS, PIN

POLICY = "owned-onecat-gu-current-down-v1"
_KINDS = {18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS"}


@dataclass(frozen=True)
class OneCatGU:
    method: str
    device_budget_bytes: int
    global_host_bytes: int

    def __post_init__(self):
        if type(self.method) is not str or self.method not in METHODS:
            raise ValueError("explicit qualified-trial GU method required")
        if any(type(n) is not int or not 0 < n <= sys.maxsize
               for n in (self.device_budget_bytes, self.global_host_bytes)):
            raise ValueError("explicit positive device/global-host preparation credits required")

    @property
    def rows(self):
        return 32

    @property
    def identity(self):
        return (POLICY, self.method, self.device_budget_bytes, self.global_host_bytes, "reference-grouped4")


def _key(plan):
    return (str(plan.source.path), plan.source.offset, plan.source.ggml_type,
            plan.row_bytes, plan.source.rows, tuple(plan.row_ranges), tuple(plan.columns), plan.source.row_bytes)


def _describe(prototype, staging_bytes):
    from .gguf_weights import ExpertPlan
    from ..expert_partition import ExpertPartition
    from tensorfold.cuda.sm70.onecat_moe_packing import requirements

    c = prototype.cfg
    if c.execution_dtype != "float16":
        raise ValueError("1Cat GU trial requires original GGUF Half execution")
    if (c.hidden, c.experts, c.top_k, c.moe_width, c.shared_width, len(prototype.layers)) != (2560, 512, 10, 160, 80, 48):
        raise ValueError("1Cat trial requires the actual SC117 48-layer geometry")
    descriptors = []
    payload = 0
    table_groups = set()
    max_host_chunk = 0
    for layer in prototype.layers:
        bank = layer.moe.experts
        p = bank.partition
        if descriptors and p != descriptors[0][1]:
            raise ValueError("1Cat model mixes rank-local expert placements across layers")
        if (type(bank) is not ExpertPlan or type(p) is not ExpertPartition or p.world != 8 or p.etp != 4
                or p.experts != 512 or p.full_width != 640 or p.width != 160 or p.count != 256
                or bank.count != 512 or bank.width != 160 or bank.dims != 2560):
            raise ValueError("1Cat trial differs from immutable E256/lane160 placement")
        expected_rows = tuple((expert * 640 + p.column, 160) for expert in range(p.first, p.stop))
        kind = bank.gate.source.ggml_type
        if kind not in _KINDS or bank.up.source.ggml_type != kind:
            raise ValueError("initial 1Cat trial requires a supported same-format GU pair")
        sizes = requirements(kind)
        max_rows = staging_bytes // bank.gate.source.row_bytes // 160 * 160
        if max_rows < 160:
            raise MemoryError("GGUF staging cannot hold a complete owned expert")
        count = min(256, max_rows // 160)
        max_host_chunk = max(max_host_chunk, count * (sizes["codes_shape"][1] + sizes["scales_shape"][1]))
        pair = []
        for matrix in (bank.gate, bank.up):
            if (not matrix.packed or matrix.n != 256 * 160 or matrix.k != 2560 or matrix.column_offset != 0
                    or tuple(matrix.row_ranges) != expected_rows or matrix.source.rows != 512 * 640
                    or matrix.row_bytes != 10 * {18: 98, 21: 110, 22: 82, 23: 136}[kind]):
                raise ValueError("1Cat GU source does not retain original expert rows/whole input blocks")
            pair.append((_key(matrix), kind, max_rows))
            payload += sizes["output_bytes"]
        table_groups.add("iq2" if kind == 22 else "iq3" if kind in (18, 21) else "empty")
        descriptors.append((tuple(pair), p, kind))
    keys = [key for pair, _, _ in descriptors for key, _, _ in pair]
    if len(set(keys)) != 96:
        raise ValueError("initial 1Cat GU trial requires distinct original gate/up bank sources")
    payload += (8192 if "iq2" in table_groups else 0) + (3072 if "iq3" in table_groups else 0)
    return tuple(descriptors), payload, max_host_chunk, table_groups


class _Lease:
    def __init__(self):
        self.enabled = False
        self.closed = False
        self.retiring = False


class _Sink:
    """One fixed matrix owner, journaled before either GPU allocation."""
    def __init__(self, key, kind, max_rows, device):
        self.key, self.kind, self.max_rows, self.device = key, kind, max_rows, device
        self.codes = self.scales = None
        self.written = 0
        self.raw_hash = hashlib.sha256()
        self.codes_hash = hashlib.sha256()
        self.scales_hash = hashlib.sha256()
        self.receipt = None

    def start(self):
        import torch
        from tensorfold.cuda.sm70.onecat_moe_packing import requirements

        if self.codes is not None or self.scales is not None:
            raise RuntimeError("1Cat matrix allocation cannot be repeated")
        sizes = requirements(self.kind)
        self.codes = torch.empty(sizes["codes_shape"], dtype=torch.uint8, device=self.device)
        self.scales = torch.empty(sizes["scales_shape"], dtype=torch.uint8, device=self.device)

    def consume(self, selected, destination):
        import numpy as np
        import torch
        from tensorfold.cuda.sm70.onecat_moe_packing import _pack_expert, requirements

        sizes = requirements(self.kind)
        if (type(selected) is not torch.Tensor or selected.device.type != "cpu" or selected.dtype != torch.uint8
                or not selected.is_contiguous() or selected.dim() != 2 or selected.shape[1] != self.key[3]
                or type(destination) is not int or destination != self.written or destination % 160
                or not 0 < selected.shape[0] <= self.max_rows or selected.shape[0] % 160
                or destination + selected.shape[0] > 40960 or self.codes is None or self.receipt is not None):
            raise ValueError("cold GU chunk differs from its exact contiguous source lease")
        count = selected.shape[0] // 160
        raw = selected.numpy()
        codes = np.empty((count, sizes["codes_shape"][1]), np.uint8)
        scales = np.empty((count, sizes["scales_shape"][1]), np.uint8)
        for expert in range(count):
            view = raw[expert * 160:(expert + 1) * 160]
            _, packed_codes, packed_scales = _pack_expert(view, self.kind)
            codes[expert] = packed_codes
            scales[expert] = packed_scales
            self.raw_hash.update(memoryview(view))
            del packed_codes, packed_scales
        begin = destination // 160
        # Blocking copies retain both NumPy/Torch CPU borrowers through transfer.
        # No temporary GPU plane tensor or per-expert H2D copy is introduced.
        self.codes[begin:begin + count].copy_(torch.from_numpy(codes), non_blocking=False)
        self.scales[begin:begin + count].copy_(torch.from_numpy(scales), non_blocking=False)
        self.codes_hash.update(memoryview(codes))
        self.scales_hash.update(memoryview(scales))
        self.written += selected.shape[0]
        if self.written == 40960:
            self.receipt = {"kind": self.kind, "experts": 256, "raw_sha256": self.raw_hash.hexdigest(),
                            "codes_sha256": self.codes_hash.hexdigest(), "scales_sha256": self.scales_hash.hexdigest()}


class _Bank:
    def __init__(self, spec, lease, gu, down, gate_sink, up_sink, table, partition, bank, packer=None):
        self.spec, self.lease, self.gu, self.down = spec, lease, gu, down
        self.gate_sink, self.up_sink, self.table, self.partition = gate_sink, up_sink, table, partition
        self.raw_gate, self.raw_up, self.raw_down = bank.gate, bank.up, bank.down
        self.packer = packer

    def device_tensors(self):
        # A mutable shared frame is an execution owner, never immutable model
        # weight identity. Engine's additive resident credit pays it once.
        if self.packer is not None:
            return ()
        return (self.gate_sink.codes, self.gate_sink.scales, self.up_sink.codes, self.up_sink.scales, self.table)

    def run_slots(self, bank, x, picks, *, local, activation, out, grouping):
        if (not self.lease.enabled or self.lease.retiring or self.lease.closed or bank.gate is not self.raw_gate
                or bank.up is not self.raw_up or bank.down is not self.raw_down or bank.partition != self.partition):
            raise RuntimeError("1Cat GU bank/owner is not the adopted immutable resident bank")
        if self.packer is not None:
            self.packer.pack_active(bank.gate.data, bank.up.data, picks,
                self.gate_sink.codes, self.gate_sink.scales, self.up_sink.codes, self.up_sink.scales,
                first=self.partition.first, ggml_type=self.gate_sink.kind,
                gate_column_offset=bank.gate.column_offset, up_column_offset=bank.up.column_offset)
        self.gu.gate_up(x, picks, self.gate_sink.codes, self.gate_sink.scales,
                       self.up_sink.codes, self.up_sink.scales, self.table, activation,
                       first=self.partition.first, format_name=_KINDS[self.gate_sink.kind], method=self.spec.method)
        # Original shared Dense program after GU initialization; never the GU
        # activation status slot, never Q8_1 and never a routed expert remap.
        shared = bank.owned_shared_result(x)
        return self.down.run(x, picks, bank.gate, bank.up, bank.down, first=self.partition.first,
                             shared=shared, local=local, activation=activation, out=out, grouping=grouping)


class OneCatOwner:
    """Cold fixed owner; Engine journals it before native/GPU preparation.

    Fixed weight tensors remain retained after close, like Engine.w. Close
    rejects execution first and marks closed only after a positive device fence. An uncertain fence
    retains all references and cannot produce a positive closed owner.
    """
    def __init__(self, spec, device):
        import torch

        if type(spec) is not OneCatGU:
            raise TypeError("typed private GU selector required")
        self.spec, self.device = spec, torch.device(device)
        self.lease = _Lease()
        self.gu = self.down = None
        self.descriptors = self.sinks = self.forecast_record = None
        self.tables = {"iq2": None, "iq3": None, "empty": None}
        self.bound = [False] * 48
        self.started = self.complete = False
        self.phase_before = self.phase_after = None
        self.source_planes_sha256 = None
        self.native_error = None
        self._status = [None, None]
        self._native_started = False
        self.retirement_error = None

    def prepare_native(self):
        if self._native_started:
            raise RuntimeError("1Cat native preparation cannot be retried")
        self._native_started = True
        try:
            from tensorfold.cuda.sm70 import gguf_ops, onecat_moe_gu

            self.gu = onecat_moe_gu.prepare(self.device)
            self.down = gguf_ops.prepare_owned_from_activation(self.device)
            return {"policy": POLICY, "selector": self.spec.identity,
                    "gu_module": self.gu.module.__name__, "down_module": self.down.module.__name__}
        except BaseException as error:
            self.native_error = error
            raise

    def prepare_cohort(self, engine, channel, operation):
        """Reuse the existing operator vote; preserve initiating prepare errors."""
        from .execution_identity import _phase, _retire_store, _fatal

        def protected():
            try:
                return operation()
            except BaseException as error:
                self.native_error = error
                raise

        try:
            rows = _phase(engine, channel, 'prepare', protected, may_refuse=False)
            if any(row != rows[0] for row in rows):
                raise RuntimeError("native 1Cat operator preparation differs across ranks")
            _retire_store(engine, channel, stages=('prepare',))
        except BaseException as error:
            primary = self.native_error if self.native_error is not None else error
            self._status[0] = error
            self._status[1] = primary
            try:
                _fatal(engine, primary)
            except BaseException as notification:
                self._status[1] = notification
            raise_failures(primary, self._status)

    def forecast(self, prototype, staging_bytes):
        descriptors, payload, host_chunk, groups = _describe(prototype, staging_bytes)
        if payload > self.spec.device_budget_bytes:
            raise MemoryError("1Cat plane residency exceeds explicit device credit")
        # Minimum payload alone is not a managed-heap/RSS certificate. The
        # explicit GLOBAL caller credit must include all8 ranks and NumPy cost;
        # its physical adequacy is an independent qualification gate.
        if self.spec.global_host_bytes < 8 * (host_chunk + 217600):
            raise MemoryError("1Cat GLOBAL host credit cannot contain known simultaneous chunk payloads")
        if self.descriptors is not None:
            if descriptors != self.descriptors:
                raise ValueError("loaded GU source/placement differs from admitted descriptors")
            return self.forecast_record
        self.descriptors = descriptors
        self.sinks = tuple(_Sink(key, kind, rows, self.device)
                           for pair, _, _ in descriptors for key, kind, rows in pair)
        self.forecast_record = {"policy": POLICY, "pin": PIN, "method": self.spec.method,
            "banks": 48, "decode_rows_limit": 32, "output_kind": "unweighted-slots",
            "additional_device_bytes": payload,
            "global_host_credit_bytes": self.spec.global_host_bytes,
            "max_host_chunk_payload_bytes_per_rank": host_chunk,
            "original_raw_retained": True, "baseline_grouped_gate_up_rows": 4,
            "tables": tuple(sorted(groups)),
            "format_banks": {str(kind): sum(item[2] == kind for item in descriptors) for kind in _KINDS},
            "qualification": "private-unqualified"}
        return self.forecast_record

    def begin_load(self, prototype, staging_bytes):
        import torch
        from tensorfold.cuda.sm70.onecat_moe_packing import table

        self.forecast(prototype, staging_bytes)
        if self.started or self.gu is None or self.down is None or self.lease.closed or self.lease.retiring:
            raise RuntimeError("1Cat cold loading lacks its once-only native/admission owner")
        self.started = True
        self.phase_before = {"free": int(torch.cuda.mem_get_info(self.device)[0]),
                             "allocated": int(torch.cuda.memory_allocated(self.device)),
                             "reserved": int(torch.cuda.memory_reserved(self.device))}
        for group in self.forecast_record["tables"]:
            kind = {"iq2": 22, "iq3": 18, "empty": 23}[group]
            cpu = table(kind)
            self.tables[group] = torch.empty(cpu.shape, dtype=torch.uint8, device=self.device)
            self.tables[group].copy_(torch.from_numpy(cpu), non_blocking=False)

    def sink(self, plan):
        if not self.started or self.complete or self.lease.closed:
            raise RuntimeError("1Cat source callback is outside its cold load lease")
        key = _key(plan)
        for sink in self.sinks:
            if key == sink.key:
                sink.start()
                return sink
        return None

    def bind_bank(self, plan, bank):
        for index, (pair, partition, kind) in enumerate(self.descriptors):
            if pair[0][0] == _key(plan.gate) and pair[1][0] == _key(plan.up):
                gate, up = self.sinks[2 * index:2 * index + 2]
                if self.bound[index] or gate.receipt is None or up.receipt is None or bank.partition != partition:
                    raise RuntimeError("1Cat resident bank lacks complete matching source planes")
                group = "iq2" if kind == 22 else "iq3" if kind in (18, 21) else "empty"
                bank._onecat_provider = _Bank(self.spec, self.lease, self.gu, self.down,
                                             gate, up, self.tables[group], partition, bank)
                self.bound[index] = True
                return
        raise ValueError("1Cat resident bank was not in the admitted model plan")

    def finish_load(self):
        import torch

        if not self.started or self.complete or not all(self.bound) or any(s.receipt is None for s in self.sinks):
            raise RuntimeError("1Cat model load has incomplete planes/banks")
        torch.cuda.synchronize(self.device)
        actual = sum(t.numel() * t.element_size() for sink in self.sinks for t in (sink.codes, sink.scales))
        actual += sum(t.numel() * t.element_size() for t in self.tables.values() if t is not None)
        if actual != self.forecast_record["additional_device_bytes"]:
            raise MemoryError("materialized 1Cat weight planes differ from admitted byte claim")
        self.phase_after = {"free": int(torch.cuda.mem_get_info(self.device)[0]),
                            "allocated": int(torch.cuda.memory_allocated(self.device)),
                            "reserved": int(torch.cuda.memory_reserved(self.device))}
        self.source_planes_sha256 = hashlib.sha256(json.dumps(
            [sink.receipt for sink in self.sinks], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.complete = True

    def adopt(self, spec):
        if self.retirement_error is not None:
            raise self.retirement_error
        if self.lease.retiring:
            raise RuntimeError("retiring 1Cat owner cannot be adopted again")
        if spec is not self.spec or not self.complete or self.lease.closed:
            raise RuntimeError("1Cat provider lacks complete owned admission")
        self.lease.enabled = True

    def profile(self):
        if not self.complete:
            raise RuntimeError("incomplete 1Cat provider has no positive profile")
        return {**self.forecast_record, "selector": self.spec.identity,
                "gu_module": self.gu.module.__name__, "down_module": self.down.module.__name__}

    def record(self):
        return {"profile": self.profile(), "partition": self.descriptors[0][1].identity(),
                "source_planes_sha256": self.source_planes_sha256,
                "source_planes": tuple(s.receipt for s in self.sinks),
                "loading_phase_before": self.phase_before, "loading_phase_after": self.phase_after,
                "phase_scope": "whole weight load; not exact plane physical bytes",
                "closed": self.lease.closed, "enabled": self.lease.enabled}

    def close(self):
        import torch

        if not self.lease.closed:
            self.lease.retiring = True
            self.lease.enabled = False
            try:
                if self.started:
                    torch.cuda.synchronize(self.device)
            except BaseException as error:
                if self.retirement_error is None:
                    self.retirement_error = error
                raise
            self.lease.closed = True
