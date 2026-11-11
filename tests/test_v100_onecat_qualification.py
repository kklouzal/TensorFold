"""Frame binding controls using actual owners and CPU-only tensor metadata doubles."""
import builtins
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("onecat_qualification_test_tool", ROOT / "tools/v100_onecat_qualification.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


class TensorMetadata:
    """No CUDA allocation, Tensor operation or actual admission is simulated."""
    def __init__(self, shape, pointer, offset=0):
        self.shape, self.pointer, self.offset = shape, pointer, offset
        self.device, self.dtype = "cuda:fixture", "torch.uint8"

    def numel(self):
        return math.prod(self.shape)

    def untyped_storage(self):
        return NS(data_ptr=lambda: self.pointer)

    def storage_offset(self):
        return self.offset

    def is_contiguous(self):
        return True


def fixture(rank=0, method="tensorfold-fp32-epilogue"):
    from tensorfold.families.qwen4_exp.cuda import onecat_experts as api
    from tensorfold.families.qwen4_exp.cuda import onecat_frame_experts as frame_api
    from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
    from tensorfold.cuda.sm70.onecat_gpu_packing import layout

    spec = tool.create_capability(method, device_budget_bytes=117976064, global_host_bytes=1 << 20)
    owner = object.__new__(frame_api.OneCatFrameOwner)
    owner.spec, owner.lease = spec, api._Lease()
    owner.lease.enabled = True
    owner.complete, owner.native_error, owner.device = True, None, "cuda:fixture"
    partition = ExpertPartition(8, rank, 512, 640, 4)
    plan = Plan(partition, 2560, 11, 24, 4096, False)
    object.__setattr__(plan, "_gate_up", spec)
    object.__setattr__(plan, "_onecat", True)
    handles = []
    for name in ("tf_onecat_gu_fixture", "tf_gguf_fixture", "tf_onecat_pack_fixture"):
        module = ModuleType(name)
        module.__file__ = __file__  # Locator only; no ELF integrity claim.
        handles.append(NS(module=module))
    owner.gu, owner.down, owner.packer = handles
    pointer = (rank + 1) << 32
    owner.frame = TensorMetadata((117964800,), pointer)
    owner.tables = {name: TensorMetadata((size,), pointer + (1 << 30) + size)
                    for name, size in (("empty", 0), ("iq2", 8192), ("iq3", 3072))}
    owner.views = {}
    for kind in (18, 21, 22, 23):
        pair, offset, facts = [], 0, layout(kind)
        for _ in range(2):
            output = frame_api._View(kind)
            output.codes = TensorMetadata(facts["codes_shape"], pointer, offset)
            offset += output.codes.numel()
            output.scales = TensorMetadata(facts["scales_shape"], pointer, offset)
            offset += output.scales.numel()
            pair.append(output)
        owner.views[kind] = tuple(pair)
    kinds = [22] * 20 + [21] * 10 + [18] * 17 + [23]
    blocks = {22: 82, 21: 110, 18: 98, 23: 136}
    layers, descriptors, bindings = [], [], []
    for index, kind in enumerate(kinds):
        pair = tuple(((f"original-{index}-{role}", index, kind), kind, 160) for role in ("gate", "up"))
        descriptors.append((pair, partition, kind))
        def raw():
            return NS(data=TensorMetadata((40960, 10 * blocks[kind]), pointer + (1 << 31)),
                      n=40960, k=2560, ggml_type=kind, column_offset=0)
        bank = NS(gate=raw(), up=raw(), down=NS(ggml_type=20, column_offset=0),
                  partition=partition, _owned_expert_plan=plan)
        group = "iq2" if kind == 22 else "iq3" if kind in (18, 21) else "empty"
        bank._onecat_provider = api._Bank(spec, owner.lease, owner.gu, owner.down,
            *owner.views[kind], owner.tables[group], partition, bank, packer=owner.packer)
        bindings.append({"layer": index, "kind": kind,
            "gate_source_descriptor_sha256": hashlib.sha256(json.dumps(pair[0][0], separators=(",", ":")).encode()).hexdigest(),
            "up_source_descriptor_sha256": hashlib.sha256(json.dumps(pair[1][0], separators=(",", ":")).encode()).hexdigest(),
            "gate_raw_bytes": bank.gate.data.numel(), "up_raw_bytes": bank.up.data.numel(),
            "down_kind": bank.down.ggml_type, "down_column_offset": bank.down.column_offset,
            "source_scope": "descriptor hashes, not GPU content certificates; existing model authority owns raw content"})
        layers.append(NS(moe=NS(experts=bank)))
    owner.raw_bindings, owner.descriptors, owner.bound = bindings, tuple(descriptors), [True] * 48
    owner.phase_before = {"free": 8 << 30, "allocated": 20 << 30, "reserved": 21 << 30}
    owner.phase_after = {"free": 7 << 30, "allocated": 21 << 30, "reserved": 22 << 30}
    owner.forecast_record = {"policy": frame_api.POLICY, "pin": api.PIN, "method": spec.method,
        "banks": 48, "decode_rows_limit": 32, "output_kind": "unweighted-slots",
        "additional_device_bytes": max(layout(k)["pair_plane_bytes"] for k in kinds) + 11264,
        "additional_device_scope": "persistent mutable execution frame; additive resident credit, not immutable weights",
        "frame_slab_bytes": 117964800, "shared_lookup_bytes": 11264, "packer_workspace_bytes": 0,
        "global_host_credit_bytes": spec.global_host_bytes,
        "host_policy": "metadata/codebook only; no host packed-bank cache",
        "packing_policy": "current-layer active valid owned experts; no host readback",
        "frame_ownership": "exclusive decoder worker/current stream; no reentry",
        "original_raw_retained": True, "baseline_grouped_gate_up_rows": 4,
        "tables": ("empty", "iq2", "iq3"), "format_banks": dict(Counter(map(str, kinds))),
        "qualification": "private-unqualified"}
    w = NS(meta={"onecat_experts": owner, "owned_expert_plan": plan}, layers=layers, mtp=None)
    engine = NS(w=w, _owned_expert_plan=plan, _onecat_owner=owner,
        _owned_expert_gate_up_requested=spec, tp=8, rank=rank, depth=0, expert_partition=partition,
        capacity_plan={"onecat_gu_weights": owner.profile()},
        multi=NS(w=w, owned_expert_resources={"onecat_weight_residency": owner.profile()}))
    return engine, spec


class FrameBinding(unittest.TestCase):
    def test_baseline_import_isolation_and_budget_refusal(self):
        original = builtins.__import__
        def reject(name, *args, **kwargs):
            if name.startswith("tensorfold"):
                raise AssertionError("baseline imported runtime")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=reject):
            self.assertIsNone(tool.create_capability(None))
            self.assertEqual(tool.bound_receipt(NS(), None), {})
            for kwargs in ({"global_host_bytes": 1}, {"device_budget_bytes": 1}):
                with self.assertRaises(ValueError):
                    tool.create_capability(None, **kwargs)

    def test_explicit_methods_and_budgets(self):
        from tensorfold.families.qwen4_exp.cuda.onecat_experts import METHODS, OneCatGU
        for method in METHODS:
            e, selected = fixture(method=method)
            self.assertIs(type(selected), OneCatGU)
            self.assertEqual(tool.bound_receipt(e, selected)["record"]["profile"]["method"], method)
        for device, host in ((None, None), (True, 1), (1, 0), (1, -1), (sys.maxsize + 1, 1)):
            with self.subTest(device=device, host=host), self.assertRaises(ValueError):
                tool.create_capability(METHODS[0], device_budget_bytes=device, global_host_bytes=host)
        with self.assertRaises(ValueError):
            tool.create_capability("invented", device_budget_bytes=1, global_host_bytes=1)

    def test_all_eight_partitions_use_local_pointer_not_common_identity(self):
        records = []
        for rank in range(8):
            e, s = fixture(rank)
            record = tool.bound_receipt(e, s)
            self.assertEqual(record["rank"], rank)
            self.assertEqual(len(record["record"]["raw_bindings"]), 48)
            self.assertNotIn("source_planes", record["record"])
            self.assertIn("no invocation", record["qualification"])
            self.assertIn("not persistent/cross-rank/content identity", record["storage_pointer_scope"])
            records.append(record)
        self.assertEqual(len({r["record"]["frame"]["storage_pointer_local_only"] for r in records}), 8)
        self.assertTrue(all(r["record"]["profile"] == records[0]["record"]["profile"] for r in records))

    def test_unrequested_selection_surfaces_refused(self):
        mutations = [lambda e: setattr(e, "_onecat_owner", object()),
            lambda e: e.w.meta.update(onecat_experts=object()),
            lambda e: e.capacity_plan.update(onecat_gu_weights={}),
            lambda e: setattr(e, "_owned_expert_plan", NS(_onecat=True)),
            lambda e: e.w.meta.update(owned_expert_plan=NS(_onecat=True)),
            lambda e: e.capacity_plan.update(owned_expert_native={"onecat_weight_residency": {}}),
            lambda e: e.w.layers.append(NS(moe=NS(experts=NS(_onecat_provider=object()))))]
        for mutate in mutations:
            e = NS(w=NS(meta={}, layers=[]), capacity_plan={})
            mutate(e)
            with self.assertRaises(ValueError):
                tool.bound_receipt(e, None)

    def test_owner_provider_retirement_and_packer_fail_closed(self):
        mutations = [lambda e: e.w.meta.update(onecat_experts=object()),
            lambda e: setattr(e._onecat_owner.lease, "enabled", False),
            lambda e: setattr(e._onecat_owner.lease, "retiring", True),
            lambda e: setattr(e._onecat_owner.lease, "closed", True),
            lambda e: setattr(e._onecat_owner, "complete", False),
            lambda e: setattr(e._onecat_owner, "native_error", RuntimeError("prepare failed")),
            lambda e: setattr(e, "depth", 1), lambda e: setattr(e, "depth", False),
            lambda e: setattr(e.w.layers[0].moe.experts._onecat_provider, "raw_gate", object()),
            lambda e: setattr(e.w.layers[0].moe.experts._onecat_provider, "packer", object()),
            lambda e: setattr(e.w.layers[0].moe.experts._onecat_provider, "gate_sink", object()),
            lambda e: e._onecat_owner.bound.__setitem__(47, False)]
        for mutate in mutations:
            e, s = fixture()
            mutate(e)
            with self.assertRaises(ValueError):
                tool.bound_receipt(e, s)

    def test_old_owner_and_retained_plane_schema_rejected(self):
        from tensorfold.families.qwen4_exp.cuda.onecat_experts import OneCatOwner
        e, s = fixture()
        e._onecat_owner = object.__new__(OneCatOwner)
        with self.assertRaises(ValueError):
            tool.bound_receipt(e, s)
        e, s = fixture()
        with patch.object(type(e._onecat_owner), "record", return_value={"source_planes": (), "source_planes_sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "schema"):
                tool.bound_receipt(e, s)

    def test_raw_descriptor_digest_order_and_byte_mutants(self):
        for field, value in (("layer", 1), ("kind", 18), ("gate_source_descriptor_sha256", "0" * 64),
                             ("up_source_descriptor_sha256", "malformed"), ("gate_raw_bytes", 0),
                             ("down_column_offset", 1), ("source_scope", "GPU content verified")):
            e, s = fixture()
            e._onecat_owner.raw_bindings[0][field] = value
            with self.assertRaisesRegex(ValueError, "descriptor"):
                tool.bound_receipt(e, s)

    def test_frame_shape_offset_storage_and_pointer_mutants(self):
        mutations = [lambda o: setattr(o.frame, "shape", (117964799,)),
            lambda o: setattr(o.frame, "pointer", True),
            lambda o: setattr(o.frame, "offset", 1),
            lambda o: setattr(o.tables["iq2"], "pointer", o.frame.pointer),
            lambda o: setattr(o.views[18][0].codes, "pointer", 42),
            lambda o: setattr(o.views[18][1].scales, "offset", 0),
            lambda o: setattr(o.views[18][0].codes, "shape", (255, 153600)),
            lambda o: setattr(o.tables["iq2"], "shape", (8191,))]
        for mutate in mutations:
            e, s = fixture()
            mutate(e._onecat_owner)
            with self.assertRaises(ValueError):
                tool.bound_receipt(e, s)

    def test_profile_exact_types_and_loading_observations(self):
        for key, value in (("additional_device_bytes", 4056689664), ("original_raw_retained", 1),
                           ("frame_slab_bytes", 1), ("packer_workspace_bytes", 1), ("banks", 47)):
            e, s = fixture()
            e._onecat_owner.forecast_record[key] = value
            with self.assertRaises(ValueError):
                tool.bound_receipt(e, s)
        for bad in (True, -1, None, sys.maxsize + 1):
            e, s = fixture()
            e._onecat_owner.phase_after["free"] = bad
            with self.assertRaises(ValueError):
                tool.bound_receipt(e, s)

    def test_receipt_data_cannot_mutate_owner(self):
        e, s = fixture()
        record = tool.bound_receipt(e, s)
        record["record"]["raw_bindings"][0]["kind"] = 0
        record["record"]["profile"]["format_banks"]["18"] = 0
        self.assertEqual(e._onecat_owner.raw_bindings[0]["kind"], 22)
        self.assertEqual(e._onecat_owner.forecast_record["format_banks"]["18"], 17)

    def test_origin_error_preserves_owner_and_propagates(self):
        e, s = fixture()
        e._onecat_owner.packer.module.__file__ = str(ROOT / "absent-onecat-native-file.so")
        with self.assertRaises(FileNotFoundError):
            tool.bound_receipt(e, s)
        self.assertTrue(e._onecat_owner.lease.enabled)
        self.assertFalse(e._onecat_owner.lease.closed)


if __name__ == "__main__":
    unittest.main()
