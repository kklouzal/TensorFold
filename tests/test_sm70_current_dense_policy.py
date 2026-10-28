"""Distinct current Dense State epoch and worker dispatch compatibility."""
import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy, source = family.policy, family.source
PROFILE = prefix_policy.OWNER_TARGET_DENSE_PROFILE

def test_cache_epoch_is_distinct_and_only_existing_greedy_int8_text_domain_is_borrowed():
    profiles = ("reference", "optimized", "source-family-trial", "source-family-prefix-trial", "source-packed-trial", prefix_policy.OWNER_TARGET_PRELUDE_PROFILE)
    assert all(prefix_policy.identity(PROFILE) != prefix_policy.identity(old) for old in profiles)
    assert all(prefix_policy.state_epoch(PROFILE) != prefix_policy.state_epoch(old) for old in profiles)
    weights = NS(meta={"execution_provider": "sm70", "sm70_profile": PROFILE})
    for dtype, temperature, vision, draft, expected in (
            ("int8", 0, None, True, True), ("fp16", 0, None, True, False),
            ("int8", .7, None, True, False), ("int8", 0, object(), True, False),
            ("int8", 0, None, False, False)):
        kwargs = dict(draft=draft, vision=vision, position=24, kv_pair=get_pair(dtype))
        sampling = Sampling(17, temperature)
        assert prefix_policy.reuse_allowed(weights, sampling, **kwargs) is expected
        assert prefix_policy.serial_reuse_allowed(weights, sampling, **kwargs) is expected


@pytest.mark.parametrize("selected_profile", [PROFILE, prefix_policy.DENSE_INDEPENDENT_PROFILE])
def test_worker_handshake_seals_actual_dense_selection_and_keeps_old_profile_packet(monkeypatch, selected_profile):
    from .test_flashnext_stochastic_prefix import method
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

    torch = family.torch
    precision = NS(fp8=lambda: False, same_on_ranks=lambda a, b: None)
    agree = method(FlashNextEngine, "_same_settings", {"json": json, "prompt_precision": precision})
    proxy = NS(int64=torch.int64, tensor=lambda data, **kw: torch.tensor(data, dtype=kw["dtype"]),
               empty=lambda shape, **kw: torch.empty(shape, dtype=kw["dtype"]), equal=torch.equal)
    packets, stale = [], []

    def gather(value, destination):
        packets.append(value.clone())
        destination.copy_(value.repeat(8))
        if stale:
            destination.view(8, -1)[1, 9] = stale[0]

    engine = NS(execution_provider="sm70", sm70_profile="source-packed-trial", depth=2, confidence=.7,
        max_len=262144, streams=26, graphs_enabled=False, prefill_rows=256, tp=8,
        rope=NS(metadata=lambda: {"rope_type": "default"}), kv_pair=NS(handshake=0), comm=NS(all_gather=gather))
    monkeypatch.setattr(GGUFDense, "implementation", "reference")
    agree(engine, proxy, None)
    old_packet = packets[-1].clone()
    monkeypatch.setattr(GGUFDense, "implementation", "native32")
    agree(engine, proxy, None)
    assert torch.equal(old_packet, packets[-1])
    engine.sm70_profile = selected_profile
    agree(engine, proxy, None)
    native_digest = int(packets[-1][9])
    monkeypatch.setattr(GGUFDense, "implementation", "reference")
    agree(engine, proxy, None)
    assert int(packets[-1][9]) != native_digest
    stale.append(native_digest)
    with pytest.raises(RuntimeError, match="different settings"):
        agree(engine, proxy, None)


@pytest.mark.parametrize("hidden", [256, 2560])
def test_new_tp8_oracle_preserves_actual_toy_embedding_and_stock_mtp_aliases(monkeypatch, hidden):
    import ast
    from dataclasses import fields, is_dataclass, replace
    from pathlib import Path
    import sys
    from types import ModuleType
    import numpy as np
    from tensorfold.families.qwen4_exp.cuda import gdn
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
    from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense, DenseExperts

    fixture_directory = Path(__file__).parent / "cuda"

    def actual_function(path, name, namespace):
        tree = ast.parse(path.read_text())
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]

    class CpuAllocation:
        # Run the exact fixture construction bodies; only route their explicit
        # CUDA allocation requests to CPU. No forward math or GPU claim occurs.
        def __getattr__(self, name):
            operation = getattr(family.torch, name)
            if name not in ("randn", "ones", "zeros", "tensor"):
                return operation

            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                return operation(*args, **kwargs)

            return allocate

    cpu = CpuAllocation()
    small = actual_function(fixture_directory / "test_flashnext_sm70_family.py", "_small_weights",
                            dict(torch=cpu, gdn=gdn, np=np))
    module = ModuleType("test_flashnext_sm70_family")
    module._small_weights = small
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stock = actual_function(fixture_directory / "test_flashnext_sm70_mtp.py", "_weights_with_stock_mtp_protocol",
                            dict(torch=cpu, Dense=Dense, DenseExperts=DenseExperts, replace=replace))
    original = stock(head_dim=256, hidden=hidden, lowrank=320 if hidden == 2560 else 32)
    assert original.cfg.hidden == hidden
    original.layers.append(replace(original.layers[1], index=original.cfg.layers))
    assert isinstance(original.embed, tuple) and len(original.embed) == 1
    assert hasattr(original.embed[0], "lookup")
    converter = actual_function(fixture_directory / "test_sm70_owner_piece_graphs_tp8.py", "actual_target_dense",
                                dict(torch=family.torch, GGUFDense=GGUFDense, converted={},
                                     fields=fields, is_dataclass=is_dataclass, replace=replace))
    actual = converter(original)
    assert actual.embed is original.embed
    assert actual.cfg is original.cfg
    assert actual.mtp.fc_e is original.mtp.fc_e and actual.mtp.fc_h is original.mtp.fc_h
    assert actual.head.weight is original.head.weight
    assert isinstance(actual.head, GGUFDense)
    assert actual.layers[1].attn.proj is actual.layers[2].attn.proj
    unchanged_list = [original.embed, original.mtp.fc_e]
    assert converter(unchanged_list) is unchanged_list


@pytest.mark.parametrize("failure", [False, True])
def test_dense_kernel_preparation_precedes_selector_publication_and_propagates_failure(source, monkeypatch, failure):
    from tensorfold.cuda.sm70 import gguf_ops, greedy_native, family_native

    module = source.module
    monkeypatch.setenv(module.PROFILE_ENV, PROFILE)
    before = family.selectors(source.policy), source.policy.weights.GGUFDense.implementation, module._SELECTED
    monkeypatch.setattr(family_native, "extension", lambda: source.calls.append("family") or NS(
        prepare_value_guard=lambda: source.calls.append("guard"),
        prepare_owner_prelude=lambda: source.calls.append("prelude")))

    def prepare():
        assert (family.selectors(source.policy), source.policy.weights.GGUFDense.implementation,
                module._SELECTED) == before
        source.calls.append("dense-residency")
        if failure:
            raise RuntimeError("dense function residency failed")
        return 4

    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("gguf") or NS(prepare_dense32=prepare))
    monkeypatch.setattr(greedy_native, "extension", lambda: source.calls.append("greedy"))
    if failure:
        with pytest.raises(RuntimeError, match="dense function residency failed"):
            module.configure(source.text, tp=8, torch=family.torch)
        assert (family.selectors(source.policy), source.policy.weights.GGUFDense.implementation,
                module._SELECTED) == before
        assert source.calls == ["audit", "family", "guard", "prelude", "gguf", "dense-residency"]
    else:
        assert module.configure(source.text, tp=8, torch=family.torch) == PROFILE
        assert source.calls == ["audit", "family", "guard", "prelude", "gguf", "dense-residency", "greedy", "audit"]
        assert source.policy.weights.GGUFDense.implementation == "native32"
