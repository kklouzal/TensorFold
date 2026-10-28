"""Distinct current Dense State epoch and worker dispatch compatibility."""
import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

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


def test_worker_handshake_seals_actual_dense_selection_and_keeps_old_profile_packet(monkeypatch):
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
    engine.sm70_profile = PROFILE
    agree(engine, proxy, None)
    native_digest = int(packets[-1][9])
    monkeypatch.setattr(GGUFDense, "implementation", "reference")
    agree(engine, proxy, None)
    assert int(packets[-1][9]) != native_digest
    stale.append(native_digest)
    with pytest.raises(RuntimeError, match="different settings"):
        agree(engine, proxy, None)


def test_new_tp8_oracle_installs_actual_target_dense_and_preserves_stock_mtp_owner():
    import ast
    from dataclasses import dataclass, fields, is_dataclass, replace
    from pathlib import Path
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
    from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense

    tree = ast.parse((Path(__file__).parent / "cuda/test_sm70_owner_piece_graphs_tp8.py").read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "actual_target_dense")
    namespace = dict(torch=family.torch, GGUFDense=GGUFDense, converted={},
                     fields=fields, is_dataclass=is_dataclass, replace=replace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_tp8_dense_fixture", "exec"), namespace)

    @dataclass
    class Fixture:
        target: object
        mtp: object

    weight = family.torch.ones((4, 8), dtype=family.torch.float16)
    target = NS(kernel="gguf", weight=weight)
    stock = Dense(weight)
    actual = namespace["actual_target_dense"](Fixture([target, target], (stock, target)))
    assert isinstance(actual.target[0], GGUFDense)
    assert actual.target[0].weight is weight
    assert actual.mtp[0] is stock
    assert actual.target[0] is actual.target[1] is actual.mtp[1]
