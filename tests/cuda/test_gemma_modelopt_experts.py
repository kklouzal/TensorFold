"""Gemma DIRECT ModelOpt A4 against independent SK1 projection composition.

The optional real case requires the publisher-SHA-verified immutable snapshot
of llmfan46's 26B export (a9e81e26ed81f541ec1d7314e72c61237a24566f).
Only layer0 is packed; only four experts have separate reference projections.
These are arithmetic/adapter gates, not original-BF16 quality measurements.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modelopt_a4_inputs import PREFIX, Reader

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Gemma ModelOpt A4 requires CUDA SM12", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from tensorfold.cuda.nvfp4.modelopt_experts import load_modelopt_a4
from tensorfold.families.gemma4.cuda.modelopt_experts import ModelOptExperts
from tensorfold.families.gemma4.cuda.nvfp4_experts import geglu


def _projection_reference(source, expert, *, prefix):
    """Independent existing single-linear pack/matmul; original DIRECT scalars."""
    result = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{prefix}.experts.{expert}.{projection}"
        weight = source.get(base + ".weight")
        scales = source.get(base + ".weight_scale")
        # No inverse roundtrip: retain the stored F32 dequantization globals.
        weight_scale = float(source.get(base + ".weight_scale_2"))
        act = float(source.get(base + ".input_scale"))
        result.append(Fp4Linear.from_checkpoint(weight.cuda(), scales.cuda(), weight_scale, act=act))
    return result


def _reference(linears, x, picks, weights, *, activation=geglu):
    # Default factory activation is explicit, but the oracle below uses native
    # eager GELU directly so it cannot inherit a production activation defect.
    assert activation is geglu
    output = torch.zeros_like(x)
    for row, ids in enumerate(picks.cpu().tolist()):
        for slot, expert in sorted(enumerate(ids), key=lambda pair: pair[1]):
            gate, up, down = linears[expert]
            g = checkpoint.prompt(checkpoint.A4, x[row:row + 1], gate)
            u = checkpoint.prompt(checkpoint.A4, x[row:row + 1], up)
            hidden = torch.nn.functional.gelu(g, approximate="tanh") * u
            value = checkpoint.prompt(checkpoint.A4, hidden, down)
            term = (value.float() * weights[row, slot]).to(torch.bfloat16)
            output[row:row + 1] = output[row:row + 1] + term
    return output


def _exercise(source, cfg, selected, *, prefix=PREFIX):
    bank = ModelOptExperts(load_modelopt_a4(source, prefix, cfg, device="cuda:0", swiglu=geglu))
    assert bank.bank.direct_input and bank.bank.swiglu is geglu
    # Check every original direct scalar, not just the reference subset.
    for expert in range(cfg.experts):
        for p, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            base = f"{prefix}.experts.{expert}.{projection}"
            for suffix, field in (("weight_scale_2", 2), ("input_scale", 4)):
                expected = source.get(base + "." + suffix).reshape(1).view(torch.int32)
                stored = bank.bank.fields[field][expert, p:p + 1] if p < 2 else bank.bank.fields[field + 1][expert]
                assert torch.equal(stored.cpu().reshape(1).view(torch.int32), expected)
    refs = {expert: _projection_reference(source, expert, prefix=prefix) for expert in selected}
    generator = torch.Generator().manual_seed(20261101)
    x = (torch.randn((3, cfg.hidden), generator=generator) * .3).cuda().bfloat16()
    x[0] = .5
    orders = [selected[::-1], selected[1:] + selected[:1], selected[::2] + selected[1::2]]
    picks = torch.tensor(orders, device="cuda", dtype=torch.int32)
    weights = torch.linspace(.03125, .812345, len(selected), device="cuda", dtype=torch.float32)
    weights = weights[None].expand(3, -1).contiguous()
    expected = _reference(refs, x, picks, weights)
    for rows in (1, 3):
        actual = bank.forward(x[:rows], picks[:rows], weights[:rows])
        assert torch.equal(actual, expected[:rows])
    serial = torch.cat([bank.forward(x[r:r + 1], picks[r:r + 1], weights[r:r + 1]) for r in range(3)])
    assert torch.equal(serial, expected)
    reverse = torch.tensor([2, 0, 1], device="cuda")
    assert torch.equal(bank.forward(x[reverse], picks[reverse], weights[reverse]), expected[reverse])
    # Change slot order while preserving each expert's original FP32 weight.
    slot_order = torch.arange(len(selected) - 1, -1, -1, device="cuda")
    assert torch.equal(bank.forward(x, picks[:, slot_order], weights[:, slot_order]), expected)
    return bank, refs


@pytest.mark.parametrize("shared", [False, True])
def test_synthetic_direct_scales_geglu_and_ordered_combine(shared):
    source = Reader(count=3)
    if shared:
        scale = source.payload[PREFIX + ".experts.0.gate_proj.input_scale"]
        for name, value in source.payload.items():
            if name.endswith(("gate_proj.input_scale", "up_proj.input_scale")):
                value.copy_(scale)
    owners = _exercise(source, source.cfg, [0, 1, 2])
    torch.cuda.synchronize()
    del owners


def test_real_26b_layer0_direct_scales_geglu_and_ordered_combine():
    directory = os.environ.get("TF_GEMMA_MODELOPT_TEST_MODEL")
    if not directory:
        pytest.skip("set TF_GEMMA_MODELOPT_TEST_MODEL to the verified pinned 26B snapshot")
    from tensorfold.checkpoint_metadata import read_config, read_json
    from tensorfold.cuda.direct_read import SafeTensors
    from tensorfold.tensor_file import checkpoint_path
    from tensorfold.families.gemma4.cuda.modelopt_weights import validate_headers

    root = Path(directory)
    raw = read_config(root)
    index = read_json(root, "model.safetensors.index.json")
    assert isinstance(index, dict) and isinstance(index.get("weight_map"), dict)
    mapping = index["weight_map"]
    assert mapping and all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())
    files = [checkpoint_path(root, name) for name in sorted(set(mapping.values()))]
    source = SafeTensors([])
    primary = None
    try:
        for file in files:
            shard = SafeTensors([file], reader=source.reader)
            assert not set(shard.where).intersection(source.where)
            source.where.update(shard.where)
        assert set(source.where) == set(mapping)
        assert all(source.where[name][0] == checkpoint_path(root, file) for name, file in mapping.items())
        headers = {name: {"dtype": value[3], "shape": value[4]} for name, value in source.where.items()}
        cfg = validate_headers(raw, headers, vision=True)
        assert (cfg.layers, cfg.experts, cfg.moe_width, cfg.hidden) == (30, 128, 704, 2816)

        class ProjectionSource:
            def info(self, name):
                return headers[name]

            def get(self, name):
                return source.get(name)

            def get_rows(self, name, lo, hi, device="cpu"):
                return source.get(name, device=device)[lo:hi]

        prefix = "model.language_model.layers.0"
        assert prefix + ".experts.0.gate_proj.weight" in headers
        assert PREFIX + ".experts.0.gate_proj.weight" not in headers
        owners = _exercise(ProjectionSource(), SimpleNamespace(experts=128, moe_width=704, hidden=2816),
                           [0, 17, 63, 127], prefix=prefix)
    except BaseException as error:
        primary = error
        raise
    finally:
        # Do not close the reader after failed CUDA retirement. This bounded
        # test is process-terminal on such failure; Root owns process teardown.
        from tensorfold.cleanup import raise_failures

        try:
            torch.cuda.synchronize()
        except BaseException as retirement:
            raise_failures(primary, [retirement])
        try:
            source.close()
        except BaseException as close_error:
            raise_failures(primary, [close_error])
    del owners
