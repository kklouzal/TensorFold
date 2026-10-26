"""Progressive pooling boundaries against frozen complete-history/scalar oracles."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tensorfold.cuda.sm70 import family_ops as ops

pytestmark = pytest.mark.torch


def _legacy_pool(raw, pooled, position, weight, frequency, eps, ratio, rows, *,
                 rope=None, delta=None, length=0, sections=(11, 11, 10), scale=1.0):
    """Frozen old full-history calculation: independent of progressive indexing.

    Its full allocation is intentional reference-only evidence. Unchanged RMS
    and rotary primitives remain their own numerical authorities; this oracle
    retains the original full-row reduction shape and complete update mask.
    """
    count, dims = raw.shape[0] // ratio, raw.shape[1]
    blocks = torch.arange(count, device=raw.device)
    values = raw[:count * ratio].reshape(count, ratio, dims).float()
    total = values[:, 0]
    for index in range(1, ratio):
        total = total + values[:, index]
    values = (total / ratio).half()
    values = ops.rotary(ops.rms(values, weight, eps), frequency, blocks * ratio,
                        rope=rope, delta=delta, length=length, sections=sections, scale=scale)
    valid = ((blocks >= torch.div(position.reshape(-1)[0], ratio, rounding_mode="floor"))
             & ((blocks + 1) * ratio <= position.reshape(-1)[0] + rows))
    pooled[:count].copy_(torch.where(valid[:, None], values, pooled[:count]))


class _PoolWork(TorchDispatchMode):
    """Observe actual raw reads/conversions, rather than inspect source text."""
    def __init__(self, raw):
        self.raw = raw
        self.read_indices = []
        self.float_conversion_sizes = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function is torch.ops.aten.index_select.default and args[0] is self.raw:
            self.read_indices.append(args[2].clone())
        if (function is torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.float32
                and args[0].dtype == torch.float16):
            self.float_conversion_sizes.append(args[0].numel())
        return function(*args, **kwargs)


@pytest.mark.parametrize("dims", [64, 128, 256])
@pytest.mark.parametrize("start,rows", [(0, 1), (3, 1), (17, 37), (262000, 37), (262145, 3)])
def test_long_history_exact_old_pool_results_without_reading_or_converting_full_history(dims, start, rows, monkeypatch):
    monkeypatch.setattr(ops, "_fused", lambda value: None)
    capacity, ratio = 262148, 4
    raw = torch.full((capacity, dims), float("nan"), dtype=torch.float16)
    first = start // ratio * ratio
    generator = torch.Generator().manual_seed(701 + start)
    raw[first:start + rows].copy_(torch.randn((start + rows - first, dims), generator=generator).half())
    pooled = torch.full(((capacity + ratio - 1) // ratio, dims), float("nan"), dtype=torch.float16)
    expected = pooled.clone()
    weight = torch.linspace(0.5, 1.5, dims)
    frequency = torch.tensor([1e-4, 3e-5, 9e-7, 7e-8])
    position = torch.tensor([start], dtype=torch.int64)
    delta = torch.tensor([11], dtype=torch.int32)
    _legacy_pool(raw, expected, position, weight, frequency, 1e-6, ratio, rows, delta=delta, scale=1.25)
    with _PoolWork(raw) as work:
        ops.qsa_pool(raw, pooled, position, weight, frequency, 1e-6, NS(ratio=ratio), rows,
                     delta=delta, rope_scale=1.25)
    assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))
    for read in work.read_indices:
        assert int(read.min()) >= first and int(read.max()) < start + rows
    assert work.float_conversion_sizes
    assert max(work.float_conversion_sizes) <= max(16, ((rows + ratio - 1) // ratio) * ratio) * dims


def test_scalar_pooling_average_rounding_and_incomplete_groups():
    raw = torch.full((17, 128), float("nan"), dtype=torch.float16)
    raw[:4] = torch.tensor([1, 3, 5, 7], dtype=torch.float16)[:, None]
    raw[4:8] = torch.tensor([-2, -4, -6, -8], dtype=torch.float16)[:, None]
    pooled = torch.full((5, 128), 19.0, dtype=torch.float16)
    # Scalar oracle: means4 and-5; every coordinate is equal, so the RMS
    # inverse is 1/sqrt(mean^2+eps), then stored FP16. Zero frequencies make
    # rotary exactly identity. No raw future row may influence either group.
    ops.qsa_pool(raw, pooled, torch.tensor([3]), torch.ones(128), torch.zeros(4), 1e-6, NS(ratio=4), 5)
    expected = torch.tensor([4.0, -5.0]) * torch.rsqrt(torch.tensor([16.0, 25.0]) + 1e-6)
    assert torch.equal(pooled[:2], expected.half()[:, None].expand(2, 128))
    assert torch.equal(pooled[2:], torch.full((3, 128), 19.0, dtype=torch.float16))


@pytest.mark.parametrize("capacity", [15, 16, 17, 63, 64, 65])
def test_append_resume_and_cancel_recompute_only_new_complete_groups(capacity):
    ratio, dims = 4, 128
    raw = torch.full((capacity, dims), float("nan"), dtype=torch.float16)
    pooled = torch.full(((capacity + 3) // 4, dims), 27.0, dtype=torch.float16)
    expected = pooled.clone()
    generator = torch.Generator().manual_seed(capacity)
    values = torch.randn((capacity, dims), generator=generator).half()
    weight, frequency = torch.ones(dims), torch.tensor([1e-3, 1e-5])
    start = 0
    for requested in (3, 5, 1, 7, 4, 2, 11, capacity):
        rows = min(requested, capacity - start)
        if not rows:
            break
        raw[start:start + rows].copy_(values[start:start + rows])
        position = torch.tensor([start])
        _legacy_pool(raw, expected, position, weight, frequency, 1e-6, ratio, rows)
        ops.qsa_pool(raw, pooled, position, weight, frequency, 1e-6, NS(ratio=ratio), rows)
        assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))
        start += rows
    # Restore a prefix ending inside a group, discard its tentative suffix,
    # then append a different continuation crossing that group's boundary.
    start = min(7, capacity - 4)
    rows = min(6, capacity - start)
    raw[start:start + rows].copy_(-values[start:start + rows])
    before = pooled.clone()
    _legacy_pool(raw, expected, torch.tensor([start]), weight, frequency, 1e-6, ratio, rows)
    ops.qsa_pool(raw, pooled, torch.tensor([start]), weight, frequency, 1e-6, NS(ratio=ratio), rows)
    assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(pooled[:start // ratio], before[:start // ratio])
    # A cancelled zero-row append performs no raw reads or output mutation.
    before = pooled.clone()
    with _PoolWork(raw) as work:
        ops.qsa_pool(raw, pooled, torch.tensor([start]), weight, frequency, 1e-6, NS(ratio=ratio), 0)
    assert not work.read_indices and not work.float_conversion_sizes
    assert torch.equal(pooled, before)


def test_image_axes_and_delta_keep_original_pool_rotary_positions():
    raw = torch.arange(33 * 128, dtype=torch.float32).remainder(29).sub(14).half().reshape(33, 128)
    pooled = torch.full((9, 128), 13.0, dtype=torch.float16)
    expected = pooled.clone()
    rope = torch.stack((torch.arange(33), torch.arange(33) * 2, torch.arange(33) * 3), dim=1)
    options = dict(rope=rope, delta=torch.tensor([7]), length=21, sections=(2, 2, 2))
    weight, frequency, position = torch.linspace(0.75, 1.25, 128), torch.arange(6) / 11, torch.tensor([15])
    _legacy_pool(raw, expected, position, weight, frequency, 1e-6, 4, 13, **options)
    ops.qsa_pool(raw, pooled, position, weight, frequency, 1e-6, NS(ratio=4), 13, **options)
    assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("live", [0, 1, 3, 7])
def test_live_prefix_never_publishes_padded_future_groups(live):
    start, rows, capacity, dims = 5, 7, 35, 128
    raw = torch.full((capacity, dims), float("nan"), dtype=torch.float16)
    raw[4:start + live].fill_(3)
    pooled = torch.full((9, dims), 17.0, dtype=torch.float16)
    expected = pooled.clone()
    position, weight, frequency = torch.tensor([start]), torch.ones(dims), torch.zeros(4)
    _legacy_pool(raw, expected, position, weight, frequency, 1e-6, 4, live)
    ops.qsa_pool(raw, pooled, position, weight, frequency, 1e-6, NS(ratio=4), rows,
                 valid_rows=torch.tensor([live], dtype=torch.int32))
    assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("start,live", [(-1, 3), (33, 3), (4, -1), (4, 8)])
def test_invalid_pool_frame_fails_before_any_pooled_mutation(start, live):
    raw = torch.ones((35, 128), dtype=torch.float16)
    pooled = torch.full((9, 128), 17.0, dtype=torch.float16)
    before = pooled.clone()
    controls = (raw, pooled, torch.tensor([start]), torch.ones(128), torch.zeros(4), 1e-6, NS(ratio=4), 7)
    with pytest.raises(ValueError, match="live frame"):
        ops.qsa_pool(*controls, valid_rows=torch.tensor([live], dtype=torch.int32))
    assert torch.equal(pooled, before)
    status = torch.tensor([0, -2], dtype=torch.int32)
    ops.qsa_pool(*controls, valid_rows=torch.tensor([live], dtype=torch.int32), status=status, layer=5)
    assert status.tolist() == [16, 5]
    assert torch.equal(pooled, before)


@pytest.mark.parametrize("format_name", ["fp16", "int8", "rotorquant-signed128-4",
                                        "rotorquant-signed64-norm4", "rotorquant-planar4-sm70-v1"])
def test_every_admitted_sm70_state_uses_the_same_progressive_indexer_contract(format_name):
    from tests.test_sm70_rotorquant_state import model
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = model(8)
    weights.meta.update(full=NS(kv_heads=2), rank=0, execution_provider="sm70", history_layout="unique")
    state = State(weights, 35, 4, format_name)
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging

    raw, pooled = torch.zeros((35, 128), dtype=torch.float16), state.pooled[0]
    raw[:16].copy_(torch.arange(16 * 128).remainder(23).sub(11).half().reshape(16, 128))
    coordinates = torch.arange(35, dtype=torch.int64)[:, None].expand(-1, 3)
    staging = Staging(13, state.layers, "cpu", mtp=False)
    staging.begin(mtp=False)
    state.compact_indexer.append(0, 0, raw[:3], coordinates[:3], staging, 0, 3, mtp=False, expected=(0, 0))
    state.set_pos(3)
    pooled.fill_(17)
    expected = pooled.clone()
    options = (torch.tensor([3]), torch.linspace(0.75, 1.25, 128), torch.tensor([0.03, 0.001]), 1e-6)
    _legacy_pool(raw, expected, *options, 4, 13)
    staging.begin(mtp=False)
    entry = state.compact_indexer.append(0, 3, raw[3:16], coordinates[3:16], staging, 0, 13,
                                          mtp=False, expected=(0, 0))
    entry.pool(pooled, options[1], options[2], options[3], (11, 11, 10), 1.0)
    assert state.ikc[0].shape == (4, 128)
    assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))
