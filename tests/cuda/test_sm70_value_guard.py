"""Actual SM70 guard: independent ATen oracle, strided owners, masks and graph replay."""
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                               reason="requires actual SM70")


def equal_bits(actual, expected):
    dtype = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual.contiguous().view(dtype), expected.contiguous().view(dtype))


def reference(value, status, flag, layer, live):
    valid = torch.isfinite(value) & (value.abs() <= 2**30)
    mask = torch.ones_like(valid) if live is None else (torch.arange(value.shape[0], device=value.device) <
        live.reshape(-1)[0]).reshape(value.shape[0], *([1] * (value.ndim - 1)))
    bad = (~valid & mask).any()
    status[0].bitwise_or_(bad.int() * flag)
    status[1].copy_(torch.where(bad & (status[1] == -2), layer, status[1]))
    return torch.where(valid, value, 0)


@pytest.fixture(scope="module")
def native():
    from tensorfold.cuda.sm70.family_native import extension

    module = extension()
    assert module.prepare_value_guard() == 1
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape,strided", [((0, 3, 256), False), ((1, 3, 256), True),
    ((3, 3, 256), True), ((32, 4, 128), True), ((256, 1, 256), False), ((7, 128), True)])
@pytest.mark.parametrize("live_count", [None, -1, 0, 1, 300])
def test_strided_guard_and_live_fault_latch_match_original_bits(native, dtype, shape, strided, live_count):
    rows, dims = shape[0], shape[-1]
    heads = shape[1] if len(shape) == 3 else 1
    source = torch.zeros((rows, heads * dims * 3 + 37), dtype=dtype, device="cuda")
    value = (source[:, :heads * dims * 2].view(rows, heads, 2, dims)[:, :, 0] if len(shape) == 3
             else source[:, 11:11 + dims]) if strided else torch.empty(shape, dtype=dtype, device="cuda")
    pattern = torch.tensor([0., -0., 1., -1., float("inf"), float("-inf"), float("nan"),
                            2**30, -(2**30), 2**31, 2**-24, -(2**-24)], dtype=dtype, device="cuda")
    if value.numel():
        value.copy_(pattern.repeat((value.numel() + pattern.numel() - 1) // pattern.numel())[:value.numel()].reshape(shape))
    raw = value.clone()
    slab = torch.full((rows, heads * dims + 43), 37, dtype=dtype, device="cuda")
    out = slab[:, 13:13 + heads * dims].view(shape)
    live = None if live_count is None else torch.tensor([live_count], dtype=torch.int32, device="cuda")
    for initial, flag, layer in [((0, -2), 1, 3), ((4, 11), 16, -1), ((0, -2), 0, 7)]:
        expected_status = torch.tensor(initial, dtype=torch.int32, device="cuda")
        expected = reference(value, expected_status, flag, layer, live)
        status = torch.tensor(initial, dtype=torch.int32, device="cuda")
        allocated = torch.cuda.memory_allocated()
        native.value_guard(value, out, status, flag, layer, live)
        assert torch.cuda.memory_allocated() == allocated
        equal_bits(out, expected)
        equal_bits(value, raw)
        assert torch.equal(status, expected_status)
        assert bool((slab[:, :13] == 37).all()) and bool((slab[:, 13 + heads * dims:] == 37).all())



def test_fp32_threshold_neighbors_exceptional_payloads_and_subnormal_bits(native):
    # Independently chosen IEEE binary32 encodings: signed zeros, smallest
    # subnormals, threshold predecessor/equal/successor, signed NaNs and infinities.
    patterns = torch.tensor([0, -2147483648, 1, -2147483647, 1317011455, 1317011456, 1317011457,
                             -830472193, -830472192, -830472191, 2143294004, -4194269,
                             2139095040, -8388608], dtype=torch.int32, device="cuda")
    value = patterns.view(torch.float32).view(2, 7)
    out = torch.empty_like(value)
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    expected_status = status.clone()
    expected = reference(value, expected_status, 4, 3, None)
    native.value_guard(value, out, status, 4, 3, None)
    equal_bits(out, expected)
    assert torch.equal(value.view(torch.int32).reshape(-1), patterns)
    assert torch.equal(status, expected_status)

def test_dynamic_live_graph_and_first_bad_layer_order(native):
    value = torch.ones((3, 2, 32), dtype=torch.float32, device="cuda")
    value[2, 1, 7] = float("nan")
    out = torch.empty_like(value)
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    live = torch.tensor([1], dtype=torch.int32, device="cuda")
    native.value_guard(value, out, status, 2, 9, live)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.value_guard(value, out, status, 2, 9, live)
        native.value_guard(value, out, status, 4, 12, live)
    for count, expected in [(1, [0, -2]), (3, [6, 9])]:
        live.fill_(count)
        status.copy_(torch.tensor([0, -2], dtype=torch.int32, device="cuda"))
        graph.replay()
        assert status.tolist() == expected and out[2, 1, 7].item() == 0


@pytest.mark.parametrize("failure", ["source-alias", "status-alias", "output-shape", "output-dtype",
                                      "transposed", "live-shape", "live-dtype"])
def test_bad_native_owner_geometry_refuses_before_status_or_output_writes(native, failure):
    value = torch.ones((3, 32), dtype=torch.float32, device="cuda")
    value[0, 0] = float("nan")
    out = torch.full_like(value, 42)
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    live = None
    if failure == "source-alias":
        out = value
    elif failure == "status-alias":
        status = value.view(torch.int32).reshape(-1)[:2]
    elif failure == "output-shape":
        out = out[:2]
    elif failure == "output-dtype":
        out = out.half()
    elif failure == "transposed":
        value, out = value.T, out.T
    elif failure == "live-shape":
        live = torch.zeros(2, dtype=torch.int32, device="cuda")
    else:
        live = torch.zeros(1, dtype=torch.int64, device="cuda")
    old_out, old_status = out.clone(), status.clone()
    with pytest.raises(RuntimeError):
        native.value_guard(value, out, status, 2, 1, live)
    equal_bits(out, old_out)
    assert torch.equal(status, old_status)
