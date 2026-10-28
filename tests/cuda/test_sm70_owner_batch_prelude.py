"""Batched descriptor prep against independent per-State Torch preparation."""
import pytest
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                               reason="requires actual SM70")


@pytest.mark.parametrize("lengths", [(1,), (1,) * 24, (3,) * 8, (1, 2, 3, 9, 17), (128, 128)])
@pytest.mark.parametrize("independent_rows", [False, True])
@pytest.mark.parametrize("adverse", [False, True])
def test_descriptor_batch_matches_independent_per_state_torch_stored_bits(lengths, independent_rows, adverse):
    from unittest.mock import patch
    from tensorfold.cuda.sm70.family_native import extension
    from tensorfold.cuda.sm70 import family_ops as ops

    native = extension()
    assert native.prepare_owner_batch() == 1
    torch.manual_seed(18419)
    rows, qh, dim, ih, di = sum(lengths), 3, 256, 4, 128
    projection = torch.randn((rows, 2688), dtype=torch.float16, device="cuda") * 3
    if adverse:
        projection[0, 0] = float("nan")
        projection[-1, 1792:2048].fill_(65504)
        projection[-1, -1] = float("inf")
    norm = [torch.randn(width, device="cuda") * .2 + 1 for width in (dim, dim, di)]
    frequency = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    positions, statuses, records, references, first = [], [], [], [], 0
    for index, count in enumerate(lengths):
        absolute = 262144 - count if index % 2 else 273 + index * 19
        position = torch.tensor([absolute], dtype=torch.int32, device="cuda")
        status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
        positions.append(position)
        statuses.append(status)
        records.extend([(position.data_ptr(), status.data_ptr(), first, count)] * count)
        q = torch.empty((count, qh, dim), dtype=torch.float16, device="cuda")
        iq = torch.empty((count, ih, di), dtype=torch.float16, device="cuda")
        key, value = [torch.empty((count, 1, dim), dtype=torch.int8, device="cuda") for _ in range(2)]
        ks, vs = [torch.empty((count, 1, 8), dtype=torch.float16, device="cuda") for _ in range(2)]
        ik = torch.empty((count, di), dtype=torch.float16, device="cuda")
        reference_status = status.clone()
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        with patch.object(ops, "_NATIVE_FAMILY", False):
            ops.attn_prep(projection[first:first + count], zero, *norm, frequency, q, key, value, iq, ik,
                1e-6, q_heads=qh, kv_heads=1, head_dim=dim, index_heads=ih, index_dim=di,
                ks=ks, vs=vs, bits=8, status=reference_status, delta=position, layer=11,
                independent_rows=independent_rows)
        references.append((q, key, value, iq, ik, ks, vs, reference_status))
        first += count
    descriptors = torch.tensor(records, dtype=torch.int64, device="cuda")
    q = torch.empty((rows, qh, dim), dtype=torch.float16, device="cuda")
    iq = torch.empty((rows, ih, di), dtype=torch.float16, device="cuda")
    key, value = [torch.empty((rows, 1, dim), dtype=torch.int8, device="cuda") for _ in range(2)]
    ks, vs = [torch.empty((rows, 1, 8), dtype=torch.float16, device="cuda") for _ in range(2)]
    ik = torch.empty((rows, di), dtype=torch.float16, device="cuda")
    allocated = torch.cuda.memory_allocated()
    native.owner_prep_batch(projection, descriptors, *norm, frequency, q, key, value, iq, ik, ks, vs,
                            11, 1e-6, 1., independent_rows)
    assert torch.cuda.memory_allocated() == allocated
    outputs = (q, key, value, iq, ik, ks, vs)
    for actual, expected in zip(outputs, zip(*[row[:7] for row in references])):
        reference = torch.cat(expected)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual, expected in zip(statuses, (row[7] for row in references)):
        assert torch.equal(actual, expected)

@pytest.mark.parametrize("failure", ["descriptor-shape", "descriptor-dtype", "input-shape",
                                     "output-shape", "key-input-alias", "output-alias", "norm-layout"])
def test_native_batch_boundary_rejects_malformed_geometry_before_writes(failure):
    from tensorfold.cuda.sm70.family_native import extension

    native = extension()
    rows = 2
    projection = torch.zeros((rows, 2688), dtype=torch.float16, device="cuda")
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    position = torch.tensor([37], dtype=torch.int32, device="cuda")
    descriptors = torch.tensor([(position.data_ptr(), status.data_ptr(), 0, rows)] * rows,
                               dtype=torch.int64, device="cuda")
    qw, kw, iw, freq = [torch.ones(width, device="cuda") for width in (256, 256, 128, 32)]
    q = torch.full((rows, 3, 256), 13, dtype=torch.float16, device="cuda")
    iq = torch.full((rows, 4, 128), 13, dtype=torch.float16, device="cuda")
    key, value = [torch.full((rows, 1, 256), 13, dtype=torch.int8, device="cuda") for _ in range(2)]
    ks, vs = [torch.full((rows, 1, 8), 13, dtype=torch.float16, device="cuda") for _ in range(2)]
    ik = torch.full((rows, 128), 13, dtype=torch.float16, device="cuda")
    if failure == "descriptor-shape":
        descriptors = descriptors[:, :3].contiguous()
    elif failure == "descriptor-dtype":
        descriptors = descriptors.to(torch.float64)
    elif failure == "input-shape":
        projection = projection[:, :2600].contiguous()
    elif failure == "output-shape":
        q = q[:1]
    elif failure == "key-input-alias":
        key = projection.view(torch.int8).reshape(-1)[:rows * 256].view(rows, 1, 256)
    elif failure == "output-alias":
        value = key
    else:
        qw = torch.ones(512, device="cuda")[::2]
    outputs = (projection, status, q, key, value, iq, ik, ks, vs)
    originals = [item.clone() for item in outputs]
    with pytest.raises(RuntimeError):
        native.owner_prep_batch(projection, descriptors, qw, kw, iw, freq,
                                q, key, value, iq, ik, ks, vs, 3, 1e-6, 1., True)
    for actual, expected in zip(outputs, originals):
        assert torch.equal(actual, expected)


def test_graph_replay_reads_live_per_state_positions_and_reuses_outputs():
    from tensorfold.cuda.sm70.family_native import extension

    native = extension()
    rows = 4
    projection = torch.linspace(-1., 1., rows * 2688, device="cuda", dtype=torch.float16).view(rows, 2688)
    positions = [torch.tensor([37], dtype=torch.int32, device="cuda") for _ in range(2)]
    statuses = [torch.tensor([0, -2], dtype=torch.int32, device="cuda") for _ in range(2)]
    descriptors = torch.tensor([(positions[0].data_ptr(), statuses[0].data_ptr(), 0, 1)] +
        [(positions[1].data_ptr(), statuses[1].data_ptr(), 1, 3)] * 3, dtype=torch.int64, device="cuda")
    qw, kw, iw, frequency = [torch.ones(width, device="cuda") for width in (256, 256, 128, 32)]
    q = torch.empty((rows, 3, 256), dtype=torch.float16, device="cuda")
    iq = torch.empty((rows, 4, 128), dtype=torch.float16, device="cuda")
    key, value = [torch.empty((rows, 1, 256), dtype=torch.int8, device="cuda") for _ in range(2)]
    ks, vs = [torch.empty((rows, 1, 8), dtype=torch.float16, device="cuda") for _ in range(2)]
    ik = torch.empty((rows, 128), dtype=torch.float16, device="cuda")
    def invoke():
        native.owner_prep_batch(projection, descriptors, qw, kw, iw, frequency,
                                q, key, value, iq, ik, ks, vs, 3, 1e-6, 1., True)
    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    positions[0].fill_(262143)
    positions[1].fill_(262140)
    invoke()
    expected = [item.clone() for item in (q, key, value, iq, ik, ks, vs)]
    graph.replay()
    for actual, original in zip((q, key, value, iq, ik, ks, vs), expected):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert all(item.tolist() == [0, -2] for item in statuses)
