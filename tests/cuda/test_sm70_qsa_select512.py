"""Candidate-only exact finite QSA selector; no score arithmetic/dispatch change."""
import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70.family_native import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='actual SM70 candidate qualification required')


@pytest.fixture(scope='module')
def native():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('candidate requires SM70')
    return extension()


def oracle(scores, lengths):
    """Independent numeric lexsort; signed zeros tie by logical index."""
    out = np.full((len(lengths), 512), -1, dtype=np.int32)
    status = np.zeros(len(lengths), dtype=np.int32)
    for row, length in enumerate(lengths):
        if not 0 <= int(length) <= scores.shape[1]:
            status[row] = 2
        elif not np.isfinite(scores[row, :length]).all():
            status[row] = 1
        else:
            chosen = np.lexsort((np.arange(length), -scores[row, :length].astype(np.float64)))[:512]
            out[row, :len(chosen)] = np.sort(chosen)
    return out, status


def invoke(native, cpu, lengths, *, padded=True):
    rows, columns = cpu.shape
    source = torch.full((rows + 2, columns + 11), 19., dtype=torch.float32, device='cuda')
    scores = source[1:rows + 1, 3:columns + 3]
    scores.copy_(torch.from_numpy(cpu))
    raw = source.view(torch.int32).clone()
    out_base = torch.full((rows + 2, 525), -777, dtype=torch.int32, device='cuda')
    output = out_base[1:rows + 1, 3:515] if padded else torch.empty((rows, 512), dtype=torch.int32, device='cuda')
    status_base = torch.full((rows + 2,), -99, dtype=torch.int32, device='cuda')
    status = status_base[1:rows + 1]
    ends = torch.tensor(lengths, dtype=torch.int32, device='cuda')
    assert native.qsa_select512(scores, ends, output, status) is None
    expected, flags = oracle(cpu, lengths)
    assert torch.equal(output.cpu(), torch.from_numpy(expected))
    assert torch.equal(status.cpu(), torch.from_numpy(flags))
    assert torch.equal(source.view(torch.int32), raw)  # Preserve every source bit, including NaN/zero suffixes.
    if padded:
        assert torch.all(out_base[0] == -777) and torch.all(out_base[-1] == -777)
        assert torch.all(out_base[:, :3] == -777) and torch.all(out_base[:, 515:] == -777)
    assert status_base[0] == -99 and status_base[-1] == -99
    return scores, ends, output, status


@pytest.mark.parametrize('rows', [0, 1, 5, 17, 256])
def test_full_context_row_bounds_and_cutoff_ties(native, rows):
    columns = 65537
    values = np.zeros((rows, columns), dtype=np.float32)
    counts = [0, 511, 512, 513, 1023, 1024, 1025, 65537]
    lengths = [counts[row % len(counts)] for row in range(rows)]
    invoke(native, values, lengths)
    if rows:
        invoke(native, values, [-1 if row % 2 else columns + 1 for row in range(rows)])


@pytest.mark.parametrize('columns', [1025, 65536, 65537])
def test_general_radix_numeric_order_extremes_zero_subnormals_and_many_pivot_ties(native, columns):
    rng = np.random.default_rng(741)
    values = rng.standard_normal((5, columns)).astype(np.float32)
    values[0] = 0
    values[0, 1::2] = -0.
    values[1] = np.floor(values[1] * 2)
    values[2] = -1
    values[2, :400] = 1
    values[2, 400:1000] = .5  # Pivot-equal set extends across compaction chunks at long widths.
    bits = np.array([0xff7fffff, 0x80800000, 0x80000001, 0x80000000,
                     0x00000000, 0x00000001, 0x00800000, 0x7f7fffff], dtype=np.uint32)
    values[3] = np.resize(bits.view(np.float32), columns)
    values[4] = np.resize(bits[::-1].view(np.float32), columns)
    invoke(native, values, [columns] * 5)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_valid_prefix_is_contained_and_suffix_ignored(native, bad):
    values = np.zeros((3, 65537), dtype=np.float32)
    values[0, 512] = bad
    values[1, 513:] = bad
    values[2, 0] = bad
    invoke(native, values, [513, 513, 1])


def test_actual_unchanged_fp32_score_output_matches_current_large_count_qsa_policy(native):
    from types import SimpleNamespace
    from tensorfold.cuda.sm70 import family_ops
    rows, columns, ratio, top, dim = 5, 65537, 4, 512, 128
    gen = torch.Generator(device='cuda').manual_seed(903)
    query = torch.randn((rows, 4, dim), device='cuda', dtype=torch.float16, generator=gen)
    pooled = torch.randn((columns, dim), device='cuda', dtype=torch.float16, generator=gen)
    scratch = SimpleNamespace(ratio=ratio, budget=ratio*top,
        sparse=torch.empty(rows, dtype=torch.int32, device='cuda'), nk=torch.empty(rows, dtype=torch.int32, device='cuda'),
        ids=torch.full((rows, ratio*(top+1)), -1, dtype=torch.int32, device='cuda'),
        scores=torch.empty((rows, columns), dtype=torch.float32, device='cuda'))
    position = torch.tensor([262144 - rows], dtype=torch.int32, device='cuda')
    # Existing dispatch always takes original Torch FP32 scores for count>4096.
    family_ops.qsa_rows(query, pooled, position, scratch, rows)
    lengths = torch.div(position + torch.arange(rows, device='cuda') + 1, ratio, rounding_mode='floor').int()
    output = torch.empty((rows, 512), dtype=torch.int32, device='cuda')
    status = torch.full((rows,), -1, dtype=torch.int32, device='cuda')
    saved = scratch.scores.view(torch.int32).clone()
    native.qsa_select512(scratch.scores, lengths, output, status)
    assert torch.all(status == 0)
    assert torch.equal(output, scratch.ids[:, :ratio*top:ratio] // ratio)
    assert torch.equal(scratch.scores.view(torch.int32), saved)


def test_nondefault_stream_and_changed_graph_replay(native):
    rows, columns = 5, 65537
    scores = torch.zeros((rows, columns), dtype=torch.float32, device='cuda')
    lengths = torch.tensor([columns, 1025, 513, 512, 0], dtype=torch.int32, device='cuda')
    output = torch.empty((rows, 512), dtype=torch.int32, device='cuda')
    status = torch.empty(rows, dtype=torch.int32, device='cuda')
    changed = torch.arange(columns, dtype=torch.float32, device='cuda').expand(rows, -1).contiguous()
    stream, current = torch.cuda.Stream(), torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        scores.copy_(changed)
        native.qsa_select512(scores, lengths, output, status)
    done = torch.cuda.Event()
    done.record(stream)
    current.wait_event(done)
    expected, flags = oracle(changed.cpu().numpy(), lengths.cpu().numpy())
    assert torch.equal(output.cpu(), torch.from_numpy(expected)) and torch.equal(status.cpu(), torch.from_numpy(flags))
    # Warm before capture; graph buffers and all input/output owners remain live.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        native.qsa_select512(scores, lengths, output, status)
    for bad in (False, True, False):
        scores.zero_()
        lengths.fill_(columns)
        if bad:
            lengths[0] = -1
            scores[1, 0] = torch.nan
        output.fill_(-777)
        status.fill_(-99)
        graph.replay()
        expected, flags = oracle(scores.cpu().numpy(), lengths.cpu().numpy())
        assert torch.equal(output.cpu(), torch.from_numpy(expected)) and torch.equal(status.cpu(), torch.from_numpy(flags))


@pytest.mark.parametrize('bad', ['score-dtype', 'length-dtype', 'output-dtype', 'rows', 'columns', 'stride',
                               'output-shape', 'status-shape', 'input-output-alias', 'output-status-alias', 'length-status-alias'])
def test_binding_refuses_malformed_or_overlapping_owners_before_write(native, bad):
    rows, columns = 2, 1025
    scores = torch.zeros((rows, columns), dtype=torch.float32, device='cuda')
    lengths = torch.full((rows,), columns, dtype=torch.int32, device='cuda')
    output = torch.full((rows, 512), -777, dtype=torch.int32, device='cuda')
    status = torch.full((rows,), -99, dtype=torch.int32, device='cuda')
    if bad == 'score-dtype':
        scores = scores.half()
    elif bad == 'length-dtype':
        lengths = lengths.long()
    elif bad == 'output-dtype':
        output = output.long()
    elif bad == 'rows':
        scores = torch.empty((257, columns), dtype=torch.float32, device='cuda')
    elif bad == 'columns':
        scores = torch.empty((rows, 65538), dtype=torch.float32, device='cuda')
    elif bad == 'stride':
        scores = scores[:, ::2]
    elif bad == 'output-shape':
        output = output[:, :-1]
    elif bad == 'status-shape':
        status = status[:1]
    elif bad == 'input-output-alias':
        output = scores.view(torch.int32)[:, :512]
    elif bad == 'output-status-alias':
        status = output.view(-1)[:rows]
    else:
        status = lengths
    before = output.clone()
    with pytest.raises(RuntimeError):
        native.qsa_select512(scores, lengths, output, status)
    assert torch.equal(output, before)
