"""Captured owned QSA selector: the eager native selector's scores and selection at every context the slab holds."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')

HEADS, DIM, RATIO, BUDGET = 4, 128, 4, 2048
TOP, WIDTH = BUDGET // RATIO, BUDGET + RATIO + 2
STATUS, POSITION, LIVE_ROWS, ROW_OFFSET, INDEXER_OWNER, WORDS = 6, 7, 8, 13, 14, 16


@pytest.fixture(scope='module')
def native():
    from tensorfold.cuda.sm70.family_native import extension
    return extension()


class Frame:
    """One owner rank's descriptor rows over one pooled history (rows share the position, ROW_OFFSET = row)."""

    def __init__(self, rows, groups, *, seed, rank=2, owners=None):
        g = torch.Generator(device='cuda').manual_seed(seed)
        self.rows, self.rank = rows, rank
        self.query = (torch.randn((rows, HEADS, DIM), generator=g, device='cuda') * 0.5).half()
        self.pooled = (torch.randn((groups, DIM), generator=g, device='cuda') * 0.5).half()
        self.position = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.live = torch.ones(1, dtype=torch.int32, device='cuda')
        self.status = torch.full((2,), 0, dtype=torch.int32, device='cuda')
        self.status[1] = -2
        self.desc = torch.zeros((rows, WORDS), dtype=torch.int64, device='cuda')
        owners = [rank] * rows if owners is None else owners
        host = torch.zeros((rows, WORDS), dtype=torch.int64)
        for row in range(rows):
            host[row, STATUS] = self.status.data_ptr()
            host[row, POSITION] = self.position.data_ptr()
            host[row, LIVE_ROWS] = self.live.data_ptr()
            host[row, ROW_OFFSET] = row
            host[row, INDEXER_OWNER] = owners[row]
        self.desc.copy_(host)
        self.owners = owners
        self.pooled_rows = torch.full((rows,), self.pooled.data_ptr(), dtype=torch.int64, device='cuda')
        self.scores = torch.full((rows, groups), -7., dtype=torch.float32, device='cuda')
        self.selection = torch.full((rows, WIDTH), -5, dtype=torch.int32, device='cuda')

    def run(self, native, layer=9):
        native.qsa_select_owned(self.query, self.desc, self.pooled_rows, self.scores, self.selection, self.rank,
                                RATIO, BUDGET, layer)


def eager_scores(native, frame, row, complete):
    """The eager native selector's (qsa_rows: one block per row, <= 4096 groups) FP32 scores, window by window."""
    out = []
    for start in range(0, complete, 4096):
        count = min(4096, complete - start)
        window = frame.pooled[start:start + count].contiguous()
        position = torch.tensor([65536 * RATIO], dtype=torch.int32, device="cuda")  # every window group complete
        sparse = torch.zeros(1, dtype=torch.int32, device='cuda')
        counts = torch.zeros(1, dtype=torch.int32, device='cuda')
        ids = torch.zeros((1, BUDGET + RATIO), dtype=torch.int32, device='cuda')
        scores = torch.zeros((1, count), dtype=torch.float32, device='cuda')
        native.qsa_rows(frame.query[row:row + 1].contiguous(), window, position, sparse, counts, ids, scores,
                        1, RATIO, BUDGET, count)
        out.append(scores[0].cpu().numpy())
    return np.concatenate(out)


def expected_row(scores, end):
    """The canonical wire row: top budget/ratio groups (numeric order, lower group first), ascending, then the tail."""
    complete = end // RATIO
    row = np.zeros(WIDTH, dtype=np.int32)
    if complete <= TOP:
        row[WIDTH - 2], row[WIDTH - 1] = end, 0
        return row
    chosen = np.sort(np.lexsort((np.arange(complete), -scores[:complete].astype(np.float64)))[:TOP])
    row[:BUDGET] = (chosen[:, None] * RATIO + np.arange(RATIO)).reshape(-1)
    row[BUDGET:BUDGET + RATIO] = complete * RATIO + np.arange(RATIO)
    row[WIDTH - 2], row[WIDTH - 1] = BUDGET + end % RATIO, 1
    return row


@pytest.mark.parametrize('position', [0, 2043, 2044, 2047, 4096 - 3, 16384 - 5, 16384, 65536 - 5, 200000, 262144 - 5])
def test_selection_matches_eager_native_scores(native, position):
    rows, groups = 5, (262144 + RATIO) // RATIO
    frame = Frame(rows, groups, seed=position + 11)
    frame.position.fill_(position)
    frame.run(native)
    torch.cuda.synchronize()
    assert frame.status.tolist() == [0, -2]
    got = frame.selection.cpu().numpy()
    for row in range(rows):
        end = position + row + 1
        complete = end // RATIO
        scores = eager_scores(native, frame, row, complete) if complete > TOP else None
        if scores is not None:  # the captured slab holds the eager native selector's exact scores
            assert np.array_equal(frame.scores[row, :complete].cpu().numpy().view(np.int32), scores.view(np.int32))
        np.testing.assert_array_equal(got[row], expected_row(scores, end) if scores is not None else expected_row(None, end))


def test_matches_eager_native_selection_within_its_envelope(native):
    """Within qsa_rows' 4096-group envelope the wire rows equal its ids/count/sparse outputs bit for bit."""
    for position in (3000, 9000, 16384 - 6):
        rows = 5
        frame = Frame(rows, 4096, seed=position)
        frame.position.fill_(position)
        frame.run(native)
        sparse = torch.zeros(rows, dtype=torch.int32, device='cuda')
        counts = torch.zeros(rows, dtype=torch.int32, device='cuda')
        ids = torch.zeros((rows, BUDGET + RATIO), dtype=torch.int32, device='cuda')
        scores = torch.zeros((rows, 4096), dtype=torch.float32, device='cuda')
        native.qsa_rows(frame.query, frame.pooled, frame.position, sparse, counts, ids, scores, rows, RATIO, BUDGET,
                        min(4096, (position + rows + RATIO - 1) // RATIO))
        torch.testing.assert_close(frame.selection[:, :BUDGET + RATIO], ids, rtol=0, atol=0)
        torch.testing.assert_close(frame.selection[:, WIDTH - 2], counts, rtol=0, atol=0)
        torch.testing.assert_close(frame.selection[:, WIDTH - 1], sparse, rtol=0, atol=0)


def test_ties_non_owner_rows_and_failures(native):
    rows = 4
    frame = Frame(rows, 8192, seed=5, owners=[2, 3, 2, 2])
    frame.query.fill_(0.25)
    frame.pooled.fill_(0.5)                                  # every score ties: the lowest groups win
    frame.position.fill_(8192 * RATIO - rows - 1)
    frame.run(native)
    got = frame.selection.cpu().numpy()
    assert not got[1].any()                                  # another rank's row: zeros only
    end = 8192 * RATIO - rows + 0
    np.testing.assert_array_equal(got[0], expected_row(np.full(8192, 1.0, np.float32), end))
    assert frame.status.tolist() == [0, -2]
    # A nonfinite score fails the frame (bit 32, first layer) and leaves the row's ids zero.
    frame.pooled[100].fill_(float('inf'))
    frame.run(native, layer=7)
    torch.cuda.synchronize()
    assert frame.status.tolist() == [32, 7]
    assert not frame.selection[0, :BUDGET + RATIO].any().item()
    # A context beyond the score slab fails the same way, without reading past it.
    frame = Frame(2, 1024, seed=6)
    frame.position.fill_(1024 * RATIO + 8)
    frame.run(native, layer=3)
    torch.cuda.synchronize()
    assert frame.status.tolist() == [32, 3]
    # A missing position pointer too.
    frame = Frame(1, 1024, seed=7)
    frame.desc[0, POSITION] = 0
    frame.run(native, layer=4)
    torch.cuda.synchronize()
    assert frame.status.tolist() == [32, 4]


def test_graph_replay_reads_the_device_position(native):
    rows, groups = 5, 32768
    frame = Frame(rows, groups, seed=21)
    frame.position.fill_(100)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        frame.run(native)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        frame.run(native)
    for position in (1500, 9000, 70000, groups * RATIO - rows - 1):
        frame.position.fill_(position)
        graph.replay()
        torch.cuda.synchronize()
        got = frame.selection.cpu().numpy()
        for row in (0, rows - 1):
            end = position + row + 1
            scores = eager_scores(native, frame, row, end // RATIO) if end // RATIO > TOP else None
            np.testing.assert_array_equal(got[row], expected_row(scores, end))
    assert frame.status.tolist() == [0, -2]


def test_scores_against_fp64_no_less_accurate_than_eager_long_context(native):
    """Above 4096 groups the eager owner scores with FP32 Torch matmuls; the captured scores must be no less
    accurate against an FP64 evaluation of the same expression (ReLU per head, sum over heads, / sqrt(dim))."""
    from tensorfold.cuda.sm70 import family_ops

    rows, groups = 5, 16384
    frame = Frame(rows, groups, seed=99)
    frame.desc[:, ROW_OFFSET] = 0
    frame.position.fill_(groups * RATIO - 1)                # every row completes exactly all groups
    frame.run(native)
    torch.cuda.synchronize()
    assert frame.status.tolist() == [0, -2]
    q64, k64 = frame.query.double(), frame.pooled.double()
    exact = (torch.einsum('rhd,gd->rhg', q64, k64).clamp_min(0).sum(1) / DIM ** 0.5)
    eager = torch.zeros((rows, groups), dtype=torch.float32, device='cuda')
    for head in range(HEADS):
        eager = eager + family_ops._matmul32(frame.query[:, head], frame.pooled.T).clamp_min(0)
    eager = eager / DIM ** 0.5
    captured = frame.scores[:, :groups]
    err_captured = (captured.double() - exact).abs()
    err_eager = (eager.double() - exact).abs()
    print(f'captured |err| mean {err_captured.mean().item():.3e} max {err_captured.max().item():.3e}; '
          f'eager mean {err_eager.mean().item():.3e} max {err_eager.max().item():.3e}')
    assert err_captured.mean().item() <= err_eager.mean().item()
    assert err_captured.max().item() <= err_eager.max().item()
