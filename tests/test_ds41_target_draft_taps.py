"""The trained DSpark tap ordering at real Target pause boundaries."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v4.cuda.target import Shard, Target


def _target(hidden, layers=40):
    target = Target.__new__(Target)
    target.device = torch.device("cpu")
    target.g = SimpleNamespace(hidden=hidden)
    target.w = SimpleNamespace(layers=[SimpleNamespace(index=i, engram_wkv=None) for i in range(layers)])
    return target


@pytest.mark.parametrize("sharded", [False, True])
def test_taps_use_entering_residual_in_row_order_across_pauses(sharded):
    tokens, hidden = 1027, 8
    target = _target(hidden)
    indices = torch.tensor([1026, 0, 904, 128, 1026])
    shard = Shard.of(tokens, 7, 8) if sharded else None
    st = {"next": 0, "shard": shard, "step": SimpleNamespace(ids=torch.zeros(tokens, dtype=torch.int64))}
    out = torch.full((len(indices), 3 * hidden), torch.nan, dtype=torch.float16)
    raw = torch.arange(tokens * 4 * hidden, dtype=torch.float32).reshape(tokens, 4, hidden) / 8
    marks, stops = [], []

    def advance(state, stop, receipt):
        assert state is st and receipt is marks and state["next"] <= stop
        stops.append(stop)
        residual = raw + (stop * 0.5)
        state.update(next=stop, hidden=residual if shard is None else shard.take(residual))

    target.layers = advance

    def oracle(cut):
        value = raw[indices].double() + cut * 0.5
        return ((value[:, 0] + value[:, 1] + value[:, 2] + value[:, 3]) / 256).half()

    if shard is not None:
        owners = indices // shard.size
        columns = torch.arange(len(indices))

        def gather(part, received):
            assert received.shape == (8, len(indices), hidden) and part.dtype == received.dtype == torch.float16
            expected = oracle(st["next"])
            owned = owners == shard.rank
            assert torch.equal(part[owned], expected[owned])
            # Supply independent results only at their owning rank. Stale or
            # wrongly selected rows remain NaN instead of weakening the check.
            received.fill_(torch.nan)
            received[owners, columns] = expected

        target.c = SimpleNamespace(world=SimpleNamespace(all_gather=gather))

    target.layers_with_draft_taps(st, 74, indices, out, marks)
    assert torch.isnan(out).all()
    target.layers_with_draft_taps(st, 75, indices, out, marks)
    assert torch.equal(out[:, :hidden], oracle(74))
    assert torch.isnan(out[:, hidden:]).all()
    target.layers_with_draft_taps(st, target.units, indices, out, marks)
    assert stops == [74, 75, 76, 78, 80]
    assert torch.equal(out, torch.cat([oracle(cut) for cut in (74, 76, 78)], dim=1))


def test_taps_reject_untrained_layer_geometry_and_bad_buffers_before_running():
    hidden = 8
    target = _target(hidden, layers=39)
    indices = torch.tensor([0])
    out = torch.empty((1, 3 * hidden), dtype=torch.float16)
    st = {"next": 0, "shard": None, "step": SimpleNamespace(ids=torch.zeros(1, dtype=torch.int64))}
    with pytest.raises(ValueError, match="full trained DeepSeek"):
        target.layers_with_draft_taps(st, target.units, indices, out)

    target = _target(hidden)
    with pytest.raises(ValueError, match="FP16 destination"):
        target.layers_with_draft_taps(st, target.units, indices, out.float())
    with pytest.raises(ValueError, match="int64 row indices"):
        target.layers_with_draft_taps(st, target.units, indices.int(), out)
    assert st["next"] == 0
