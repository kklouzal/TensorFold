"""Batched prelude bounded owner resources and explicit startup policy."""
import pytest
torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.batch_prelude import workspace_bytes  # noqa: E402


@pytest.mark.parametrize("rows", [1, 3, 32, 256, 288, 4096])
def test_descriptor_storage_and_fixed_metadata_are_funded(rows):
    device, host = workspace_bytes(rows)
    assert device >= rows * 32 and device % 512 == 0
    assert host == rows * 32 + 16384


@pytest.mark.parametrize("rows", [False, 0, -1, 1.5, 4097])
def test_descriptor_envelope_refuses_without_allocation(rows):
    with pytest.raises(ValueError):
        workspace_bytes(rows)

def setup_owner(monkeypatch, *, pending):
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda.batch_prelude import BatchPrelude
    from tensorfold.families.qwen4_exp.cuda import history_owner, compact_indexer

    owner = object.__new__(BatchPrelude)
    owner.rows, owner.pending, owner.error = 4, pending, None
    owner.descriptor = torch.empty((4, 4), dtype=torch.int64)
    owner.host = torch.empty_like(owner.descriptor)
    owner.view = owner.host.numpy()
    failure = RuntimeError("previous descriptor DMA failed")
    def fail():
        raise failure
    owner.upload = NS(synchronize=fail)
    role = NS(inv_freq=torch.ones(32), sections=(11, 11, 10), amplitude=1.)
    weights = NS(cfg=NS(heads=3, head_dim=256, index_heads=4, index_dim=128),
                 positions=NS(target=role), device="cpu")
    fmt = NS(bits=8, codec=0)
    state = NS(image_positions=None, rope_delta=0, media_profile_identity=None, media_input_span=None,
               history_placement=NS(target_layers=1), att_index={0: 0},
               kc=[NS(key_format=fmt, value_format=fmt)], pos=31,
               pos_dev=torch.tensor([31], dtype=torch.int32), kv_status=torch.tensor([0, -2], dtype=torch.int32))
    buffers = NS(rows=4, target_indexer_frame=object(), target_indexer_position=role,
                 pa=torch.zeros((4, 2688), dtype=torch.float16),
                 gated=torch.zeros((4, 768), dtype=torch.float16))
    monkeypatch.setattr(history_owner, "_native_prelude_position", lambda _position: True)
    monkeypatch.setattr(compact_indexer, "_scalar_text_frame", lambda _frame: True)
    return owner, weights, buffers, state, NS(index=0), failure


def test_unknown_prior_dma_retirement_poison_is_latched_before_owner_reuse(monkeypatch):
    owner, weights, buffers, state, layer, primary = setup_owner(monkeypatch, pending=True)
    with pytest.raises(RuntimeError) as caught:
        owner.submit(weights, buffers, layer, [(state, 0, 1)], 1, mtp=False)
    assert caught.value is primary and owner.error is primary
    with pytest.raises(RuntimeError, match="requires model retirement") as stopped:
        owner.submit(weights, buffers, layer, [(state, 0, 1)], 1, mtp=False)
    assert stopped.value.__cause__ is primary


def test_nontext_segment_declines_before_descriptor_or_scratch_mutation(monkeypatch):
    owner, weights, buffers, state, layer, _primary = setup_owner(monkeypatch, pending=True)
    state.image_positions = object()
    owner.host.fill_(9)
    buffers.gated.fill_(7)
    assert owner.submit(weights, buffers, layer, [(state, 0, 1)], 1, mtp=False) is False
    assert owner.error is None and bool((owner.host == 9).all()) and bool((buffers.gated == 7).all())

def test_actual_eight_request_fixture_budget_keeps_all_normal_prefills_overlapped():
    from tests.cuda.test_sm70_owner_piece_graphs_tp8 import _batch_fixture_requests

    streams = _batch_fixture_requests()
    assert len(streams) == 8
    assert [len(stream.prompt) for stream in streams] == [9, 13, 17, 21, 25, 29, 33, 37]
    maximum_prompt_passes = sum((len(stream.prompt) + 7) // 8 for stream in streams)
    assert maximum_prompt_passes == 28
    assert all(stream.count == 96 and stream.count > 3 * maximum_prompt_passes + 1
               and len(stream.prompt) + stream.count <= 257 for stream in streams)
    assert all(stream.stop_eos is False and stream.sampling is None for stream in streams)
