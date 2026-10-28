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


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_batch_publication_retains_current_role_counts_and_bounded_batch_diagnostics(monkeypatch, mtp, exhausted):
    from types import SimpleNamespace as NS
    from tensorfold.cuda.sm70 import family_native
    from tensorfold.families.qwen4_exp.cuda import history_owner

    owner, weights, buffers, state, layer, _ = setup_owner(monkeypatch, pending=False)
    owner.upload = NS(record=lambda stream: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: object())
    monkeypatch.setattr(family_native, "extension", lambda: NS(owner_prep_batch=lambda *args: None))
    weights.cfg.eps = 1e-6
    weights.positions.mtp = weights.positions.target
    weights.meta = {"sm70_owner_prelude_submitted_frames": 0, "sm70_owner_prelude_submitted_mtp_frames": 0,
        "sm70_owner_prelude_mtp_counter_overflowed": False,
        "sm70_owner_prelude_submitted_batches": (1 << 63) - 1 if exhausted else 0,
        "sm70_owner_prelude_submitted_mtp_batches": (1 << 63) - 1 if exhausted else 0,
        "sm70_owner_prelude_batch_counter_overflowed": False, "sm70_owner_prelude_max_batch_segments": 0}
    buffers.mtp_indexer_frame = buffers.target_indexer_frame
    buffers.mtp_indexer_position = buffers.target_indexer_position
    state.mtp_len, state.mtp_pos, state.mtp_kc = 19, state.pos_dev, state.kc[0]
    layer.attn = NS(q_scale=torch.ones(256), k_scale=torch.ones(256), iq_scale=torch.ones(128))
    buffers.prefill = False
    buffers.q = torch.empty((4, 3, 256), dtype=torch.float16)
    buffers.iq = torch.empty((4, 4, 128), dtype=torch.float16)
    published = []

    def publish(st, _layer, w, _buffers, first, last, *args, **kwargs):
        published.append((st, first, last, kwargs["mtp"]))
        history_owner._record_native_prelude(w, mtp=kwargs["mtp"])

    monkeypatch.setattr(history_owner, "_publish_native_frame", publish)
    other = NS(**vars(state))
    other.pos_dev, other.mtp_pos, other.kv_status = state.pos_dev.clone(), state.mtp_pos.clone(), state.kv_status.clone()
    assert owner.submit(weights, buffers, layer, [(state, 0, 2), (other, 2, 4)], 4, mtp=mtp)
    assert len(published) == 2 and all(row[3] is mtp for row in published)
    assert weights.meta["sm70_owner_prelude_submitted_frames"] == 2
    assert weights.meta["sm70_owner_prelude_submitted_mtp_frames"] == (2 if mtp else 0)
    assert weights.meta["sm70_owner_prelude_submitted_batches"] == ((1 << 63) - 1 if exhausted else 1)
    assert weights.meta["sm70_owner_prelude_submitted_mtp_batches"] == ((1 << 63) - 1 if exhausted else int(mtp))
    assert weights.meta["sm70_owner_prelude_batch_counter_overflowed"] is exhausted
    assert weights.meta["sm70_owner_prelude_max_batch_segments"] == 2


def test_actual_engine_pre_admission_batch_packet_and_counter_funding():
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.cuda.sm70.execution_policy import OWNER_BATCH_PRELUDE_TRIAL

    method = ast.parse(textwrap.dedent(inspect.getsource(FlashNextEngine._initialize))).body[0]
    block = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.sm70_profile == OWNER_BATCH_PRELUDE_TRIAL")
    namespace = dict(__name__=FlashNextEngine.__module__, __package__="tensorfold.families.qwen4_exp.cuda",
        self=NS(sm70_profile=OWNER_BATCH_PRELUDE_TRIAL), OWNER_BATCH_PRELUDE_TRIAL=OWNER_BATCH_PRELUDE_TRIAL,
        execution_rows=None, adaptive_sessions=True, streams=24, each=3, rows0=256, mtp=True, tp=8,
        options={"device_workspace_bytes": 0, "host_workspace_bytes": 0})
    exec(compile(ast.Module(body=[block], type_ignores=[]), "actual_engine_batch_funding", "exec"), namespace)
    assert namespace["options"] == {"device_workspace_bytes": 11264, "host_workspace_bytes": 8 * 84992}
