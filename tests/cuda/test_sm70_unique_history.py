"""Eight real V100 ranks: unique owner execution and bounded live migration.

These component gates require the pinned SM70 foundation. They do not replace
full-checkpoint/context, output-quality, allocator-peak or performance gates.
"""

import socket
import time
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")


def _weights(rank, device, comm):
    cfg = NS(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
             kv_heads=1, head_dim=256, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
             ple_layers=[], eos=(0,), heads=1, index_heads=2, index_budget=8, eps=1e-6,
             mrope_section=(1, 1, 1), rope_attention_factor=1.0)
    return NS(cfg=cfg, device=device, layers=[NS(index=i, linear=False) for i in range(12)], mtp=None,
              positions=None, comm=comm, inv_freq=torch.tensor([1.0], device=device),
              meta={"world": 8, "rank": rank, "full": NS(kv_heads=2), "execution_dtype": "float16",
                    "execution_provider": "sm70", "history_layout": "unique", "attention_degree": 1})



def _prepare_zero_indexer_rows(state, weights, start, count, ordinals):
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging, resolve_indexer_coordinates

    stage = Staging(count, state.layers, weights.device, mtp=False)
    stage.begin(mtp=False)
    raw = torch.zeros((count, weights.cfg.index_dim), dtype=torch.float16, device=weights.device)
    coordinates = resolve_indexer_coordinates(state, start, count)
    scale = torch.ones(weights.cfg.index_dim, device=weights.device)
    for ordinal in ordinals:
        if state.compact_indexer.entries[ordinal] is not None:
            entry = state.compact_indexer.append(ordinal, start, raw, coordinates, stage, 0, count,
                mtp=False, expected=(state.slot_generation, state.history_generation))
            entry.pool(state.pooled[ordinal], scale, weights.inv_freq, weights.cfg.eps,
                       weights.cfg.mrope_section, weights.cfg.rope_attention_factor)



def _worker(rank, port, format_name):
    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.sm70 import family_ops as ops
    from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers, _encode, attention_layer
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    owner, failed = NCCL(), False
    try:
        owner.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_unique_history_uid")
        device = torch.device("cuda", rank)
        w = _weights(rank, device, owner)
        state = State(w, 257, 3, format_name)
        state.kv_begin()
        pair = get_pair(format_name)
        key = torch.zeros((135, 1, 256), dtype=torch.float16, device=device)
        value = torch.full_like(key, 1 if rank < 4 else 2)
        codes, scales = _encode(key, pair.key, state.kv_status, 0, error=2)
        values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
        for cache in state.kc:
            assert cache._empty_physical == (cache.reserved_rows == 0)
            cache.append(0, codes[:129], values[:129], scales[:129], value_scales[:129])
        _prepare_zero_indexer_rows(state, w, 0, 129, range(len(state.kc)))
        state.set_pos(129)
        scratch = AttnScratch(3, 1, 256, 257, device, budget=8, ratio=4, dtype=torch.float16)
        projection = torch.zeros((3, 4 * 256 + 3 * 128), dtype=torch.float16, device=device)
        projection[:, 3 * 256:4 * 256].fill_(1 if rank < 4 else 2)
        buffers = NS(pa=projection, q=torch.empty((3, 1, 256), dtype=torch.float16, device=device),
                     iq=torch.empty((3, 2, 128), dtype=torch.float16, device=device),
                     attn_o=torch.empty((3, 1, 256), dtype=torch.float16, device=device),
                     attn=scratch, prefill=False, input_preparation_context=None,
                     target_indexer_coordinates=None, target_indexer_frame=None,
                     target_indexer_position=None, target_indexer_frame_id=0)
        buffers.compact_indexer_stage = Staging(3, state.layers, device, mtp=False)
        buffers.target_indexer_coordinate_storage = buffers.compact_indexer_stage.coordinates
        buffers.owner_kernel_family, buffers.owner_graph_family, buffers.owner_graphs = "torch-reference", "uncaptured", None
        buffers.history_owner = OwnerBuffers(w, 3, scratch)
        layer = NS(index=0, attn=NS(q_scale=torch.ones(256, device=device), k_scale=torch.ones(256, device=device),
                                  iq_scale=torch.ones(128, device=device), ik_scale=torch.ones(128, device=device)))
        state.kv_begin()
        actual = attention_layer(layer, w, [(state, 0, 3)], buffers, 3).clone()
        # Pre-existing complete-history attention is independent of the page
        # owner/migration implementation. Query/indexer inputs are exactly zero.
        full = KVCache(257, 1, 256, device, pair=pair)
        for original, frame in ((full.k, codes), (full.v, values), (full.ks, scales), (full.vs, value_scales)):
            original[:132].copy_(frame[:132])
        pos = torch.tensor([129], dtype=torch.int32, device=device)
        ops.qsa_rows(torch.zeros((3, 2, 128), dtype=torch.float16, device=device),
                     torch.zeros((65, 128), dtype=torch.float16, device=device), pos, scratch, 3, context=132)
        expected = ops.attention(torch.zeros_like(actual), full.k, full.v, pos, scratch, 3, 256 ** -0.5,
                                 context=132, ks=full.ks, vs=full.vs,
                                 k_bits=pair.key.bits, k_codec=pair.key.codec,
                                 v_bits=pair.value.bits, v_codec=pair.value.codec)
        assert torch.equal(actual, expected)
        _prepare_zero_indexer_rows(state, w, 129, 3, range(1, len(state.kc)))
        state.set_pos(132)
        state.begin_history_migration(4)
        for _ in range(12):
            if not state.migrate_history_bundle():
                break
        else:
            raise AssertionError("finite native migration did not retire")
        assert state.history_placement.degree == 4 and state.history_migration_bytes == 0
        logical = torch.arange(257, device=device)
        for cache in state.kc:
            if cache.reserved_rows == 0 and any(offset >= 0 for offset in cache.offsets):
                assert not cache._empty_physical
            offsets, owned = cache.local_indices(logical)
            for stored, reference in ((cache.k, full.k), (cache.v, full.v),
                                      (cache.ks, full.ks), (cache.vs, full.vs)):
                # Only layer0's three new rows were prepared by the real layer;
                # other layers retain their original129-row committed fixture.
                expected_bytes = reference.clone()
                if cache.layer:
                    expected_bytes[129:132].zero_()
                assert torch.equal(stored[offsets[owned]].view(torch.uint8),
                                   expected_bytes[owned].view(torch.uint8))
        # Execute again after real degree1->4 publication. Layer0 now has
        # physical rows on formerly empty ranks; a stale initial empty proof
        # would remove their visible mass/value contributions. Use the same
        # independent full-history attention oracle and unchanged byte check.
        # Distinct new values ensure losing a newly owned visible tail cannot
        # be hidden by renormalizing an otherwise uniform-value history.
        value[132:].fill_(3 if rank < 4 else 4)
        values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
        projection[:, 3 * 256:4 * 256].fill_(3 if rank < 4 else 4)
        state.kv_begin()
        actual = attention_layer(layer, w, [(state, 0, 3)], buffers, 3).clone()
        for original, frame in ((full.k, codes), (full.v, values), (full.ks, scales), (full.vs, value_scales)):
            original[:135].copy_(frame)
        pos.fill_(132)
        ops.qsa_rows(torch.zeros((3, 2, 128), dtype=torch.float16, device=device),
                     torch.zeros((65, 128), dtype=torch.float16, device=device), pos, scratch, 3, context=135)
        expected = ops.attention(torch.zeros_like(actual), full.k, full.v, pos, scratch, 3, 256 ** -0.5,
                                 context=135, ks=full.ks, vs=full.vs,
                                 k_bits=pair.key.bits, k_codec=pair.key.codec,
                                 v_bits=pair.value.bits, v_codec=pair.value.codec)
        assert torch.equal(actual, expected)
        _prepare_zero_indexer_rows(state, w, 132, 3, range(1, len(state.kc)))
        state.set_pos(135)
        state.kv_check()
        owner.ready("unique-history-component-complete", every=1, timeout=60)
    except BaseException:
        failed = True
        raise
    finally:
        owner.close(abort=failed)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("format_name", ["int8", "rotorquant-planar4-sm70-v1"])
def test_zero_indexer_fixture_commits_produced_rows_and_coordinates(rank, format_name):
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import validate_target_indexer_frame
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = _weights(rank, "cpu", None)
    invalid = State(weights, 257, 3, format_name)
    with pytest.raises(ValueError, match="produced cache span/generation"):
        invalid.set_pos(129)
    state = State(weights, 257, 3, format_name)
    assert weights.positions is None and state.coordinate_profile_identity is None
    _prepare_zero_indexer_rows(state, weights, 0, 129, range(len(state.kc)))
    state.set_pos(129)
    expected = torch.zeros((4, 3), dtype=torch.int64)
    expected[0].fill_(128)
    for entry, pooled in zip(state.compact_indexer.entries, state.pooled):
        if entry is not None:
            assert entry.n == 129
            assert torch.equal(entry.coordinates, expected)
            assert torch.count_nonzero(entry.raw) == torch.count_nonzero(pooled) == 0
    # Separate layer0 production from the untouched layers, as the real
    # attention call does; their spans must all survive until one State commit.
    _prepare_zero_indexer_rows(state, weights, 129, 3, (0,))
    _prepare_zero_indexer_rows(state, weights, 129, 3, range(1, len(state.kc)))
    state.set_pos(132)
    for entry, pooled in zip(state.compact_indexer.entries, state.pooled):
        if entry is not None:
            assert entry.n == 132
            assert torch.count_nonzero(entry.coordinates) == 0
            assert torch.count_nonzero(entry.raw) == torch.count_nonzero(pooled) == 0
    buffers = NS(input_preparation_context=None, target_indexer_frame=None, target_indexer_coordinates=None,
                 target_indexer_position=None, target_indexer_coordinate_storage=None)
    assert validate_target_indexer_frame(buffers, [(state, 0, 3)]) is None
    state.kv_check()


@pytest.mark.parametrize("format_name", ["int8", "rotorquant-planar4-sm70-v1"])
def test_sm70_tp8_unique_histories_and_real_page_migration(format_name):
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the pinned V100 foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port, format_name), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("native unique-history participant did not retire")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
