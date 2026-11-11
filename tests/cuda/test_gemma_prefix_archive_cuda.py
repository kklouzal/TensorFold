"""SM121 archive transport/ownership, without weights or model arithmetic.

Installed runtime imports are mandatory. These synthetic byte-word controls do
not qualify model quality, attention mathematics, capacity admission or P3.
The largest declared application-array envelope is 192MiB; allocator/context/
graph-pool observations are reported separately for the operator's GPU grant.
"""
from importlib.metadata import distribution
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gemma_cuda_target_metadata_test import tiny_config
from test_gemma_prefix_ranges import fill_target
from tensorfold.families.gemma4.cuda.dflash import DraftState
from tensorfold.families.gemma4.cuda.prefix_archive import GemmaPrefixArchive
from tensorfold.families.gemma4.cuda.state import (
    State, capacity_bytes, companion_rings, derived_capacity_bytes,
)
from tensorfold.families.gemma4.cuda.weights import Config

ARRAY_ENVELOPE_BYTES = 192 << 20
ARCHIVE_BYTES = 8 << 20
NATIVE_CONTEXT = 262144
DELAY_CYCLES = 200_000_000  # same bounded native delay as test_grammar_mask


def config(context=2048, layers=4):
    raw = tiny_config()
    raw['text_config']['num_hidden_layers'] = layers
    raw['text_config']['layer_types'] = ['sliding_attention', 'full_attention'] * (layers // 2)
    raw['text_config']['max_position_embeddings'] = context
    return Config.from_dict(raw)


def draft_config():
    return SimpleNamespace(layer_types=('sliding_attention', 'full_attention'),
                           sliding_window=8, kv_heads=1, head_dim=8)


def target_bytes(cfg, context, rows, mode='none'):
    return capacity_bytes(cfg, context, rows) + derived_capacity_bytes(cfg, context, mode) + 4


def draft_bytes(context, rows):
    cfg = draft_config()
    return sum(companion_rings(cfg.layer_types, cfg.sliding_window, 5, rows, context)) * 8 * 4 + 4


def require_arrays(nbytes):
    # Before State/Draft/tensor creation, not an after-allocation assertion.
    assert type(nbytes) is int and 0 < nbytes <= ARRAY_ENVELOPE_BYTES
    return nbytes


@pytest.fixture
def gpu(record_property):
    # No skip may count as the native acceptance result. cuda/conftest handles
    # ordinary CPU discovery; the released GPU invocation must collect12 cases.
    initialized = torch.cuda.is_initialized()
    assert torch.cuda.is_available(), 'requires the released actual GB10 CUDA phase'
    assert torch.cuda.get_device_capability(0) == (12, 1), 'requires actual SM121'
    installed = Path(distribution('tensorfold').locate_file('tensorfold')).resolve()
    for cls in (State, DraftState, GemmaPrefixArchive):
        assert Path(inspect.getfile(cls)).resolve().is_relative_to(installed), 'source overlay is forbidden'
    device = torch.device('cuda:0')
    torch.cuda.init()
    torch.cuda.synchronize(device)
    before = {'allocated': torch.cuda.memory_allocated(device),
              'reserved': torch.cuda.memory_reserved(device),
              'free': torch.cuda.mem_get_info(device)[0]}
    torch.cuda.reset_peak_memory_stats(device)
    record_property('gpu_authority', json.dumps({'device': str(device), 'capability': [12, 1],
        'name': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'torch_git': torch.version.git_version, 'initialized_before': initialized,
        'application_array_envelope_bytes': ARRAY_ENVELOPE_BYTES,
        'scope': 'transport only; no RoPE/attention tables or model weights allocated'}))
    yield device
    torch.cuda.synchronize(device)
    record_property('allocator_observation', json.dumps({'before': before,
        'after_allocated': torch.cuda.memory_allocated(device),
        'after_reserved': torch.cuda.memory_reserved(device),
        'peak_allocated': torch.cuda.max_memory_allocated(device),
        'peak_reserved': torch.cuda.max_memory_reserved(device),
        'after_free': torch.cuda.mem_get_info(device)[0],
        'scope': 'PyTorch allocator observations, not whole-process/cgroup/CUDA-context accounting'}))


def target_arrays(state):
    return tuple(t for t in (state.raw, state.inv, state.keys, state.values, state.pos,
                             state.derived_keys, state.derived_values) if t is not None)


def pointers(engine):
    return tuple(t.data_ptr() for t in (*target_arrays(engine._state),
                                       *engine._draft.keys, *engine._draft.values, engine._draft.pos))


def make_pair(device):
    cfg, context, rows = config(), 2048, 1024
    require_arrays(target_bytes(cfg, context, rows) + draft_bytes(context, rows)
                   + ARCHIVE_BYTES + (rows * cfg.hidden * 2) + 3 * rows * 8)
    state = State(cfg, device, context, rows=rows)
    draft = DraftState(draft_config(), device, context, rows, 5)
    engine = SimpleNamespace(_state=state, _draft=draft, _prefix_draft_kind='dflash',
        context_window=context, target=SimpleNamespace(config=cfg,
            project_logits=lambda row: row[:, :3].float().clone()))
    return engine, GemmaPrefixArchive(engine, ARCHIVE_BYTES)


def fill_pair(engine, end):
    fill_target(engine._state, end)  # independent maintained BF16/F32 word oracle
    draft = engine._draft
    for layer, ring in enumerate(draft.rings):
        for row in range(max(0, end - ring), end):
            draft.keys[layer][row % ring].view(torch.int16).fill_(0x3F00 + row + layer * 1024)
            draft.values[layer][row % ring].view(torch.int16).fill_(0xBF00 + row + layer * 1024 - 65536)
    draft.length = end
    words = (torch.arange(end, device=engine._state.device, dtype=torch.int32) + 0x3F00).to(torch.int16)
    return words.view(torch.bfloat16)[:, None].expand(end, engine.target.config.hidden).clone()


def assert_words(tensor, word):
    assert torch.all(tensor.view(torch.int16).cpu() == word)


@pytest.mark.parametrize('mode', ['none', 'k', 'kv'])
def test_real_stream_target_wrap_offset_words_and_storage_identity(gpu, mode):
    cfg, context, rows = config(256), 256, 16
    require_arrays(2 * target_bytes(cfg, context, rows, mode) + (1 << 20))
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        source = State(cfg, gpu, context, rows, global_cache=mode, cache_owner=object() if mode != 'none' else None)
        dest = State(cfg, gpu, context, rows, global_cache=mode, cache_owner=object() if mode != 'none' else None)
        fill_target(source, 133)
        payload = source.cache_export_rows(117, 133)
        for values in payload.values():
            for held in values:
                assert all(held.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
                           for live in target_arrays(source))
        for value in target_arrays(source):
            value.zero_()
        for value in target_arrays(dest):
            value.fill_(-777)
        before = tuple(value.data_ptr() for value in target_arrays(dest))
        dest.poisoned = True
        for group, values in payload.items():
            dest.cache_import_rows(group, 119, values, 2, 10)
        dest.cache_finish_restore(129)
    stream.synchronize()
    assert before == tuple(value.data_ptr() for value in target_arrays(dest))
    assert dest.position == dest.high_water == 129 and not dest.poisoned
    assert dest.global_cache_ready is (mode == 'none')
    for layer in range(2):
        for row in (119, 128):
            assert_words(dest.raw[layer, row], 0x3F00 + row + layer * 1024)
            assert torch.all(dest.inv[layer, row].view(torch.int32).cpu() == 0x3F000000 + row + layer * 65536)
            assert_words(dest.keys[layer, row % dest.ring], 0x3F00 + row + layer * 1024)
            assert_words(dest.values[layer, row % dest.ring], 0xBF00 + row + layer * 1024 - 65536)
    assert torch.all(dest.raw[:, :119].cpu() == -777) and torch.all(dest.raw[:, 129:].cpu() == -777)


@pytest.mark.parametrize('cut', [1, 17, 63, 64, 65, 133])
def test_interior_pair_restore_keeps_captured_graph_backing_and_endpoint(gpu, cut):
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        engine, archive = make_pair(gpu)
        tokens = tuple(range(133))
        hidden = fill_pair(engine, len(tokens))
        assert archive.capture(tokens, 0, len(tokens), hidden)
    stream.synchronize()
    archive.reap(positive_retirement=True)
    identity = archive.identity(tokens[:cut])
    assert identity is not None and identity.position == cut
    before = pointers(engine)
    with torch.cuda.stream(stream):
        for value in (*target_arrays(engine._state), *engine._draft.keys, *engine._draft.values):
            value.zero_()
        observed = torch.empty_like(engine._state.raw[:, cut - 1:cut])
        drafted = torch.empty_like(engine._draft.keys[1][cut - 1:cut])
        observed.copy_(engine._state.raw[:, cut - 1:cut])
        drafted.copy_(engine._draft.keys[1][cut - 1:cut])
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        observed.copy_(engine._state.raw[:, cut - 1:cut])
        drafted.copy_(engine._draft.keys[1][cut - 1:cut])
    with torch.cuda.stream(stream):
        restored, logits = archive.restore(tokens[:cut])
        graph.replay()
    stream.synchronize()
    archive.reap(positive_retirement=True)
    assert pointers(engine) == before and restored == cut
    assert engine._state.position == engine._draft.length == cut and not engine._state.poisoned
    assert archive.identity(tokens[:cut]) == identity
    for layer in range(2):
        assert_words(observed[layer], 0x3F00 + cut - 1 + layer * 1024)
    assert_words(drafted, 0x3F00 + cut - 1 + 1024)
    assert torch.equal(logits.cpu(), torch.full((1, 3), hidden[cut - 1, 0].float().cpu().item()))
    graph = None  # retire graph consumer before releasing its live State owners
    archive.close()


def test_delayed_actual_event_holds_pins_until_positive_stream_retirement(gpu):
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        engine, archive = make_pair(gpu)
        tokens = tuple(range(133))
        hidden = fill_pair(engine, len(tokens))
    stream.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(DELAY_CYCLES)
        assert archive.capture(tokens, 0, len(tokens), hidden)
    assert archive.pending, 'bounded delay expired before observation; no pending-event claim is established'
    event = archive.pending[0].event
    assert not event.query(), 'bounded delay precondition expired; rerun only by operator disposition'
    before = archive.store.stats
    archive.reap()
    after = archive.store.stats
    assert before['pins'] > 0 and before['pins'] == after['pins'] and archive.pending
    assert before['charged_bytes'] == after['charged_bytes']
    stream.synchronize()
    archive.reap(positive_retirement=True)
    assert not archive.pending and archive.store.stats['pins'] == 0
    assert archive.identity(tokens) is not None
    archive.close()


def test_partial_real_gpu_import_failure_retains_pair_payload_and_pins(gpu, monkeypatch):
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        engine, archive = make_pair(gpu)
        tokens = tuple(range(133))
        hidden = fill_pair(engine, len(tokens))
        assert archive.capture(tokens, 0, len(tokens), hidden)
    stream.synchronize()
    archive.reap(positive_retirement=True)
    original, calls, primary = engine._state.cache_import_rows, [], RuntimeError('injected after actual CUDA import')
    def partial(*args, **kwargs):
        original(*args, **kwargs)
        calls.append(True)
        raise primary
    monkeypatch.setattr(engine._state, 'cache_import_rows', partial)
    with torch.cuda.stream(stream), pytest.raises(RuntimeError) as caught:
        archive.restore(tokens[:65])
    assert caught.value is archive.failure is primary and len(calls) == 1
    assert engine._state.poisoned and archive.pending and archive.store.stats['pins'] > 0
    assert archive.store.stats['payload_bytes'] > 0
    with pytest.raises(RuntimeError, match='retain its owners'):
        archive.restore(tokens[:65])
    assert len(calls) == 1
    stream.synchronize()  # positively retire the real accepted copy, then its pins
    archive.close()
    assert archive.store.stats['closed'] and archive.store.stats['payload_bytes'] == 0


def test_native_context_tail_target_and_draft_ranges_without_false_frontier(gpu):
    cfg, context, rows = config(NATIVE_CONTEXT, layers=2), NATIVE_CONTEXT, 16
    require_arrays(2 * target_bytes(cfg, context, rows) + 2 * draft_bytes(context, rows) + (1 << 20))
    stream = torch.cuda.Stream(device=gpu)
    with torch.cuda.stream(stream):
        source, dest = State(cfg, gpu, context, rows), State(cfg, gpu, context, rows)
        left, right = DraftState(draft_config(), gpu, context, rows, 5), DraftState(draft_config(), gpu, context, rows, 5)
        for value in (*target_arrays(source), *left.keys, *left.values):
            value.fill_(1)
        source.position = source.high_water = left.length = context
        for row in range(context - 4, context):
            source.raw[0, row].view(torch.int16).fill_(0x3F00 + row % 2048)
            left.keys[1][row].view(torch.int16).fill_(0x4300 + row % 1024)
        target = source.cache_export_rows(context - 4, context, groups=('target_full',))
        draft = left.cache_export_rows(context - 4, context, groups=('draft_full',))
        dest.poisoned = True
        dest.cache_import_rows('target_full', context - 3, target['target_full'], 1, 3)
        right.cache_import_rows('draft_full', context - 3, draft['draft_full'], 1, 3)
    stream.synchronize()
    assert source.context == cfg.max_position == context == NATIVE_CONTEXT
    assert dest.poisoned and dest.position == right.length == 0  # tail copy is not complete-prefix authority
    for row in range(context - 3, context):
        assert_words(dest.raw[0, row], 0x3F00 + row % 2048)
        assert_words(right.keys[1][row], 0x4300 + row % 1024)
