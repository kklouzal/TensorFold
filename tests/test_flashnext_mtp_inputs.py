"""Explicit merged next embeddings in the real CPU MTP path; no media/GPU qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import mtp
from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
from tensorfold.families.qwen4_exp.cuda.weight_types import MTPW

from test_dspark_taps_forward import cpu_weights


@pytest.fixture
def family(monkeypatch):
    weights = cpu_weights(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_a: SimpleNamespace(multi_processor_count=80))
    c = weights.cfg
    generator = torch.Generator().manual_seed(734)
    projections = [Dense(torch.randn((c.hidden, c.hidden), generator=generator).half() * .03) for _ in range(2)]
    # Reuse the actual tiny family attention/MoE protocol. MTP never applies
    # target PLE; embedding and flattened four-stream hidden normalization
    # retain the existing source implementation.
    layer = replace(weights.layers[1], index=c.layers, ple=None)
    weights.mtp = MTPW(torch.ones(c.hidden), torch.ones(c.streams * c.hidden),
                       *projections, layer, weights.mixer)
    return weights


def buffers(weights):
    result = Buffers(weights, 4, 32)
    # CPU arithmetic has no asynchronous staging. Keep the real CUDA stage's
    # event contract in the fixture instead of changing supported device code.
    result.staged = SimpleNamespace(synchronize=lambda: None, record=lambda: None)
    return result


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_complete_text_next_inputs_preserve_legacy_output_and_owned_cache(family, cache):
    c = family.cfg
    tokens = [4, 17, 31]
    streams = torch.linspace(-.25, .5, len(tokens) * c.streams * c.hidden).reshape(len(tokens), -1).half()
    actual = family.embed[0].lookup(torch.tensor(tokens)).clone()
    legacy, supplied = [State(family, 32, 4, cache) for _ in range(2)]
    left, right = [buffers(family) for _ in range(2)]
    expected = mtp.mtp_forward(family, legacy, left, tokens, streams, last_only=False).clone()
    result = mtp.mtp_forward(family, supplied, right, tokens, streams, last_only=False, next_inputs=actual)
    assert torch.equal(result, expected)
    assert torch.equal(right.mtp_e[:len(tokens)], actual)
    for name in ("k", "v", "ks", "vs"):
        first, second = getattr(legacy.mtp_kc, name), getattr(supplied.mtp_kc, name)
        if first is not None:
            assert torch.equal(first[:len(tokens)], second[:len(tokens)])
    assert torch.equal(legacy.mtp_pooled, supplied.mtp_pooled)


def test_mixed_visual_next_inputs_skip_all_placeholder_lookups_and_keep_hc(family, monkeypatch):
    c = family.cfg
    tokens = [4, 31, 31]  # identical placeholder IDs are intentionally different projected rows
    streams = torch.linspace(-.5, .75, len(tokens) * c.streams * c.hidden).reshape(len(tokens), -1).half()
    actual = torch.stack([family.embed[0].lookup(torch.tensor([4]))[0],
                          torch.linspace(-.125, .375, c.hidden).half(),
                          torch.linspace(.25, -.5, c.hidden).half()])
    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    monkeypatch.setattr(mtp, "_embed", lambda *_a, **_k: pytest.fail("supplied visual input reached token lookup"))
    result = mtp.mtp_forward(family, state, buf, tokens, streams, last_only=False, next_inputs=actual)
    assert result.shape == (len(tokens), c.vocab) and torch.isfinite(result).all()
    assert torch.equal(buf.mtp_e[:len(tokens)], actual)
    assert torch.equal(buf.mtp_in[:len(tokens)], streams)
    assert not torch.equal(actual[1], actual[2]) and tokens[1] == tokens[2]


@pytest.mark.parametrize("layout", ["expanded", "permuted", "negative", "shifted", "self", "strided"])
def test_hc_staging_preserves_aliased_values_and_valid_strided_inputs(family, monkeypatch, layout):
    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    width = buf.mtp_in.shape[1]
    buf.mtp_in.copy_(torch.arange(buf.mtp_in.numel()).reshape(buf.mtp_in.shape))
    if layout == "expanded":
        source = buf.mtp_in[:1].expand(3, -1)
    elif layout == "permuted":
        source = buf.mtp_in.as_strided((3, width), (1, 3))
    elif layout == "negative":
        source = torch._neg_view(buf.mtp_in[:3])
    elif layout == "shifted":
        source = buf.mtp_in[1:4]
    elif layout == "self":
        source = buf.mtp_in[:3]
    else:
        source = torch.arange(6 * width).reshape(3, 2 * width).to(buf.mtp_in.dtype)[:, ::2]
    expected = source.clone()
    original, cloned = torch.Tensor.clone, []
    def clone(tensor, *args, **kwargs):
        cloned.append(tensor.numel())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    segments = mtp.mtp_stage(buf, [(state, [4, 17, 31], source)])
    assert segments == [(state, 0, 3)] and torch.equal(buf.mtp_in[:3], expected)
    assert cloned == ([] if layout in ("self", "strided") else [3 * width])


def test_hc_staging_snapshots_cross_window_aliases_before_any_destination_write(family):
    first, second = [State(family, 32, 4, "fp16") for _ in range(2)]
    buf = buffers(family)
    buf.mtp_in.copy_(torch.arange(buf.mtp_in.numel()).reshape(buf.mtp_in.shape))
    expected = buf.mtp_in[[2, 0, 1]].clone()
    windows = [(first, [4], buf.mtp_in[2:3]),
               (second, [17, 31], (buf.mtp_in[:0], buf.mtp_in[:2]))]
    assert mtp.mtp_stage(buf, windows) == [(first, 0, 1), (second, 1, 3)]
    assert torch.equal(buf.mtp_in[:3], expected)


@pytest.mark.parametrize("name", ["ids_host", "ids", "last_host", "last"])
def test_hc_sources_aliasing_metadata_are_read_before_metadata_is_overwritten(family, name):
    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    metadata = getattr(buf, name).view(buf.mtp_in.dtype)
    metadata.fill_(1)
    source = metadata[:1].expand(1, buf.mtp_in.shape[1])
    expected = source.clone()
    mtp.mtp_stage(buf, [(state, [4], source)])
    assert torch.equal(buf.mtp_in[:1], expected)


@pytest.mark.parametrize("retirement_fails", [False, True])
def test_partial_alias_snapshot_failure_preserves_inputs_and_original_error(family, monkeypatch, retirement_fails):
    first, second = [State(family, 32, 4, "fp16") for _ in range(2)]
    buf = buffers(family)
    buf.mtp_in.copy_(torch.arange(buf.mtp_in.numel()).reshape(buf.mtp_in.shape))
    buf.ids.fill_(9)
    buf.ids_host.fill_(10)
    buf.last.fill_(11)
    buf.last_host.fill_(12)
    before = [tensor.clone() for tensor in (buf.mtp_in, buf.ids, buf.ids_host, buf.last, buf.last_host)]
    primary, secondary, calls = MemoryError("second HC snapshot"), RuntimeError("staging event"), []
    original = torch.Tensor.clone
    def clone(tensor, *args, **kwargs):
        calls.append(tensor)
        if len(calls) == 2:
            raise primary
        return original(tensor, *args, **kwargs)
    events = []
    def record():
        events.append("record")
        if retirement_fails:
            raise secondary
    buf.staged = SimpleNamespace(synchronize=lambda: events.append("wait"), record=record)
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    with pytest.raises(MemoryError) as caught:
        mtp.mtp_stage(buf, [(first, [4], buf.mtp_in[1:2]), (second, [17], buf.mtp_in[:1])])
    assert caught.value is primary and events == ["wait", "record"]
    assert all(torch.equal(saved, tensor) for saved, tensor in
               zip(before, (buf.mtp_in, buf.ids, buf.ids_host, buf.last, buf.last_host)))
    if retirement_fails:
        assert primary.__cause__ is secondary and primary._tensorfold_retained_owners


def test_ready_media_cannot_infer_hidden_coordinates_from_next_token_ids(family):
    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    state.media_input_span = object()  # an active producer claim requires the explicit coordinate handoff
    buf.staged = SimpleNamespace(synchronize=lambda: pytest.fail("unbound visual window began staging"))
    streams = torch.zeros((1, family.cfg.streams * family.cfg.hidden)).half()
    with pytest.raises(ValueError, match="actual issued coordinate windows"):
        mtp.mtp_stage(buf, [(state, [31], streams)])


def test_ready_media_cannot_use_placeholder_lookup_without_actual_next_embeddings(family, monkeypatch):
    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    state.media_input_span = object()
    monkeypatch.setattr(mtp, "_kv_begin", lambda *_a: pytest.fail("missing media inputs began cache writes"))
    monkeypatch.setattr(mtp, "_embed", lambda *_a: pytest.fail("missing media inputs looked up placeholder"))
    with pytest.raises(ValueError, match="placeholder-token lookup"):
        mtp.mtp_compute(family, [(state, 0, 1)], buf)


@pytest.mark.parametrize("role", ["target", "mixed", "mtp", "missing-media"])
def test_postpreparation_input_guard_contains_the_cohort_before_cache_writes(family, monkeypatch, role):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation

    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    # The numerical body is never entered. Bind the actual preparation owner
    # after standalone CPU allocations to isolate this caller/error boundary.
    family.positions = object()
    family.meta["coordinate_profile_identity"] = "a" * 64
    context = buf.input_preparation_context = install_input_preparation(family)
    notifications = []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    monkeypatch.setattr(forward, "_kv_begin", lambda *_a: pytest.fail("invalid target input wrote cache"))
    monkeypatch.setattr(mtp, "_kv_begin", lambda *_a: pytest.fail("invalid MTP input wrote cache"))
    bad = torch.zeros((2, family.cfg.hidden)).half()
    with pytest.raises(ValueError):
        if role == "target":
            forward.compute(family, [(state, 0, 1)], buf, input_embeddings=bad)
        elif role == "mixed":
            forward.compute_mixed(family, [(state, 0, 1)], buf, [(state, 0, 1)], buf, prefill_embeddings=bad)
        elif role == "mtp":
            mtp.mtp_compute(family, [(state, 0, 1)], buf, next_inputs=bad)
        else:
            state.media_input_span = object()
            mtp.mtp_compute(family, [(state, 0, 1)], buf)
    assert notifications == [True] and context.cohort._execution_manifest_uncertain and context.cohort._abort_comm


@pytest.mark.parametrize("change", ["short", "long", "width", "dtype", "grad", "not-tensor"])
def test_bad_complete_window_refuses_before_target_cache_mutation(family, monkeypatch, change):
    c = family.cfg
    state, buf = State(family, 32, 4, "fp16"), buffers(family)
    inputs = torch.zeros((2, c.hidden), dtype=torch.float16)
    if change == "short":
        inputs = inputs[:1]
    elif change == "long":
        inputs = torch.zeros((3, c.hidden), dtype=torch.float16)
    elif change == "width":
        inputs = inputs[:, :-1]
    elif change == "dtype":
        inputs = inputs.float()
    elif change == "grad":
        inputs.requires_grad_(True)
    else:
        inputs = [[0.]]
    monkeypatch.setattr(mtp, "_kv_begin", lambda *_a: pytest.fail("invalid embeddings began target cache writes"))
    with pytest.raises(ValueError, match="complete window"):
        mtp.mtp_forward(family, state, buf, [31, 31], torch.zeros((2, c.streams * c.hidden)).half(),
                        next_inputs=inputs)
    assert state.mtp_len == state.pos == 0


def test_engine_supplied_inputs_bypass_token_lookup_graph_without_changing_legacy_route(family, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine

    engine = object.__new__(Engine)
    engine.w, engine.st, engine.mbuf = family, State(family, 32, 4, "fp16"), buffers(family)
    marker = torch.tensor([17.])
    calls = []
    engine.graphs = SimpleNamespace(mtp_forward=lambda tokens, hidden: calls.append(list(tokens)) or marker)
    hidden = torch.zeros((1, family.cfg.streams * family.cfg.hidden)).half()
    assert engine.mtp_forward([31], hidden) is marker and calls == [[31]]
    monkeypatch.setattr(mtp, "_embed", lambda *_a, **_kw: pytest.fail("explicit merged inputs entered token lookup"))
    result = engine.mtp_forward([31], hidden, next_inputs=torch.full((1, family.cfg.hidden), .125).half())
    assert result.shape == (1, family.cfg.vocab) and calls == [[31]]
