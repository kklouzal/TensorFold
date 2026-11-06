"""Private PP2 component orchestration and Half boundaries; no GPU/model qualification."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.pipeline_stage import PipelineStage
from tensorfold.families.qwen4_exp.cuda import forward


def model(index=None):
    kinds = ["linear" if i % 4 != 3 else "attention" for i in range(48)]
    cfg = NS(layers=48, layer_types=kinds, streams=4, hidden=256, quant="gguf",
             execution_dtype="float16", ple_layers=[])
    stage = None if index is None else PipelineStage(index)
    ids = range(48) if stage is None else stage.layer_ids
    meta = dict(world=8, rank=0, full=cfg, execution_provider="sm70", execution_dtype="float16",
                history_layout="unique", attention_degree=4, indexer_history="gpu")
    if stage is not None:
        meta["pipeline_stage"] = stage.identity()
    return NS(cfg=cfg, meta=meta, layers=[NS(index=i, linear=kinds[i] == "linear", ple=None) for i in ids],
              embed=(object(),) if stage is None or stage.owns_input else None,
              mixer=object() if stage is None or stage.owns_output else None,
              head=NS(n=32) if stage is None or stage.owns_output else None,
              mtp=None, x3=None, dspark=None, dflash2=None, comm=NS(world=8, rank=0))


def buffers(rows, stage=None, *, prefill=False):
    return NS(rows=rows, pipeline_stage=stage, prefill=prefill, split=None,
              h=torch.zeros((rows, 4 * 256), dtype=torch.float16),
              streams=torch.zeros((rows, 4 * 256), dtype=torch.float16),
              pss=torch.zeros((rows, 1, 4), dtype=torch.float32), ids=torch.arange(rows))


def oracle(rows):
    """Independent NumPy Half storage/reduction oracle for noncommutative test layers."""
    h = np.broadcast_to(np.arange(256, dtype=np.float32)[None, None] / np.float32(2048)
                        + np.arange(rows, dtype=np.float32)[:, None, None] / np.float32(64),
                        (rows, 4, 256)).astype(np.float16).copy()
    injection = np.array([.5, -.25, .125, 1], np.float16)[None, :, None]
    for layer in range(48):
        h = (np.roll(h.astype(np.float32), 1, axis=1)
             * np.float32(1 + (layer % 3 - 1) / 1024) + np.float32((layer % 5 - 2) / 8192)).astype(np.float16)
        partials = [np.full((rows, 256), (layer + 1) / 4096 + rank / 131072, np.float32) for rank in range(8)]
        total = partials[0]
        for value in partials[1:]:
            total = np.add(total, value, dtype=np.float32)
        addition = (total.astype(np.float16)[:, None].astype(np.float32)
                    * injection.astype(np.float32)).astype(np.float16)
        h = (h.astype(np.float32) + addition.astype(np.float32)).astype(np.float16)
    return h


@pytest.mark.parametrize("rows,prefill", [(1, False), (3, False), (3, True)])
def test_split_execution_composes_full_hc_and_flushes_last_pending_once(rows, prefill, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    events = []
    monkeypatch.setattr(compact_indexer, "validate_target_indexer_frame", lambda *args: None)
    monkeypatch.setattr(forward, "_kv_begin", lambda *args: events.append("begin"))
    monkeypatch.setattr(forward, "_kv_check", lambda *args, **kwargs: events.append("check"))
    monkeypatch.setattr(forward.image_rows, "embed", lambda *args: events.append("images"))

    def embed(w, ids, copies, out):
        events.append("embed")
        source = torch.arange(256, dtype=torch.float32)[None] / 2048 + ids[:, None].float() / 64
        out.view(rows, copies, 256).copy_(source[:, None].half())

    def layer(layer, w, segs, b, count, pending, **kwargs):
        events.append(layer.index)
        if pending is not None:
            forward._writeback(b.h, b, count, w.cfg, pending)
        h = b.h.view(rows, 4, 256)
        h.copy_((torch.roll(h.float(), 1, dims=1) * (1 + (layer.index % 3 - 1) / 1024)
                 + (layer.index % 5 - 2) / 8192).half())
        partials = torch.stack([torch.full((rows, 256), (layer.index + 1) / 4096 + rank / 131072,
                                            dtype=torch.float32) for rank in range(8)])
        inject = torch.tensor([.5, -.25, .125, 1], dtype=torch.float16).expand(rows, 4)
        return 3, partials, None, inject

    def finish(w, mixer, b, count, pending, logits=True, ends=()):
        assert mixer is w.mixer and mixer is not None
        events.append("head")
        b.streams.copy_(b.h)
        forward._writeback(b.streams, b, count, w.cfg, pending)
        return b.streams.view(rows, 4, 256).clone()

    monkeypatch.setattr(forward, "_embed", embed)
    monkeypatch.setattr(forward, "layer_forward", layer)
    monkeypatch.setattr(forward, "finish", finish)
    segs = [(object(), 0, rows)]
    reference = forward.compute(model(), segs, buffers(rows, prefill=prefill))
    assert np.array_equal(reference.numpy().view(np.uint16), oracle(rows).view(np.uint16))
    events.clear()
    first, last = model(0), model(1)
    b0, b1 = buffers(rows, PipelineStage(0), prefill=prefill), buffers(rows, PipelineStage(1), prefill=prefill)
    boundary = forward.compute_stage(first, segs, b0)
    assert boundary.shape == (rows, 4, 256) and boundary.is_contiguous()
    assert boundary.data_ptr() == b0.streams.data_ptr()  # Borrowed until caller completes the handoff.
    assert "head" not in events and [e for e in events if type(e) is int] == list(range(24))
    assert not torch.equal(boundary[:, 0], boundary[:, 1])  # A stream mean/repeated vector cannot substitute.
    saved = boundary.clone()
    result = forward.compute_stage(last, segs, b1, input_hc=boundary)
    assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
    assert torch.equal(boundary, saved)
    assert [e for e in events if type(e) is int] == list(range(48))
    assert events.count("embed") == events.count("head") == 1
    assert events.count("check") == 2


@pytest.mark.parametrize("bad", ["missing", "one-stream", "dtype", "layout", "gradient", "device"])
def test_stage_input_boundary_refuses_before_any_cache_or_operator_work(bad, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    w, b = model(1), buffers(2, PipelineStage(1))
    failures = []
    b.input_preparation_context = NS(fail=lambda error: failures.append(error))
    value = torch.zeros((2, 4, 256), dtype=torch.float16)
    if bad == "missing":
        value = None
    elif bad == "one-stream":
        value = value[:, 0]
    elif bad == "dtype":
        value = value.float()
    elif bad == "layout":
        value = torch.zeros((2, 4, 512), dtype=torch.float16)[..., ::2]
    elif bad == "gradient":
        value.requires_grad_(True)
    elif bad == "device":
        value = torch.empty((2, 4, 256), dtype=torch.float16, device="meta")
    monkeypatch.setattr(compact_indexer, "validate_target_indexer_frame", lambda *args: pytest.fail("invalid input reached frame work"))
    monkeypatch.setattr(forward, "_kv_begin", lambda *args: pytest.fail("invalid input mutated cache state"))
    with pytest.raises(ValueError, match="pipeline HC") as caught:
        forward.compute_stage(w, [(object(), 0, 2)], b, input_hc=value)
    assert failures == [caught.value]


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_boundary_half_storage_matches_next_hc_mode_zero_statistics(mode):
    from tensorfold.families.qwen4_exp.cuda import glue

    gen = torch.Generator().manual_seed(7103 + mode)
    h = (torch.randn((3, 4 * 256), generator=gen) * 2).half()
    inject = torch.randn((3, 4), generator=gen).half()
    pss = torch.zeros((3, 1, 4))
    branch = torch.randn((8, 3, 256) if mode in (3, 4) else (3, 256), generator=gen)
    y, wts = torch.randn((3, 11, 256), generator=gen), torch.randn((3, 11), generator=gen)
    args = {"inject": inject, **({"y": y, "wts": wts} if mode == 2 else {"branch": branch})}
    complete = h.clone()
    glue.hc_writeback(complete, complete, pss, 4, mode, **args)
    received, received_pss = complete.clone(), torch.zeros_like(pss)
    glue.hc_writeback(received, received, received_pss, 4, 0)
    assert torch.equal(complete.view(torch.int16), received.view(torch.int16))
    assert torch.equal(pss, received_pss)


def state_weights(index):
    from tests.test_sm70_rotorquant_state import model as cache_model

    w = cache_model(8, mtp=False)
    stage = PipelineStage(index)
    kinds = ["linear" if i % 4 != 3 else "attention" for i in range(48)]
    w.cfg.layers, w.cfg.layer_types = 48, kinds
    w.cfg.quant, w.cfg.execution_dtype = "gguf", "float16"
    for key, value in dict(heads=3, index_heads=4, index_budget=2048, low=320, experts=512, top_k=10,
                           moe_width=80, shared_width=80, heads_per_ngram=8, ple_dim=512, context=262144, vocab=1024).items():
        setattr(w.cfg, key, value)
    w.layers = [NS(index=i, linear=kinds[i] == "linear", ple=None) for i in stage.layer_ids]
    w.embed = (object(),) if stage.owns_input else None
    w.mixer, w.head = (None, None) if stage.owns_input else (object(), NS(n=128))
    w.x3 = None
    w.comm, w.draft_ids = None, None
    w.meta.update(rank=0, execution_provider="sm70", execution_dtype="float16", history_layout="unique",
                  attention_degree=4, indexer_history="gpu", pipeline_stage=stage.identity(),
                  full=NS(layers=48, layer_types=kinds, kv_heads=2, moe_width=640))
    return w


def test_actual_local_state_maps_and_numerical_identity_keep_global_layer_span():
    from tensorfold.families.qwen4_exp.cuda.state import State

    states = []
    for index in (0, 1):
        w = state_weights(index)
        state = State(w, 16, 4, "int8", limit=16)
        assert len(state.lin_index) == state.rec.shape[1] == 18
        assert len(state.att_index) == len(state.kc) == 6
        assert set(state.lin_index) | set(state.att_index) == set(PipelineStage(index).layer_ids)
        assert state.history_placement.target_layers == 6
        state.restore(state.snapshot())
        states.append(state)
    assert states[0].kv_identity != states[1].kv_identity
    for destination, source in ((states[0], states[1]), (states[1], states[0])):
        with pytest.raises(ValueError):
            destination.restore(source.snapshot())
        with pytest.raises(ValueError):
            destination.copy_prefix(source, 0, 0)


@pytest.mark.parametrize("index", [0, 1])
def test_actual_stage_buffers_allocate_no_input_stage_head_storage(index):
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    b = Buffers(state_weights(index), 3, 16)
    assert b.pipeline_stage == PipelineStage(index)
    if index == 0:
        assert b.logits.numel() == b.cand.numel() == b.cand_all.numel() == 0
    else:
        assert b.logits.shape == (3, 128) and b.cand.numel() > 0


@pytest.mark.parametrize("index", [0, 1])
def test_actual_padded_stage_layout_uses_local_recurrent_count_without_changing_global_config(index, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import padded_linears
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    # Isolate the CUDA selection gate, retaining actual padding allocation and
    # layout/shape checks on CPU. This is not a CUDA kernel/profile qualification.
    monkeypatch.setattr(padded_linears, "create", lambda w, rows, **kwargs: padded_linears.PaddedLinears(
        w, rows, **{k: v for k, v in kwargs.items() if k in ("head_columns", "linear_layers")}))
    w = state_weights(index)
    b = Buffers(w, 3, 16)
    assert w.cfg.layers == len(w.cfg.layer_types) == 48
    assert b.proj.shape[0] == b.padded_linears.layout["proj"][0][0] == 18
    assert b.logits.numel() == (0 if index == 0 else 3 * 128)
    head_width = 0 if index == 0 else 128
    actual_extra = 0
    for shape, size in b.padded_linears.layout.values():
        if len(shape) == 2:
            actual_extra += (32 - shape[0]) * shape[1] * size
        else:
            actual_extra += shape[0] * (32 - shape[1]) * shape[2] * size
    actual_extra += b.padded_linears.input_storage.numel() * b.padded_linears.input_storage.element_size()
    actual_extra += sum(t.numel() * t.element_size() for t in b.padded_linears.partial.values())
    assert padded_linears.workspace_bytes(w.cfg, 3, head_width, 8, enabled=True, linear_layers=18) == actual_extra
    c = w.cfg
    text = dict(hidden_size=c.hidden, hc_count=c.streams, hc_lowrank=c.low, num_attention_heads=c.heads * 8,
                num_key_value_heads=2, head_dim=c.head_dim, indexer_n_heads=c.index_heads,
                indexer_head_dim=c.index_dim, linear_num_key_heads=c.nk * 8, linear_num_value_heads=c.nv * 8,
                ple_embed_dim=c.ple_dim, vocab_size=c.vocab,
                layer_types=["linear_attention" if kind == "linear" else "full_attention" for kind in c.layer_types])
    assert padded_linears.workspace_bytes_from_text(text, 3, 8, enabled=True,
                                                    _pipeline_stage=PipelineStage(index)) == actual_extra
