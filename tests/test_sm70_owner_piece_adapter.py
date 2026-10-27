"""Canonical native authority and activation aliases across exact graph pieces."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import bound_graphs, multi_graphs  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, UNCAPTURED, FULL_COLLECTIVE  # noqa: E402


@pytest.mark.parametrize("kind", ["verify", "mtp"])
def test_only_canonical_activation_storage_is_borrowed_with_exact_private_rows(kind, monkeypatch):
    from tensorfold.cuda.sm70 import family_ops

    owner = object()
    source = NS(rows=32, moe=NS(plan=NS(prefill=True)), history_owner=owner,
                pa=torch.zeros(32, 17, dtype=torch.float16), attn_o=torch.zeros(32, 3, 8, dtype=torch.float16))
    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.unique, graph.native_owner = True, True
    graph.states = (object(),) * 24
    graph.dec = NS(w=NS(device="cpu", dspark=None, meta={"owner_workspaces": {"decode": owner, "mtp": owner}}),
                   buf=source, mbuf=source, capacity=262147)
    graph.buffers, graph.row_maps, graph.segment_maps, graph.map_events = {}, {}, {}, {}
    graph._resident = lambda size: None

    class Buffers:
        def __init__(self, weights, rows, capacity, **kwargs):
            self.history_owner = owner
            self.pa = torch.zeros(rows, 17, dtype=torch.float16)
            self.attn_o = torch.zeros(rows, 3, 8, dtype=torch.float16)

    empty = torch.empty
    monkeypatch.setattr(bound_graphs, "Buffers", Buffers)
    monkeypatch.setattr(torch, "empty", lambda shape, *, dtype, **kwargs: empty(shape, dtype=dtype))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    private = graph._buffer(kind)
    for name in ("pa", "attn_o"):
        alias, original = getattr(private, name), getattr(source, name)
        assert alias.shape[0] == 3 and alias.data_ptr() == original.data_ptr()
        assert alias.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        alias.fill_(7)
        assert bool((original[:3] == 7).all()) and not bool(original[3:].any())
    assert private.history_owner is owner and graph._buffer(kind) is private
    assert source.rows == 32 and len(graph.states) == 24


def test_eager_cut_uses_original_frames_while_ple_and_other_pieces_keep_bound_views(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import history_owner

    weights = NS(comm=None)
    graph = object.__new__(multi_graphs.RoundGraphs)
    graph.dec = NS(w=weights)
    canonical_state, view = object(), object()
    canonical, private, layer = NS(), NS(), object()
    original, bound = [(canonical_state, 0, 2)], [(view, 0, 2)]
    events = []
    monkeypatch.setattr(multi_graphs, "profile_replay", lambda g, comm: events.append(g))
    def attention(actual_layer, w, segments, buffers, rows, **kwargs):
        assert (actual_layer, w, segments, buffers, rows) == (layer, weights, original, canonical, 2)
        events.append("native-original-frame")
    def ple(actual_layer, w, segments, buffers):
        assert (segments, buffers) == (bound, private)
        events.append("private-ple")
    monkeypatch.setattr(history_owner, "attention_layer", attention)
    monkeypatch.setattr(multi_graphs, "ple_conv_streams", ple)
    pieces = NS(graphs=["first", "middle", "last"], inserts=[("owner_attention", layer), ("ple_conv", layer)])
    graph._replay_pieces(pieces, bound, private, owner_frame=(original, canonical))
    assert events == ["first", "native-original-frame", "middle", "private-ple", "last"]


@pytest.mark.parametrize("foreign", [False, True])
def test_frame_and_original_catalogue_are_checked_before_graph_or_recurrent_mutation(monkeypatch, foreign):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    state = NS(pos=17, mtp_len=13, capacity=262147, image_positions=None)
    owner = NS(native_collective=FULL_COLLECTIVE, native_states={} if foreign else {id(state): state},
               wait_native_reuse=lambda: pytest.fail("invalid frame borrowed native workspace"))
    source = NS(rows=32, rotor_lookup=False, owner_kernel_family=KERNEL_FAMILY,
                owner_graph_family=UNCAPTURED, owner_graphs=None, history_owner=owner)
    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.closed, graph.unique, graph.native_owner = False, True, True
    graph.dec = NS(depth=2, w=NS(), buf=source, live=lambda: 1)
    graph._prepare = lambda *args, **kwargs: pytest.fail("invalid source reached private graph staging")
    primary = ValueError("source frame revoked")
    def validate(buffers, segments):
        assert buffers is source and segments[0][0] is state
        raise primary
    monkeypatch.setattr(compact_indexer, "validate_target_indexer_frame", validate)
    with pytest.raises(RuntimeError if foreign else ValueError) as caught:
        graph._forward("verify", [(state, 0, 2)])
    if not foreign:
        assert caught.value is primary


def test_other_cohorts_decline_before_extra_native_or_private_staging_work():
    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.closed, graph.unique, graph.native_owner = False, True, True
    graph.dec = NS(buf=object())
    assert graph._forward("verify", [(object(), 0, 1), (object(), 1, 2)]) is None
    graph.dec.live = lambda: 2
    assert graph._forward("verify", [(object(), 0, 1)]) is None
