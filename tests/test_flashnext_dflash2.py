"""Independent DFlash2 metadata/probability/window/state integration controls.

CPU controls do not qualify actual TP8 feature capture or V100 kernels/quality.
The optional full-file loader check uses explicitly supplied pinned artifact data.
"""
from dataclasses import replace
from contextlib import contextmanager
import json
import hashlib
import math
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import ProbabilityRow
from tensorfold.families.qwen4_exp.cuda.dflash2_config import DFlash2Config
from tensorfold.families.qwen4_exp.cuda.dflash2_features import FeatureFrame
from tensorfold.families.qwen4_exp.cuda import dflash2_ops as ops
from tensorfold.families.qwen4_exp.cuda.dflash2_provider import DFlash2Provider, selector_chain, selector_lattice
from tensorfold.families.qwen4_exp.cuda.dflash2_runtime import DFlash2Runtime, DraftWindow
from tensorfold.families.qwen4_exp.cuda.dflash2_weights import DFlash2Plan, load_shard, plan_weights, tensor_shapes

FIXTURES = Path(__file__).with_name("fixtures") / "dflash2"


def initial_config():
    return json.loads((FIXTURES / "config.json").read_text())


def initial_header():
    raw = (FIXTURES / "header.bin").read_bytes()
    if len(raw) < 8 or struct.unpack("<Q", raw[:8])[0] != len(raw) - 8:
        raise ValueError("the reference fixture must contain one exact safetensors header prefix")
    return json.loads(raw[8:])


def small_config():
    raw = initial_config()
    raw.update(hidden_size=32, intermediate_size=64, vocab_size=32, num_hidden_layers=2,
               num_target_layers=6, num_attention_heads=8, num_key_value_heads=8, head_dim=4,
               layer_types=["sliding_attention"] * 2, max_window_layers=2, sliding_window=8,
               max_position_embeddings=4096)
    raw["dflash_config"].update(target_layer_ids=[1, 3, 5], mask_token_id=31, selector_rank=4, selector_top_k=4)
    return raw


def make_weights(directory, *, mutate=None, raw=None):
    raw = small_config() if raw is None else raw
    c = DFlash2Config.from_dict(raw)
    generator = torch.Generator().manual_seed(173)
    values = {}
    for name, shape in tensor_shapes(c).items():
        tensor = (torch.randn(shape, generator=generator) * .03).bfloat16()
        if "norm.weight" in name:
            tensor.fill_(1)
        if name.endswith("base_kernel"):
            tensor.zero_()
            tensor[:, 0].fill_(1)
        values[name] = tensor
    if mutate is not None:
        mutate(values)
    (directory / "config.json").write_text(json.dumps(raw))
    save_file(values, directory / "model.safetensors")
    return c, values


class Embedding:
    def __init__(self, weight):
        self.weight = weight

    def lookup(self, ids, *, copies, out):
        out.copy_(self.weight.index_select(0, ids.long()).repeat(1, copies))
        return out


class Head:
    def __init__(self, weight):
        self.weight = weight

    def matmul(self, hidden, *, f32):
        return ops.linear(hidden, self.weight, fp32=f32)


def target(c):
    generator = torch.Generator().manual_seed(191)
    embed = Embedding((torch.randn(c.vocab, c.hidden, generator=generator) * .05).half())
    head = Head((torch.randn(c.vocab, c.hidden, generator=generator) * .05).half())
    cfg = SimpleNamespace(hidden=c.hidden, vocab=c.vocab, layers=c.target_layers, streams=4)
    return SimpleNamespace(cfg=cfg, meta={"full": cfg, "vocab_offset": 0}, embed=(embed,), head=head,
                           comm=None, device=torch.device("cpu"))


def test_real_independent_header_matches_exact_tp8_payload_and_replicated_codebooks():
    c = DFlash2Config.from_dict(initial_config())
    header = initial_header()
    assert len(header) == 81
    assert sum(row["data_offsets"][1] - row["data_offsets"][0] for row in header.values()) == 1206131200
    for rank in range(8):
        plans = plan_weights(c, header, tp=(rank, 8))
        assert len(plans) == 81 and sum(row.nbytes for row in plans) == 460659200
        assert sum(row.nbytes for row in plans if "codebook" in row.name) == 254279680
        assert c.state_bytes(tp=8, context=262144) == 5242880
        assert c.state_bytes(tp=8, context=1024) == 2621440
    assert c.block == 4 and c.source_block == 8 and c.taps == (5, 15, 25, 35, 45)


@pytest.mark.parametrize("mutation", ["causal", "width3", "full-layer", "bad-tap", "theta-bool", "kv-shard", "v-scale"])
def test_unsupported_manifests_fail_before_weight_allocation(mutation):
    raw = initial_config()
    if mutation == "causal":
        raw["is_causal"] = True
    elif mutation == "width3":
        raw["dflash_config"]["conv_kernel_size"] = 3
    elif mutation == "full-layer":
        raw["layer_types"][0] = "full_attention"
    elif mutation == "bad-tap":
        raw["dflash_config"]["target_layer_ids"][-1] = 48
    elif mutation == "theta-bool":
        raw["rope_parameters"]["rope_theta"] = True
    elif mutation == "kv-shard":
        c = DFlash2Config.from_dict(raw)
        with pytest.raises(ValueError, match="TP"):
            c.partition(0, 3)
        return
    else:
        raw["dflash_config"]["attention_value_scale"] = .5
    with pytest.raises(ValueError):
        DFlash2Config.from_dict(raw)


def test_source_namespace_and_shared_target_shape_are_hard_boundaries():
    c = DFlash2Config.from_dict(initial_config())
    header = initial_header()
    header["embed_tokens.weight"] = {"dtype": "BF16", "shape": [c.vocab, c.hidden]}
    with pytest.raises(ValueError, match="namespace"):
        plan_weights(c, header)
    with pytest.raises(ValueError, match="hidden"):
        c.check_target(SimpleNamespace(hidden=42, vocab=c.vocab, layers=48, streams=4))


def test_bounded_f16_loader_keeps_all_shard_arrays_f16_and_shares_target_objects(tmp_path):
    c, _ = make_weights(tmp_path)
    plan = DFlash2Plan.read(tmp_path, tp=(0, 1))
    loaded = load_shard(plan, device="cpu", chunk_bytes=4096)
    prepared = load_shard(plan, device="cpu", chunk_bytes=8192)
    assert loaded.nbytes == plan.resident_bytes
    assert loaded.identity == prepared.identity and loaded.shard_identity == prepared.shard_identity
    assert all(value.dtype == torch.float16 for value in loaded.tensors.values())
    w = target(c)
    provider = DFlash2Provider(loaded, w)
    assert provider.target_embedding is w.embed and provider.target_head is w.head
    assert not any(name.startswith(("embed", "lm_head")) for name in loaded.tensors)
    resources = plan.resources(slots=40, context=4096, prefill_rows=32, execution_rows=32, keep=8, points=4)
    assert resources["state_bytes"] == 40 * c.state_bytes(tp=1, context=4096)
    assert resources["target_shared_weight_bytes"] == 0
    assert resources["snapshot_versions"] == 168
    assert resources["snapshot_bytes"] == 168 * c.state_bytes(tp=1, context=4096)


def test_source_hash_fallback_bounds_a_continuously_growing_input_before_tensor_allocation(tmp_path, monkeypatch):
    from tensorfold.engine import model_identity

    make_weights(tmp_path)
    plan = DFlash2Plan.read(tmp_path, tp=(0, 1))
    original, requests = model_identity.os.read, []
    extent = plan.path.stat().st_size

    def growing(fd, size):
        value = original(fd, size)
        requests.append(size)
        with plan.path.open("ab") as stream:
            stream.write(b"growing source" * 32)
        return value

    monkeypatch.setattr(model_identity.os, "read", growing)
    monkeypatch.setattr(torch, "empty", lambda *_a, **_k: pytest.fail("growing input reached tensor allocation"))
    with pytest.raises(ValueError, match="grew|changed"):
        load_shard(plan, device="cpu", chunk_bytes=4096)
    assert sum(requests) <= extent + 1
    assert requests[-1] == 1 and len(requests) <= (extent + model_identity._READ_BYTES - 1) // model_identity._READ_BYTES + 1


def test_config_capture_bounds_growth_and_never_uses_an_unbounded_second_read(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import dflash2_weights as loader

    make_weights(tmp_path)
    path = tmp_path / "config.json"
    extent, reads = path.stat().st_size, []
    original = loader._regular_stream

    class Growing:
        def __init__(self, stream):
            self.stream = stream

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            value = self.stream.read(size)
            with path.open("ab") as stream:
                stream.write(b" " * 1024)
            return value

    @contextmanager
    def opened(selected):
        with original(selected) as stream:
            yield Growing(stream)

    monkeypatch.setattr(loader, "_regular_stream", opened)
    monkeypatch.setattr(loader, "read_header", lambda *_a, **_k: pytest.fail("changed config reached tensor header"))
    with pytest.raises(ValueError, match="config changed"):
        DFlash2Plan.read(tmp_path, tp=(0, 1))
    assert reads == [extent + 1]


def test_config_hash_parse_and_header_admission_reject_separate_versions(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import dflash2_weights as loader

    make_weights(tmp_path)
    original = loader._metadata_object

    def changed(raw, path):
        result = original(raw, path)
        Path(path).write_bytes(raw + b" ")
        return result

    with monkeypatch.context() as scope:
        scope.setattr(loader, "_metadata_object", changed)
        with pytest.raises(ValueError, match="parse/hash"):
            DFlash2Plan.read(tmp_path, tp=(0, 1))
    plan = DFlash2Plan.read(tmp_path, tp=(0, 1))
    assert plan.config_sha256 == hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    with (tmp_path / "config.json").open("ab") as stream:
        stream.write(b" ")
    monkeypatch.setattr(torch, "empty", lambda *_a, **_k: pytest.fail("changed config reached tensor allocation"))
    with pytest.raises(ValueError, match="config changed after"):
        load_shard(plan, device="cpu", chunk_bytes=4096)


def test_config_version_is_held_through_tensor_header_planning(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import dflash2_weights as loader

    make_weights(tmp_path)
    original = loader.read_header

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with (tmp_path / "config.json").open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(loader, "read_header", changed)
    with pytest.raises(ValueError, match="config changed across"):
        DFlash2Plan.read(tmp_path, tp=(0, 1))


@pytest.mark.parametrize("kind", ["duplicate-key", "oversize"])
def test_same_open_config_capture_preserves_existing_strict_json_boundary(tmp_path, monkeypatch, kind):
    from tensorfold.families.qwen4_exp.cuda import dflash2_weights as loader

    make_weights(tmp_path)
    path = tmp_path / "config.json"
    if kind == "duplicate-key":
        path.write_bytes(b'{"hidden_size":32,"hidden_size":64}')
        message = "duplicate"
    else:
        monkeypatch.setattr(loader, "MAX_HEADER_BYTES", path.stat().st_size - 1)
        message = "byte bound"
    with pytest.raises(ValueError, match=message):
        DFlash2Plan.read(tmp_path, tp=(0, 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 65536.])
def test_source_conversion_rejects_nonfinite_or_f16_overflow(tmp_path, bad):
    def mutate(values):
        values["fc.weight"][0, 0] = bad
    make_weights(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="nonfinite or overflows"):
        load_shard(DFlash2Plan.read(tmp_path, tp=(0, 1)), device="cpu", chunk_bytes=4096)


def test_two_tap_convolution_matches_independent_loop_and_resets_each_block():
    rows, width, groups = 8, 32, 2
    x = torch.arange(rows * width, dtype=torch.float16).reshape(rows, width) / 32
    coeff = torch.empty(rows, 2, groups, dtype=torch.float16)
    coeff[:, 0].fill_(.125)
    coeff[:, 1].fill_(.25)
    base = torch.empty(2, 2, width, dtype=torch.float16)
    base[:, 0].fill_(.5)
    base[:, 1].fill_(.25)
    actual = ops.grouped_conv(x, coeff, base, side=0, block=4, group_size=16)
    expected = np.empty((rows, width), dtype=np.float16)
    for row in range(rows):
        for column in range(width):
            result = float(x[row, column]) * (.5 + float(coeff[row, 0, column // 16]))
            if row % 4:
                result += float(x[row - 1, column]) * (.25 + float(coeff[row, 1, column // 16]))
            expected[row, column] = result
    assert np.array_equal(actual.numpy(), expected)


def test_noncausal_block_mask_has_exact_2048_style_left_boundary_and_keeps_absolute_positions():
    positions = torch.arange(10, 14)
    past = torch.arange(2, 10)
    actual = ops.window_mask(positions, past, window=8)
    expected = [[key >= query - 7 for key in range(2, 10)] + [True] * 4 for query in range(10, 14)]
    assert actual.tolist() == expected
    x = torch.ones(4, 1, 4, dtype=torch.float16)
    assert torch.equal(ops.rope(x, torch.zeros(4, dtype=torch.long), 10000), x)
    assert not torch.equal(ops.rope(x, torch.arange(2048, 2052), 10000), ops.rope(x, torch.arange(4), 10000))


def lattice_fixture():
    ids = torch.tensor([[[1, 2], [3, 4]]])
    unary = torch.tensor([[[4., 0.], [0., 0.]]])
    projected = torch.tensor([[[1., 0.], [1., 0.]]], dtype=torch.float16)
    predecessor = torch.zeros(6, 2, dtype=torch.float16)
    predecessor[0, 0], predecessor[1, 0], predecessor[2, 0] = 1, 1, -1
    successor = torch.zeros_like(predecessor)
    successor[1, 0], successor[2, 0], successor[3, 0], successor[4, 0] = -3, 3, 2, -2
    lattice = selector_lattice(ids, unary, projected, predecessor, successor, torch.tensor([0]))
    return ids[0].numpy(), lattice[0].numpy()


def test_selector_uses_conditional_codebook_edges_instead_of_base_logits():
    ids, lattice = lattice_fixture()
    assert lattice.tolist() == [[[1., 3.], [1., 3.]], [[2., -2.], [-2., 2.]]]
    packet = selector_chain(ids, lattice, sampling=None, first_position=11)
    assert packet.tokens == (2, 4)
    assert [row.ids.tolist() for row in packet.q_rows] == [[2], [4]]
    assert packet.first_position == 11


def test_actual_stochastic_q_and_request_keying_feed_standard_rejection():
    ids, lattice = lattice_fixture()
    s = Sampling(17, temperature=.7, top_k=1, top_p=.01, min_p=.99)
    packet = selector_chain(ids, lattice, sampling=s, first_position=111)
    assert all(len(row.ids) == 2 for row in packet.q_rows)  # q is selector softmax, not target filtering
    assert packet == selector_chain(ids, lattice, sampling=s, first_position=111)
    for index, row in enumerate(packet.q_rows):
        previous = 0 if index == 0 else int(np.flatnonzero(ids[index - 1] == packet.tokens[index - 1])[0])
        scores = lattice[index, previous]
        ratio = row.probability(int(ids[index, 0])) / row.probability(int(ids[index, 1]))
        assert math.isclose(ratio, math.exp(float(scores[0] - scores[1]) / .7), rel_tol=8 * np.finfo(np.float64).eps)
    accepted, bonus = DFlash2Provider.verify(packet, [*packet.q_rows, ProbabilityRow([5], [1])], s, room=4)
    assert accepted == [0, 1, 2] and bonus == 5
    accepted, correction = DFlash2Provider.verify(packet,
        [ProbabilityRow([5], [1]), ProbabilityRow([5], [1]), ProbabilityRow([5], [1])], s, room=4)
    assert accepted == [0] and correction == 5


def test_ring_wrap_snapshot_and_rejection_do_not_copy_or_consume_uninitialized_history():
    c = DFlash2Config.from_dict(small_config())
    ring = DraftWindow(c, tp=(0, 8), context=64, device="cpu", generation=7)
    assert ring.snapshot()["kv"].shape[2] == 0
    values = torch.arange(2 * c.layers * 12 * c.head_dim, dtype=torch.float16).reshape(2, c.layers, 12, 1, c.head_dim)
    ring.append(values, 0)
    content, positions = ring.layer(0)
    assert positions.tolist() == list(range(4, 12))
    assert torch.equal(content, values[:, 0, -8:])
    snapshot = ring.snapshot()
    before = ring.kv.clone()
    # Tentative proposal buffers are independent; constructing them cannot
    # overwrite the ring before an accepted target feature append.
    tentative = torch.full((2, c.layers, 4, 1, c.head_dim), 99., dtype=torch.float16)
    assert torch.equal(ring.kv, before) and tentative.data_ptr() != ring.kv.data_ptr()
    ring.append(tentative[:, :, :2], 12)
    ring.restore(snapshot)
    assert ring.end == 12 and ring.generation == 7
    assert torch.equal(ring.layer(0)[0], values[:, 0, -8:])
    with pytest.raises(ValueError, match="frontier"):
        ring.append(tentative, 10)
    changed_profile = DraftWindow(c, tp=(0, 8), context=64, device="cpu", generation=7)
    with pytest.raises(ValueError, match="profile"):
        changed_profile.restore(snapshot)


def test_bounded_feature_transaction_commits_only_accepted_rows_and_maintains_zero_proposal(tmp_path):
    c, _ = make_weights(tmp_path)
    w = target(c)
    runtime = DFlash2Runtime(w, DFlash2Plan.read(tmp_path, tp=(0, 1)), context=64)
    st = SimpleNamespace(pos=0, slot_generation=0, dflash2=None)
    runtime.init_state(st)
    b = SimpleNamespace(rows=4)
    runtime.prepare_buffers(b)
    capture = runtime.begin([(st, 0, 4)], b)
    capture.owner.features.copy_(torch.arange(4 * len(c.taps) * c.hidden, dtype=torch.float16).reshape_as(capture.owner.features) / 256)
    capture.seen = list(c.taps)  # feature math has its independent completed-HC test below
    runtime.finish(capture, [(st, 0, 4)])
    with pytest.raises(ValueError, match="unresolved|await"):
        runtime.begin([(st, 0, 4)], b)
    runtime.commit(st, 2)
    st.pos = 2
    assert st.dflash2.end == 2 and b.dflash2_features.readers == 0
    packet = runtime.propose(st, 1, 0, Sampling(11), 3)
    assert packet.tokens == () and st.dflash2.end == st.pos
    previous = st.dflash2.snapshot()
    capture = runtime.begin([(st, 0, 2)], b)
    capture.owner.features[:2].fill_(.01)
    capture.seen = list(c.taps)
    runtime.finish(capture, [(st, 0, 2)])
    assert torch.equal(st.dflash2.snapshot()["kv"], previous["kv"])
    runtime.commit(st, 0)
    assert st.dflash2.end == 2 and torch.equal(st.dflash2.snapshot()["kv"], previous["kv"])
    with pytest.raises(RuntimeError, match="live windows"):
        runtime.close()
    runtime.close_state(st)
    runtime.close()


def test_tiny_complete_backbone_batch_has_true_selector_packets(tmp_path):
    c, _ = make_weights(tmp_path)
    loaded = load_shard(DFlash2Plan.read(tmp_path, tp=(0, 1)), device="cpu", chunk_bytes=4096)
    provider = DFlash2Provider(loaded, target(c))
    states = [DraftWindow(c, tp=(0, 1), context=64, device="cpu") for _ in range(2)]
    packets = provider.propose_many(states, [1, 2], [1, 3], [Sampling(17), None], [1, 1])
    assert [len(packet.tokens) for packet in packets] == [1, 3]
    assert all(row.probability(token) > 0 for packet in packets for token, row in zip(packet.tokens, packet.q_rows))


def test_runtime_identity_scopes_snapshots_even_when_content_profile_digest_is_equal(tmp_path):
    c, _ = make_weights(tmp_path)
    plan = DFlash2Plan.read(tmp_path, tp=(0, 1))
    first = DFlash2Runtime(target(c), plan, context=64, profile_identity="same-verified-profile")
    other = DFlash2Runtime(target(c), plan, context=64, profile_identity="same-verified-profile")
    a, b, foreign = [SimpleNamespace(pos=0, slot_generation=0, dflash2=None) for _ in range(3)]
    first.init_state(a)
    first.init_state(b)
    other.init_state(foreign)
    first.clone_state(a, b)
    with pytest.raises(ValueError, match="profile"):
        other.restore(foreign, first.snapshot(a))
    for runtime, states in ((first, (a, b)), (other, (foreign,))):
        for state in states:
            runtime.close_state(state)
        runtime.close()


def test_feature_lease_covers_other_slots_and_rejects_duplicate_or_stale_publication(tmp_path):
    c, _ = make_weights(tmp_path)
    runtime = DFlash2Runtime(target(c), DFlash2Plan.read(tmp_path, tp=(0, 1)), context=64)
    a, b = [SimpleNamespace(pos=0, slot_generation=0, dflash2=None) for _ in range(2)]
    runtime.init_state(a)
    runtime.init_state(b)
    buffers = SimpleNamespace(rows=4)
    runtime.prepare_buffers(buffers)
    capture = runtime.begin([(a, 0, 2)], buffers)
    capture.owner.features[:2].zero_()
    capture.seen = list(c.taps)
    runtime.finish(capture, [(a, 0, 2)])
    with pytest.raises(ValueError, match="unresolved"):
        runtime.begin([(b, 0, 1)], buffers)
    with pytest.raises(ValueError, match="already published"):
        runtime.finish(capture, [(a, 0, 2)])
    with pytest.raises(ValueError, match="repeats"):
        runtime.commit_many([(a, 1), (a, 1)])
    a.slot_generation = 1
    with pytest.raises(ValueError, match="stale"):
        runtime.commit(a, 1)
    runtime.discard(a)
    runtime.close_state(a)
    runtime.close_state(b)
    runtime.close()


def test_nonfinite_conditioning_poisoning_keeps_frontier_and_stops_publication(tmp_path):
    c, _ = make_weights(tmp_path)
    runtime = DFlash2Runtime(target(c), DFlash2Plan.read(tmp_path, tp=(0, 1)), context=64)
    state = SimpleNamespace(pos=0, slot_generation=0, dflash2=None)
    runtime.init_state(state)
    buffers = SimpleNamespace(rows=4)
    runtime.prepare_buffers(buffers)
    capture = runtime.begin([(state, 0, 1)], buffers)
    capture.owner.features[0].fill_(float("nan"))
    capture.seen = list(c.taps)
    runtime.finish(capture, [(state, 0, 1)])
    with pytest.raises(ValueError, match="finite FP16"):
        runtime.commit(state, 1)
    assert state.dflash2.poisoned and state.dflash2.end == 0
    with pytest.raises(ValueError, match="poisoned"):
        runtime.propose(state, 1, 0, Sampling(17), 1)
    runtime.close_state(state)
    runtime.close()


def test_partial_load_failure_remains_published_and_owned_until_close(tmp_path):
    def mutate(values):
        values["fc.weight"][0, 0] = float("nan")
    c, _ = make_weights(tmp_path, mutate=mutate)
    w = target(c)
    runtime = DFlash2Runtime()
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.load(w, DFlash2Plan.read(tmp_path, tp=(0, 1)), context=64)
    assert w.dflash2 is runtime and runtime.nbytes > 0
    assert runtime.identity is None
    runtime.close()
    assert w.dflash2 is None and runtime.nbytes == 0


def test_completed_hc_capture_applies_eight_rank_pending_partials_without_mutating_target(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import forward

    c = replace(DFlash2Config.from_dict(initial_config()), hidden=256)
    cfg = SimpleNamespace(hidden=256, streams=4)
    w = SimpleNamespace(cfg=cfg, meta={}, comm=None, layers=[SimpleNamespace(index=i) for i in range(48)], mixer=None)
    inject_values = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.float16)

    def embed(_weights, ids, _copies, out):
        out.copy_((ids.float()[:, None, None] + torch.arange(4)[None, :, None] / 8).expand(-1, -1, 256).flatten(1).half())
        return out

    def layer(layer, weights, _segs, buffers, rows, pending, **_kwargs):
        if pending is not None:
            forward._writeback(buffers.h, buffers, rows, weights.cfg, pending)
        partials = torch.empty(8, rows, 256, dtype=torch.float32)
        for rank in range(8):
            partials[rank].fill_((rank + 1) * (layer.index + 1) / 256)
        return 3, partials, None, inject_values.expand(rows, 4)

    def finish(weights, _mixer, buffers, rows, pending, **_kwargs):
        result = buffers.h.clone()
        forward._writeback(result, buffers, rows, weights.cfg, pending)
        return result

    monkeypatch.setattr(forward, "_embed", embed)
    monkeypatch.setattr(forward, "layer_forward", layer)
    monkeypatch.setattr(forward, "finish", finish)

    class Runtime:
        def __init__(self, rows):
            self.frame = FeatureFrame(c, rows, "cpu")

        def begin(self, segs, buffers):
            return self.frame.begin(segs, buffers)

        def finish(self, capture, _segs):
            self.features = capture.finish().clone()

        def abort(self, _capture, _segs):
            pass

    def run(ids, enabled):
        rows = len(ids)
        buffers = SimpleNamespace(ids=torch.tensor(ids, dtype=torch.int32), h=torch.empty(rows, 1024, dtype=torch.float16),
                                  pss=torch.empty(rows, 1, 4), prefill=False)
        w.dflash2 = Runtime(rows) if enabled else None
        result = forward.compute(w, [(SimpleNamespace(kv_status=None), 0, rows)], buffers)
        return result, None if w.dflash2 is None else w.dflash2.features

    baseline, _ = run([0, 1, 2], False)
    result, features = run([0, 1, 2], True)
    assert torch.equal(result, baseline)
    expected = np.empty((3, 5, 256), dtype=np.float16)
    for row in range(3):
        streams = np.array([row + stream / 8 for stream in range(4)], dtype=np.float16)
        tap = 0
        for layer_id in range(48):
            total = np.float32((layer_id + 1) / 256)
            for rank in range(1, 8):
                total = np.float32(total + np.float32((rank + 1) * (layer_id + 1) / 256))
            update = np.float16(total)
            for stream, injection in enumerate((.5, 1., 1.5, 2.)):
                streams[stream] = np.float16(np.float32(streams[stream]) + np.float32(np.float16(np.float32(update) * injection)))
            if layer_id in c.taps:
                expected[row, tap].fill(np.float16(streams.astype(np.float32).mean()))
                tap += 1
    assert np.array_equal(features.reshape_as(torch.from_numpy(expected)).numpy(), expected)
    separate = torch.cat([run([row], True)[1] for row in range(3)])
    assert torch.equal(features, separate)


@pytest.mark.skipif(not os.environ.get("TENSORFOLD_DFLASH2_ARTIFACT"), reason="optional exact large artifact was not supplied")
def test_actual_pinned_bf16_artifact_loads_one_exact_tp8_f16_shard():
    directory = os.environ["TENSORFOLD_DFLASH2_ARTIFACT"]
    filename = os.environ.get("TENSORFOLD_DFLASH2_TENSOR_FILE", "model.safetensors")
    plan = DFlash2Plan.read(directory, filename=filename,
        expected_sha256="d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f", tp=(0, 8))
    loaded = load_shard(plan, device="cpu")
    assert loaded.nbytes == 460659200 and len(loaded.tensors) == 81
    assert all(tensor.dtype == torch.float16 for tensor in loaded.device_tensors())
