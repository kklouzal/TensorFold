"""CPU checks of the DeepSeek-V4.1 DSpark head loader pieces (decode tables, layout, index reader, verification)."""

from dataclasses import replace
import json
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import mxfp4_ds41 as mx
from tensorfold.families.deepseek_v4.cuda import dspark_weights as dw


def test_target_verify_validates_static_shape_and_masks_a_local_step_before_snapshot():
    from tensorfold.families.deepseek_v4.cuda.speculation import Speculation
    from tensorfold.families.deepseek_v4.cuda.target import Step, Target

    events = []

    class Hash:
        history = torch.empty(3, 4, dtype=torch.int64)
        max_tokens = 8

        def __call__(self, ids, _positions, valid, *_metadata):
            events.append(("hash", valid))
            return torch.empty(len(ids), 2, 24, dtype=torch.int64)

        def commit_all(self, *_args):
            events.append(("commit", None))

    target = Target.__new__(Target)
    target.g = SimpleNamespace(hc=4)
    target.slots = 3
    target.hash_state = Hash()
    target.rotor_cache = object()
    target.engram_trace = None
    target.embed = lambda ids: torch.empty(len(ids), 512, dtype=torch.float16)
    spec = Speculation.__new__(Speculation)
    spec._target = target
    spec.slots = 2
    spec.hash_state = target.hash_state
    spec.snapshot_hash = lambda step: events.append(("snapshot", step))
    staged = Step(ids=torch.arange(4), positions=torch.tensor([127, 128, 0, 1]),
                  valid=torch.ones(4, dtype=torch.bool), token_req=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
                  qsl=torch.tensor([0, 2, 4], dtype=torch.int32), qsl_cpu=[0, 2, 4], num_reqs=2,
                  req_slot=torch.tensor([0, 2] + [2] * 8, dtype=torch.int32),
                  block_table=torch.tensor([[0], [-1]], dtype=torch.int32), prefill=[], decode_index=None, spec=spec)

    with pytest.raises(ValueError, match="belongs to another target"):
        Target.begin(SimpleNamespace(g=target.g), staged)
    for kwargs, error in (
        ({"qsl_cpu": [0, 0, 4]}, "bounded token rows"),
        ({"qsl_cpu": [0, 2, 3]}, "bounded token rows"),
        ({"single_token_slots": True}, "bounded token rows"),
        ({"token_req": staged.token_req.long()}, "row metadata"),
        ({"req_slot": staged.req_slot[:1]}, "slot and page tables"),
        ({"image": staged.valid}, "text decode step"),
    ):
        with pytest.raises(ValueError, match=error):
            target.begin(replace(staged, **kwargs))
    assert events == []

    result = target.begin(staged)
    assert [name for name, _ in events] == ["hash", "snapshot"]
    updated = result["step"]
    assert updated is not staged and updated is events[1][1]
    assert updated.valid is events[0][1] is result["shared"]["rotor_valid"]
    assert updated.valid.tolist() == [True, True, False, False] and staged.valid.all()
    assert updated.ids is staged.ids and updated.req_slot is staged.req_slot

    events.clear()
    ordinary = replace(staged, spec=None)
    result = target.begin(ordinary)
    assert [name for name, _ in events] == ["hash", "commit"]
    assert result["step"] is ordinary and result["shared"]["rotor_valid"] is ordinary.valid


def e4m3fn(code):
    sign = -1.0 if code & 0x80 else 1.0
    e, m = (code >> 3) & 15, code & 7
    return sign * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def test_fp8_block_decode_every_code_and_scale():
    valid = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=np.uint8)
    codes = np.resize(valid, (32, 64))                           # every finite code, in two 32x32 blocks
    scales = np.array([[0, 140]], dtype=np.uint8)
    got = dw.fp8_block_decode(codes, scales)
    for (r, c), code in np.ndenumerate(codes):
        assert float(got[r, c]) == e4m3fn(int(code)) * 2.0 ** (int(scales[r // 32, c // 32]) - 127)
    with pytest.raises(ValueError):
        dw.fp8_block_decode(np.full((32, 32), 0x7F, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8))


def test_audit_counts_fp16_effects():
    exact = torch.tensor([1.0, 2.0 ** -20, 2.0 ** -26, 70000.0, 1 + 2.0 ** -12, -0.0])
    half, a = dw.audit_fp16("t", "F32", exact)
    widened, same_audit = dw.audit_fp16("t", "F32", exact, keep_widened=True)
    assert widened.dtype == torch.float32
    assert torch.equal(widened.view(torch.int32), half.float().view(torch.int32))
    assert same_audit == a
    assert (a.overflow, a.flushed, a.subnormal, a.nonzero) == (1, 1, 1, 5)
    assert a.inexact == 3          # overflow, flush and the 1 + 2^-12 rounding


def test_mxfp4_decode_table():
    codes = torch.arange(256, dtype=torch.uint8).view(1, 16, 16).reshape(16, 16)
    scales = torch.full((16, 1), 127 - 3, dtype=torch.uint8)
    got = mx.decode(codes, scales, torch.float64)
    for r in range(16):
        for j in range(16):
            b = int(codes[r, j])
            assert got[r, 2 * j] == mx.E2M1[b & 15] / 8 and got[r, 2 * j + 1] == mx.E2M1[b >> 4] / 8


def test_layout_partitions_every_rank():
    seen_experts = {}
    for rank in range(8):
        lay = dw.Layout(rank, 8, 4)
        assert lay.heads == 8 and len(lay.experts) == 64 and len(lay.expert_rows) == 576
        for e in lay.experts:
            seen_experts.setdefault(e, set()).add((lay.expert_rows.start, lay.expert_rows.stop))
    assert sorted(seen_experts) == list(range(128))
    assert all(len(v) == 4 and sorted(v)[0][0] == 0 and sorted(v)[-1][1] == 2304 for v in seen_experts.values())
    one = dw.Layout()
    assert one.heads == 64 and len(one.experts) == 128 and len(one.expert_rows) == 2304
    with pytest.raises(ValueError):
        dw.Layout(0, 8, 1)


def write_checkpoint(tmp_path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (kind, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": kind, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    body = json.dumps(header).encode()
    (tmp_path / "x.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"".join(blobs))
    index = {name: "x.safetensors" for name in tensors}
    index["embed"] = "missing.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return dw.Checkpoint(tmp_path)


def test_checkpoint_reader_index(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1, 2, 3], dtype=np.uint8)
    ck = write_checkpoint(tmp_path, {"mtp.0.a": ("F32", a), "mtp.0.b": ("U8", b)})
    assert np.array_equal(ck.array("mtp.0.a"), a) and ck.shape("mtp.0.b") == (3,)
    assert math.prod(ck.shape("mtp.0.a")) == 6


def test_dense_owned_blocks_decode_to_exact_full_bank_slices(tmp_path):
    name = "mtp.0.dense"
    finite_codes = [code for code in range(256) if code & 0x7f != 0x7f]
    codes = np.resize(np.array(finite_codes, dtype=np.uint8), (65, 70))
    scales = np.array([[0, 100, 120], [101, 117, 127], [106, 129, 130]], dtype=np.uint8)
    ck = write_checkpoint(tmp_path, {name + ".weight": ("F8_E4M3", codes), name + ".scale": ("F8_E8M0", scales)})
    full = dw.fp8_block_decode(codes, scales).half().view(torch.int16)
    for rows, cols in ((None, None), (slice(32, 64), None), (None, slice(32, 64)), (slice(32, 65), slice(64, 70))):
        part_codes, part_scales = dw._dense_fp8_arrays(ck, name, codes.shape, rows, cols)
        assert np.shares_memory(part_codes, ck.array(name + ".weight")) and not part_codes.flags.writeable
        assert np.shares_memory(part_scales, ck.array(name + ".scale")) and not part_scales.flags.writeable
        got = dw.fp8_block_decode(part_codes, part_scales).half()
        assert got.untyped_storage().nbytes() == got.numel() * got.element_size()
        assert torch.equal(got.view(torch.int16), full[rows or slice(None), cols or slice(None)])
    with pytest.raises(ValueError, match="invalid FP8 block ownership"):
        dw._dense_fp8_arrays(ck, name, codes.shape, slice(31, 64), None)


def test_dense_owned_blocks_validate_full_scale_metadata(tmp_path):
    name = "mtp.0.dense"
    codes = np.zeros((64, 64), dtype=np.uint8)
    scales = np.full((1, 2), 127, dtype=np.uint8)
    ck = write_checkpoint(tmp_path, {name + ".weight": ("F8_E4M3", codes), name + ".scale": ("F8_E8M0", scales)})
    with pytest.raises(ValueError, match="FP8 weight shape"):
        dw._dense_fp8_arrays(ck, name, (32, 64), slice(0, 32), None)
    with pytest.raises(ValueError, match="scales do not cover"):
        dw._dense_fp8_arrays(ck, name, codes.shape, slice(0, 32), None)


def test_greedy_accept():
    from tensorfold.families.deepseek_v4.cuda.dspark_head import greedy_accept
    drafts = torch.tensor([[1, 2, 3, 4, 5], [1, 9, 3, 4, 5], [7, 2, 3, 4, 5]])
    target = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    assert greedy_accept(drafts, target).tolist() == [5, 1, 0]


def test_greedy_draft_depth_keeps_full_projection_and_markov_prefix():
    from tensorfold.families.deepseek_v4.cuda.dspark_head import DSparkHead

    b, vocab = 3, 13
    anchors = torch.arange(2 * b)[::2]
    positions = torch.arange(2 * b)[1::2]
    slots = torch.arange(2 * b)[::2]
    hidden = torch.full((b * dw.BLOCK, dw.HIDDEN), 0.12345, dtype=torch.float32)
    normalized = hidden * 1.25
    base = torch.arange(b * dw.BLOCK * (vocab + 1), dtype=torch.float32).view(-1, vocab + 1)[:, 1:]
    norm_weight, embed = object(), object()
    state = DSparkHead.__new__(DSparkHead)
    state.device = torch.device("cpu")
    state.w = SimpleNamespace(norm=norm_weight)
    calls, steps = [], []

    def forward(ids, pos, slot_ids, embedding):
        assert (ids is anchors and pos is positions and slot_ids is slots and embedding is embed)
        assert all(not t.is_contiguous() for t in (ids, pos, slot_ids))
        calls.append("forward")
        return hidden

    def norm(h, weight):
        assert h is hidden and weight is norm_weight
        calls.append("norm")
        return normalized

    def head(h):
        assert h.dtype == torch.float16 and h.shape == (b * dw.BLOCK, dw.HIDDEN)
        assert torch.equal(h, normalized.half())
        calls.append("head")
        return base

    def step(rows, previous):
        assert rows.shape == (b, vocab) and rows.stride() == ((vocab + 1) * dw.BLOCK, 1)
        assert not rows.is_contiguous()
        calls.append("markov")
        steps.append((rows.clone(), previous.clone()))
        return (rows[:, 0].to(torch.int64) + previous * 3) % 97

    state.forward = forward
    state._norm = norm
    state._sharded = lambda: SimpleNamespace(step=step)
    args = anchors, positions, slots, embed, head
    full = state.draft(*args)
    assert full.shape == (b, dw.BLOCK) and full.dtype == torch.int64
    assert calls == ["forward", "norm", "head"] + ["markov"] * dw.BLOCK
    full_steps = steps.copy()
    for i, (rows, previous) in enumerate(full_steps):
        assert torch.equal(rows, base.view(b, dw.BLOCK, vocab)[:, i])
        assert torch.equal(previous, anchors if i == 0 else full[:, i - 1])

    for depth in range(1, dw.BLOCK + 1):
        calls.clear()
        steps.clear()
        drafted = state.draft(*args, depth=depth)
        assert drafted.dtype == full.dtype and torch.equal(drafted, full[:, :depth])
        assert calls == ["forward", "norm", "head"] + ["markov"] * depth
        for (rows, previous), (expected_rows, expected_previous) in zip(steps, full_steps[:depth], strict=True):
            assert torch.equal(rows, expected_rows) and torch.equal(previous, expected_previous)

    calls.clear()
    for depth in (0, -1, dw.BLOCK + 1, True, 3.0, None):
        with pytest.raises(ValueError, match="DSpark draft depth must be an integer"):
            state.draft(*args, depth=depth)
    assert not calls
