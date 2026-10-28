"""Real-source all-slot output and common six-State arithmetic recovery gates."""

from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, FORMATS, decode_rows  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import (  # noqa: E402
    GGUFPacked,
    GGUFExperts,
    GGUFDense,
)
from test_sm70_gguf_exact import bits  # noqa: E402

MODEL = Path(
    os.environ.get(
        "TF_FLASHNEXT_GGUF",
        "/models/huihui-v2/unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    )
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="optional SM70 actual Flash Next GGUF fixture",
)
dims, count = 2560, 10


@lru_cache(maxsize=1)
def checkpoint():
    if not MODEL.is_file():
        pytest.skip("optional historical Flash Next GGUF fixture")
    return GGUFCheckpoint.open(MODEL)


def read(tensor, start, rows):
    offset, n = tensor.row_span(start, rows)
    with tensor.path.open("rb", buffering=0) as stream:
        stream.seek(offset)
        value = stream.read(n)
    if len(value) != n:
        raise IOError("short source span")
    return np.frombuffer(value, np.uint8).reshape(rows, tensor.row_bytes).copy()


def make(layer, world):
    width = 640 // world
    first = 3 * width
    base = f"blk.{layer}."
    packed = []
    metadata = []
    for part in ["gate", "up", "down"]:
        tensor = checkpoint().tensors[base + f"ffn_{part}_exps.weight"]
        if part == "down":
            raw = np.concatenate([read(tensor, e * dims, dims) for e in range(count)])
            quantum, stored = FORMATS[tensor.ggml_type][1:]
            lo = first // quantum * stored
            hi = -(-(first + width) // quantum) * stored
            data = np.ascontiguousarray(raw[:, lo:hi])
            columns = width
            offset = first % quantum
        else:
            raw = np.concatenate(
                [read(tensor, e * 640 + first, width) for e in range(count)]
            )
            data = raw
            columns = dims
            offset = 0
        packed.append(
            GGUFPacked(
                torch.tensor(data, device="cuda"),
                tensor.ggml_type,
                data.shape[0],
                columns,
                offset,
            )
        )
        metadata.append(
            {
                "name": tensor.name,
                "encoding": tensor.encoding,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "retained_bytes": data.nbytes,
            }
        )
    shared = []
    for part in ["gate", "up", "down"]:
        tensor = checkpoint().tensors[base + f"ffn_{part}_shexp.weight"]
        if part == "down":
            raw = read(tensor, 0, dims)
            value = decode_rows(tensor, raw).astype(np.float16)[
                :, first : first + width
            ]
        else:
            raw = read(tensor, first, width)
            value = decode_rows(tensor, raw).astype(np.float16)
        shared.append(
            GGUFDense(torch.from_numpy(np.ascontiguousarray(value)).to("cuda"))
        )
    return GGUFExperts(*packed, shared, count=count, width=width, dims=dims), metadata


@pytest.fixture(autouse=True)
def release_test_owners():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[0, 2])
def actual_bank(request):
    bank, metadata = make(request.param, 8)
    bank.packed_rows = 32
    return bank, metadata


@pytest.mark.parametrize("rows", [1, 3, 6, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_actual_source_complete_routed_shared_outputs(actual_bank, rows, dtype):
    bank, _ = actual_bank
    torch.manual_seed(9201 + rows)
    x = (torch.randn(rows, 2560, device="cuda") * 0.1).half()
    ids = (torch.arange(rows * 11, device="cuda").reshape(rows, 11) + rows).remainder(
        10
    )
    ids[:, -1] = 10
    expected = bank.forward_reference(x, ids).to(dtype)
    actual = torch.empty_like(expected)
    bank.forward_exact(x, ids, actual)
    bits(actual, expected)
    # Every source row uses the original path independently as well.
    serial = torch.cat(
        [
            bank.forward_reference(x[i : i + 1], ids[i : i + 1]).to(dtype)
            for i in range(rows)
        ]
    )
    bits(actual, serial)


def committed_equal(a, b):
    # State.clone() canonically restores the active recurrent bank. Its parity
    # can differ while the committed state is byte-identical.
    assert a.pos == b.pos
    for index, parity in enumerate(a.cur):
        bits(a.rec[parity, index], b.rec[b.cur[index], index])
    for name in ["conv", "ple_tail"]:
        bits(getattr(a, name), getattr(b, name))
    for left, right in zip(a.kc, b.kc):
        for name in ["k", "v", "ks", "vs"]:
            bits(getattr(left, name)[: a.pos], getattr(right, name)[: b.pos])
    for left, right in zip(a.ikc, b.ikc):
        bits(left[: a.pos], right[: b.pos])
    for left, right in zip(a.pooled, b.pooled):
        bits(left[: a.pos // a.ratio], right[: b.pos // b.ratio])
    np.testing.assert_array_equal(a.ple_history, b.ple_history)


@pytest.mark.parametrize("cache", ["fp16", "int8"])
@pytest.mark.parametrize("lanes", [1, 6])
@pytest.mark.parametrize("expert_world", [8, 4])
def test_common_full_state_actual_banks_match_original_with_graph_partial_commit_and_prefix(
    cache, lanes, expert_world, monkeypatch
):
    from test_flashnext_sm70_family import _small_weights
    from test_sm70_gguf_prepared import dense_fixture
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import padded_linears
    from tensorfold.families.qwen4_exp.cuda.decode import prefill
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage
    from tensorfold.families.qwen4_exp.cuda.multi import _slot
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    monkeypatch.setattr(padded_linears, "_ENABLED", True)
    w = dense_fixture(
        _small_weights(include_ple=True, hidden=2560, lowrank=320, head_dim=256)
    )
    w.cfg.experts, w.cfg.top_k = 10, 10
    w.cfg.moe_width = w.cfg.shared_width = 640 // expert_world
    for layer, source_layer in zip(w.layers, [0, 2]):
        layer.moe.experts = make(source_layer, expert_world)[0]
        layer.moe.experts.packed_rows = 32
        layer.moe.experts.implementation = "reference"
        layer.moe.router = (torch.randn(11, 2560, device="cuda") * 0.03).half().float()
    count = 3 if lanes == 1 else 4
    rows = lanes * count
    rb = Buffers(w, rows, 64, moe_prefill=True)
    nb = Buffers(w, rows, 64, moe_prefill=True)
    pb = Buffers(w, 12, 64, prefill=True)
    states = [State(w, 64, 4, cache) for _ in range(lanes)]
    for i, state in enumerate(states):
        prefill(
            _slot(w, state, nb, None, pb, 64, 12),
            [4, 17, 3, 99, 33, 2, 7, 5, 6],
            None,
            mtp=False,
        )
        if i % 2:
            state.rec[1 - state.cur[0]].copy_(state.rec[state.cur[0]])
            state.cur = [1 - x for x in state.cur]
    for iteration in range(2):
        copies = [state.clone() for state in states]
        tokens = [
            [(lane * 13 + i + iteration * 17) % w.cfg.vocab for i in range(count)]
            for lane in range(lanes)
        ]
        for layer in w.layers:
            layer.moe.experts.implementation = "reference"
        ref_segs = stage(w, rb, list(zip(copies, tokens)))
        expected = compute(w, ref_segs, rb, context=256).clone()
        stream_expected = rb.streams[:rows].clone()
        for layer in w.layers:
            layer.moe.experts.implementation = "packed_exact"
        segs = stage(w, nb, list(zip(states, tokens)))
        compute(w, segs, nb, context=256)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compute(w, segs, nb, context=256)
        graph.replay()
        bits(actual, expected)
        bits(nb.streams[:rows], stream_expected)
        for lane, ((state, start, end), copied) in enumerate(zip(segs, copies)):
            keep = 1 + lane % count
            commit(w, state, nb, count, keep, at=start)
            commit(w, copied, rb, count, keep, at=start)
            committed_equal(state, copied)
        del graph
    original = states[0]
    restored = State(w, 64, 4, cache)
    restored.copy_prefix(original, original.pos, 0)
    restored.restore(original.snapshot())
    committed_equal(original, restored)
    for layer in w.layers:
        layer.moe.experts.implementation = "reference"
    ref_segs = stage(w, rb, [(original, [31])])
    expected = compute(w, ref_segs, rb, context=256).clone()
    for layer in w.layers:
        layer.moe.experts.implementation = "packed_exact"
    segs = stage(w, nb, [(restored, [31])])
    bits(compute(w, segs, nb, context=256), expected)
    commit(w, original, rb, 1, 1)
    commit(w, restored, nb, 1, 1)
    committed_equal(original, restored)


# Current-checkpoint controls are opt-in independently of the historical MODEL
# default above. Plans, tensor encodings and TP views come from the maintained
# importer; no fixture assumes that a filename determines a tensor's encoding.
CURRENT_ROWS = (1, 2, 3, 26, 30, 31, 32, 33)


def _current_paths():
    names = ("TF_FLASHNEXT_GGUF", "TF_FLASHNEXT_MTP_GGUF")
    values = tuple(os.environ.get(name) for name in names)
    if not all(values):
        pytest.skip("current target/MTP controls require both explicit GGUF environment paths")
    paths = tuple(Path(value).resolve(strict=True) for value in values)
    if not all(path.is_file() for path in paths):
        pytest.fail("explicit current target/MTP inputs must be regular GGUF files")
    return paths


@lru_cache(maxsize=8)
def current_context(rank):
    """Host-only planning also supplies a bounded inventory extraction seam."""
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan
    from tensorfold.families.qwen4_exp.cuda.gguf_mtp_weights import model_plan as mtp_plan

    target_path, mtp_path = _current_paths()
    target, _ = model_plan(target_path, tp=(rank, 8))
    embedding, head = target.embed, target.head
    mtp, planner, receipt = mtp_plan(target, mtp_path, tp=(rank, 8))
    assert target.embed is embedding and target.head is head
    assert receipt["shared_target_tensors"] == {"embedding": "target.embed", "output": "target.head"}
    assert not {"token_embd.weight", "output.weight"} & {p.source.name for p in planner.matrices}
    assert target.cfg.execution_dtype == "float16"
    assert (target.cfg.hidden, target.cfg.moe_width) == (2560, 80), "current packed-native geometry required"
    assert target.cfg.experts > 1
    return target, mtp, receipt


def current_banks(role, rank):
    target, mtp, _ = current_context(rank)
    layers = target.layers if role == "target" else (mtp.layer,)
    groups = {}
    for layer in layers:
        bank = layer.moe.experts
        key = (layer.linear, *(p.source.ggml_type for p in (bank.gate, bank.up, bank.down)))
        groups.setdefault(key, []).append((layer.index, bank))
    # First and last weights in each observed arithmetic/layout class, rather
    # than every layer crossed with every row count and every input pattern.
    return tuple(case for group in groups.values() for case in
                 ((group[0],) if len(group) == 1 else (group[0], group[-1])))


def current_concats(role, rank):
    """One actual weight stack per observed shape, format and execution dtype."""
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import ConcatPlan

    target, mtp, _ = current_context(rank)
    layers = target.layers if role == "target" else (mtp.layer,)
    selected = {}
    for layer in layers:
        candidates = [("hc-attn", layer.attn_hc.down, torch.float32),
                      ("hc-ffn", layer.mlp_hc.down, torch.float32)]
        branch = layer.gdn if layer.linear else layer.attn
        candidates.append(("gdn" if layer.linear else "qkv", branch.proj, torch.float16))
        for name, plan, dtype in candidates:
            assert isinstance(plan, ConcatPlan)
            key = (dtype, tuple((p.n, p.k, p.source.ggml_type) for p in plan.parts))
            selected.setdefault(key, (layer.index, name, plan, dtype))
    return tuple(selected.values())


def current_inventory():
    """Metadata only: callers may run with CUDA_VISIBLE_DEVICES empty."""
    return {role: {str(rank): {
        "banks": [{"layer": layer, "width": b.width, "hidden": b.dims, "experts": b.count,
                   "types": [p.source.encoding for p in (b.gate, b.up, b.down)],
                   "offset": b.down.column_offset, "columns": b.down.columns}
                  for layer, b in current_banks(role, rank)],
        "concats": [{"layer": layer, "kind": kind, "dtype": str(dtype),
                     "parts": [{"tensor": p.source.name, "shape": (p.n, p.k),
                                "encoding": p.source.encoding, "columns": p.columns}
                               for p in plan.parts]}
                    for layer, kind, plan, dtype in current_concats(role, rank)]}
        for rank in range(8)} for role in ("target", "mtp")}


def current_dense(plan):
    """Independent maintained CPU GGML decoder, bounded by each selected matrix."""
    values = []
    for start, count_ in plan.row_ranges:
        # Keep raw and decoded staging bounded even for a wide actual HC part.
        rows = max(1, (8 << 20) // max(plan.source.row_bytes, plan.source.shape[-1] * 4))
        for at in range(start, start + count_, rows):
            raw = read(plan.source, at, min(rows, start + count_ - at))
            decoded = decode_rows(plan.source, raw)
            values.append(np.concatenate([decoded[:, lo:lo + n] for lo, n in plan.columns], axis=1).astype(np.float16))
    return torch.from_numpy(np.ascontiguousarray(np.concatenate(values))).to("cuda")


class _DecodedRows:
    """Only adapt independently decoded source rows to the unchanged BMM oracle."""
    def __init__(self, values, source_ids, rows_per_expert, count_):
        self.values, self.rows_per_expert = values, rows_per_expert
        self.lookup = torch.full((count_,), -1, dtype=torch.int64, device="cuda")
        self.lookup[torch.tensor(source_ids, device="cuda")] = torch.arange(len(source_ids), device="cuda")

    def decode(self, ids):
        groups = self.lookup[ids // self.rows_per_expert]
        assert bool((groups >= 0).all()), "oracle attempted an unpopulated source expert"
        return self.values.index_select(0, groups * self.rows_per_expert + ids % self.rows_per_expert)


def current_bank(plan):
    """Four real source experts in full source-ID geometry; other slots are zero.

    This samples payloads, not all checkpoint values. Native high source IDs
    and the shared sentinel retain their actual positions and byte strides.
    """
    source_ids = tuple(sorted({0, 1, plan.count // 2, plan.count - 1}))
    packed, decoded, hashes = [], [], []
    for index, part in enumerate((plan.gate, plan.up, plan.down)):
        per_expert = plan.width if index < 2 else plan.dims
        raw_parts, real_parts = [], []
        for expert in source_ids:
            if index < 2:
                assert len(part.row_ranges) == plan.count
                start, count_ = part.row_ranges[expert]
                assert count_ == per_expert
            else:
                start = expert * per_expert
            raw = read(part.source, start, per_expert)
            real = decode_rows(part.source, raw)
            real_parts.append(np.concatenate([real[:, lo:lo + n] for lo, n in part.columns], axis=1).astype(np.float16))
            raw_parts.append(np.ascontiguousarray(np.concatenate(
                [raw[:, lo:lo + n] for lo, n in part.byte_segments], axis=1)))
        data = torch.zeros((part.n, part.row_bytes), dtype=torch.uint8, device="cuda")
        for expert, raw in zip(source_ids, raw_parts):
            data[expert * per_expert:(expert + 1) * per_expert].copy_(torch.from_numpy(raw))
        packed.append(GGUFPacked(data, part.source.ggml_type, part.n, part.k, part.column_offset))
        decoded.append(_DecodedRows(torch.from_numpy(np.concatenate(real_parts)).to("cuda"),
                                    source_ids, per_expert, plan.count))
        hashes.append(hashlib.sha256(b"".join(raw.tobytes() for raw in raw_parts)).hexdigest())
    shared = tuple(GGUFDense(current_dense(part)) for part in plan.shared)
    candidate = GGUFExperts(*packed, shared, count=plan.count, width=plan.width, dims=plan.dims)
    reference = GGUFExperts(*decoded, shared, count=plan.count, width=plan.width, dims=plan.dims)
    candidate.implementation = "packed_exact"
    assert candidate.pairs == reference.pairs
    return candidate, reference, source_ids, hashes


def current_inputs(rows, dims_, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = (torch.randn(rows, dims_, device="cuda", generator=generator) * (0.2 if seed % 2 else 3.0)).half()
    # R1 must still exercise nonzero arithmetic; larger windows additionally
    # carry a complete signed-zero row alongside ordinary positive/negative rows.
    selected = values[0] if rows > 1 else values[0, ::31]
    selected.copy_(torch.copysign(torch.zeros_like(selected), selected))
    return values


def _current_compare(plan, rows, dtype, seed):
    candidate, reference, source_ids, hashes = current_bank(plan)
    x = current_inputs(rows, plan.dims, seed)
    routes = torch.tensor((*source_ids, *reversed(source_ids), source_ids[0], source_ids[-1], plan.count),
                          device="cuda", dtype=torch.int32 if seed % 2 else torch.int64)
    picks = routes.repeat(rows, 1).roll(seed % routes.numel(), dims=1).contiguous()
    expected = reference.forward_reference(x, picks).to(dtype)
    out = torch.empty_like(expected)
    assert candidate.forward(x, picks, out) is out
    bits(out, expected)
    # Serial reference also detects changed batching/reduction storage bins.
    serial = torch.cat([reference.forward_reference(x[i:i + 1], picks[i:i + 1]) for i in range(rows)]).to(dtype)
    bits(out, serial)
    return {"types": [p.source.encoding for p in (plan.gate, plan.up, plan.down)],
            "offset": plan.down.column_offset, "source_experts": source_ids, "payload_sha256": hashes}


@pytest.mark.parametrize("role", ["target", "mtp"])
@pytest.mark.parametrize("rank", range(8))
def test_current_checkpoint_packed_types_and_all_tp8_offsets(role, rank, record_property):
    checked = []
    for layer, plan in current_banks(role, rank):
        checked.append({"layer": layer, "rank": rank,
                        **_current_compare(plan, 3, torch.float32, 981 + rank)})
    record_property("current_source_cases", json.dumps(checked, sort_keys=True))


@pytest.mark.parametrize("role", ["target", "mtp"])
@pytest.mark.parametrize("rows", CURRENT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_current_checkpoint_packed_row_and_dtype_boundaries(role, rows, dtype, record_property):
    # Maximal down block offset (including a crossing shard) is chosen from
    # actual plans, independently for each source format/layout class.
    selected = {}
    for rank in range(8):
        for layer, plan in current_banks(role, rank):
            key = tuple(p.source.ggml_type for p in (plan.gate, plan.up, plan.down))
            previous = selected.get(key)
            if previous is None or plan.down.column_offset > previous[2].down.column_offset:
                selected[key] = (rank, layer, plan)
    checked = [{"rank": rank, "layer": layer,
                **_current_compare(plan, rows, dtype, 1092 + rows)} for rank, layer, plan in selected.values()]
    record_property("current_source_cases", json.dumps(checked, sort_keys=True))
