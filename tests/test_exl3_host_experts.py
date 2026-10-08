"""Compact original EXL3 payload and logical-wave contracts, without a GPU."""

from dataclasses import replace
import json
import math
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tensorfold.cuda.exl3 import host_experts as host  # noqa: E402
from tensorfold.cuda.exl3 import format as fmt  # noqa: E402

pytestmark = pytest.mark.torch

PREFIX, SHARED = "model.layers.0.mlp.experts", "model.layers.0.mlp.shared_expert"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class Checkpoint:
    """Independent in-memory file-boundary fixture: exact header plus CPU payload."""

    def __init__(self, widths=((4, 4, 4), (6, 4, 2), (8, 8, 8)), *, cb="mul1", signs=False,
                 dims=256, width=128, seed=19):
        self.where, self.headers, self.values, self.reads, self.range_reads = {}, {}, {}, [], []
        self.tail = 0
        self.count, self.dims, self.width, self.cb = len(widths) - 1, dims, width, cb
        self.widths = widths
        self.names = [f"{PREFIX}.{i}" for i in range(self.count)] + [SHARED]
        self.gen = torch.Generator().manual_seed(seed)
        for name, ks in zip(self.names, widths):
            for j, projection in enumerate(PROJECTIONS):
                k, n = (width, dims) if j == 2 else (dims, width)
                base = name + "." + projection
                shape = (k // 16, n // 16, ks[j] * 8)
                value = torch.randint(-32768, 32768, shape, dtype=torch.int32, generator=self.gen).short()
                self.add(base + ".trellis", value)
                for part, size in (("su" if signs else "suh", k), ("sv" if signs else "svh", n)):
                    if signs:
                        value = torch.randint(-32768, 32768, (size // 16,), generator=self.gen).short()
                    else:
                        # Finite original FP16 bits, both signs and unequal scales.
                        value = ((torch.rand((size,), generator=self.gen) - .5) * .15).half()
                    self.add(base + "." + part, value)
                if cb in fmt.MARKERS:
                    magic = fmt.MARKERS[cb]
                    self.add(base + "." + cb, torch.tensor(magic - 2**32, dtype=torch.int32))

    def add(self, name, tensor):
        dtype = {torch.int16: "I16", torch.int32: "I32", torch.float16: "F16"}[tensor.dtype]
        self.where[name] = "fixture.safetensors"
        self.headers[name] = ["fixture.safetensors", self.tail, self.tail + tensor.nbytes, dtype, list(tensor.shape)]
        self.tail += tensor.nbytes
        self.values[name] = tensor

    def remove(self, name):
        self.where.pop(name)
        self.headers.pop(name)
        self.values.pop(name)

    def entry(self, name):
        return tuple(self.headers[name])

    def get(self, name):
        self.reads.append(name)
        return self.values[name]

    def read(self, file, begin, end):
        """Foreign range transport, using stored byte spans rather than decoded tags."""

        self.range_reads.append((file, begin, end))
        result = torch.zeros(end - begin, dtype=torch.uint8)
        for name, (source_file, lo, hi, _, _) in self.headers.items():
            if source_file != file:
                continue
            first, last = max(begin, lo), min(end, hi)
            if first < last:
                raw = self.values[name].contiguous().reshape(-1).view(torch.uint8)
                result[first - begin:last - begin].copy_(raw[first - lo:last - lo])
        return result

    def matrices(self, device):
        """Original resident native matrices, independent of the compact loader."""
        result = [[], [], []]
        for name in self.names:
            for j, projection in enumerate(PROJECTIONS):
                base = name + "." + projection
                values = []
                for part in ("trellis", "suh", "svh"):
                    if part != "trellis" and base + "." + part not in self.values:
                        packed = self.values[base + "." + ("su" if part == "suh" else "sv")].numpy()
                        sign = np.empty(packed.size * 16, dtype=np.float16)
                        for word, bits in enumerate(packed):
                            for bit in range(16):
                                sign[word * 16 + bit] = -1 if (int(bits) & 0xFFFF) >> bit & 1 else 1
                        value = torch.from_numpy(sign)
                    else:
                        value = self.values[base + "." + part]
                    values.append(value.to(device))
                result[j].append(tuple(values))
        return result


@pytest.mark.parametrize("cb", fmt.CODEBOOKS)
@pytest.mark.parametrize("signs", [False, True])
def test_compact_bytes_original_scales_and_logical_identity(cb, signs):
    pk = Checkpoint(cb=cb, signs=signs)
    authority, tables = host.load_compact(pk, PREFIX, pk.count, SHARED)
    expected_sizes = [pk.dims * pk.width // 16 * sum(ks) for ks in pk.widths]
    assert authority.payload_bytes == sum(expected_sizes)
    assert authority.payload_bytes < len(pk.names) * max(expected_sizes)
    assert authority.metadata_bytes == 28 * len(pk.names)
    assert authority.trellis_bytes.tolist() == expected_sizes
    assert authority.starts.tolist() == [sum(expected_sizes[:i]) for i in range(len(pk.names))]
    assert authority.k2.tolist() == [list(ks) for ks in pk.widths]
    assert authority.codebook == cb and tables.cb == fmt.CODEBOOKS.index(cb)
    assert tables.count == len(pk.names) and tables.dims == pk.dims and tables.width == pk.width
    assert tables.k2_gu == (4, 8) and tables.k2_d == (2, 8)
    assert tables.keep == [] and tables.nbytes_read([0, 2]) == expected_sizes[0] + expected_sizes[2]
    for ptr in (tables.gate_ptr, tables.up_ptr, tables.down_ptr):
        assert ptr.tolist() == [0] * len(pk.names)
    mats = pk.matrices("cpu")
    scales = ((tables.suh_g, tables.svh_g), (tables.suh_u, tables.svh_u), (tables.suh_d, tables.svh_d))
    for j, projection in enumerate(PROJECTIONS):
        for e, name in enumerate(pk.names):
            actual = authority.projection(e, j)
            original = pk.values[name + "." + projection + ".trellis"]
            assert actual.data_ptr() >= authority.data.data_ptr()
            assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
            for s, target in enumerate(scales[j]):
                assert torch.equal(target[e].view(torch.int16), mats[j][e][s + 1].view(torch.int16))
    assert all(not torch.is_inference(t) for t in authority.source)
    assert all(t.device.type == "cpu" for t in authority.source)
    assert not authority.data.is_pinned()
    # All file marker reads precede any trellis or scale payload read.
    markers = sum(name.endswith((".mul1", ".mcg")) for name in pk.values)
    assert all(name.endswith((".mul1", ".mcg")) for name in pk.reads[:markers])


def test_inference_loading_still_constructs_versioned_owned_authority_and_tables():
    pk = Checkpoint(widths=((3, 5, 7), (8, 8, 8)))
    with torch.inference_mode():
        authority, tables = host.load_compact(pk, PREFIX, pk.count, SHARED)
    for t in (*authority.source, tables.gate_k2, tables.suh_g, tables.gate_ptr):
        assert not torch.is_inference(t)
        assert isinstance(t._version, int)
    expected = pk.values[PREFIX + ".0.gate_proj.trellis"].clone()
    pk.values[PREFIX + ".0.gate_proj.trellis"].zero_()
    assert torch.equal(authority.projection(0, 0), expected)


@pytest.mark.parametrize("kind", ["value", "shape", "dtype", "payload"])
def test_marker_payload_is_verified_before_loading_weight_payloads(kind):
    pk = Checkpoint()
    marker = PREFIX + ".0.gate_proj.mul1"
    if kind == "value":
        pk.values[marker].fill_(7)
    elif kind == "shape":
        pk.add(marker, torch.tensor([fmt.MARKERS["mul1"] - 2**32] * 2, dtype=torch.int32))
    elif kind == "dtype":
        pk.add(marker, torch.tensor(7, dtype=torch.int16))
    else:
        pk.values[marker] = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="marker payload"):
        host.load_compact(pk, PREFIX, pk.count, SHARED)
    assert all(name.endswith(".mul1") for name in pk.reads)


@pytest.mark.parametrize("kind,match", [
    ("missing", "missing or unexpected"), ("extra", "missing or unexpected"),
    ("unknown", "unexpected EXL3 projection parts"), ("bias", "biased EXL3"),
    ("aliases", "ambiguous duplicate"), ("mixed", "mix codebooks"),
    ("dims", "dimensions disagree"), ("range", "payload length"),
    ("shape_type", "invalid EXL3 dtype or dimensions"),
])
def test_headers_reject_incompatible_bundle_without_trellis_reads(kind, match):
    pk = Checkpoint()
    base = PREFIX + ".0.gate_proj"
    if kind == "missing":
        pk.remove(base + ".trellis")
        # Entire missing projection is caught before header parsing.
        for key in list(pk.where):
            if key.startswith(base + "."):
                pk.remove(key)
    elif kind == "extra":
        pk.add(PREFIX + ".3.gate_proj.trellis", pk.values[base + ".trellis"])
    elif kind == "unknown":
        pk.add(base + ".unknown", torch.tensor(0, dtype=torch.int16))
    elif kind == "bias":
        pk.add(base + ".bias", torch.zeros(pk.width, dtype=torch.float16))
    elif kind == "aliases":
        pk.add(base + ".su", torch.zeros(pk.dims // 16, dtype=torch.int16))
    elif kind == "mixed":
        pk.remove(base + ".mul1")
    elif kind == "dims":
        base = PREFIX + ".1.gate_proj"
        value = pk.values[base + ".trellis"]
        pk.add(base + ".trellis", value[:8].contiguous())
        pk.add(base + ".suh", torch.zeros(128, dtype=torch.float16))
    elif kind == "range":
        pk.headers[base + ".trellis"][2] -= 2
    else:
        pk.headers[base + ".trellis"][4][0] = True
    with pytest.raises(ValueError, match=match):
        host.load_compact(pk, PREFIX, pk.count, SHARED)
    assert not any(name.endswith((".trellis", ".suh", ".svh")) for name in pk.reads)


@pytest.mark.parametrize("kind", ["dtype", "short", "contiguity", "shape"])
def test_foreign_range_transport_must_match_declared_raw_byte_contract(kind):
    pk = Checkpoint()
    read = pk.read

    def malformed(file, begin, end):
        value = read(file, begin, end)
        if kind == "dtype":
            return value.float()
        if kind == "short":
            return value[:-1]
        if kind == "contiguity":
            return value.repeat_interleave(2)[::2]
        return value.view(1, -1)

    pk.read = malformed
    with pytest.raises(ValueError, match="exact contiguous CPU bytes"):
        host.load_compact(pk, PREFIX, pk.count, SHARED)


@pytest.mark.parametrize("count", [0, -1, 1024, True, 2.0])
def test_invalid_count_is_rejected_without_payload(count):
    pk = Checkpoint()
    with pytest.raises(ValueError):
        host.load_compact(pk, PREFIX, count, SHARED)
    assert pk.reads == []


@pytest.mark.parametrize("field,value", [
    ("dims", 256.0), ("width", True), ("width", 129), ("codebook", "unknown"),
    ("starts", torch.tensor([0, 1, 2], dtype=torch.int64)),
    ("starts", torch.tensor([0, 1, 2], dtype=torch.int32)),
    ("k2", torch.full((3, 3), 17, dtype=torch.int32)),
    ("k2", torch.full((3, 3), 9, dtype=torch.int32)),
    ("trellis_bytes", torch.tensor([1, 2, 3], dtype=torch.int64)),
])
def test_authority_ranges_and_exact_geometry_are_checked(field, value):
    pk = Checkpoint()
    authority, _ = host.load_compact(pk, PREFIX, pk.count, SHARED)
    with pytest.raises(ValueError):
        host._validate_authority(replace(authority, **{field: value}))


def test_authority_must_own_versioned_exact_pageable_payload():
    pk = Checkpoint()
    authority, _ = host.load_compact(pk, PREFIX, pk.count, SHARED)
    for data in (authority.data[:-1], authority.data[::2], authority.data.short()):
        with pytest.raises(ValueError):
            host._validate_authority(replace(authority, data=data))
    with torch.inference_mode():
        inference = authority.data.clone()
    with pytest.raises(ValueError, match="ordinary immutable"):
        host._validate_authority(replace(authority, data=inference))
    for expert, projection in ((-1, 0), (3, 0), (False, 0), (0, -1), (0, True)):
        with pytest.raises(ValueError):
            authority.projection(expert, projection)


@pytest.mark.parametrize("capacity", [1, 2, 4, 16])
def test_logical_waves_preserve_each_original_row_slot_exactly_once(capacity):
    pick = np.array([[3, 0, 3, -1], [1, 2, 4, 1], [4, 0, 999, 2]], dtype=np.int32)
    before = pick.copy()
    waves = host.logical_waves(pick, 5, capacity)
    assert waves == [list(range(5))[i:i + capacity] for i in range(0, 5, capacity)]
    coverage = np.zeros_like(pick)
    for active in waves:
        assert len(active) <= capacity
        coverage += np.isin(pick, active).astype(np.int32)
    assert np.array_equal(coverage, ((pick >= 0) & (pick < 5)).astype(np.int32))
    assert np.array_equal(pick, before)
    assert host.logical_waves(np.full((2, 3), -1, dtype=np.int32), 5, capacity) == []


@pytest.mark.parametrize("pick,count,capacity", [
    (np.zeros((2, 2), dtype=np.int64), 5, 1), (np.zeros(4, dtype=np.int32), 5, 1),
    ([[0]], 5, 1), (np.zeros((2, 2), dtype=np.int32), True, 1),
    (np.zeros((2, 2), dtype=np.int32), 5, 0),
])
def test_wave_boundary_rejects_ambiguous_or_unbounded_input(pick, count, capacity):
    with pytest.raises(ValueError):
        host.logical_waves(pick, count, capacity)


def test_original_payload_formula_uses_half_bits_without_uniform_padding():
    widths = ((3, 5, 7), (16, 2, 4), (8, 8, 8))
    pk = Checkpoint(widths=widths)
    authority, _ = host.load_compact(pk, PREFIX, pk.count, SHARED)
    expected = sum(math.prod(pk.values[name + "." + p + ".trellis"].shape) * 2
                   for name in pk.names for p in PROJECTIONS)
    assert authority.payload_bytes == expected
    assert authority.max_entry_bytes == max(sum(ks) * pk.dims * pk.width // 16 for ks in widths)


@pytest.mark.parametrize("rows,slots,device", [(0, 9, "cuda"), (1, 33, "cuda"), (1, 9, "cpu"),
                                                (True, 9, "cuda"), (1, 9.0, "cuda")])
def test_wave_controls_fail_before_any_allocation_for_unsupported_geometry(rows, slots, device):
    with pytest.raises(ValueError):
        host.WaveScratch(rows, slots, device)


def test_indexed_header_prefix_includes_unicode_suffix_and_rejects_different_reader():
    # Pure startup index; constructing no CUDA buffers isolates its string-boundary contract.
    cache = host.Exl3HostExpertCache.__new__(host.Exl3HostExpertCache)
    cache._sealed, cache._header_keys, cache._pack_ref = False, None, None
    pk = Checkpoint()
    pk.add(PREFIX + ".\U0001f642.gate_proj.trellis", torch.ones((8, 8, 32), dtype=torch.int16))
    keys = cache.projection_keys(pk, PREFIX, SHARED)
    assert PREFIX + ".\U0001f642.gate_proj" in keys
    with pytest.raises(ValueError, match="missing or unexpected"):
        host.load_compact(pk, PREFIX, pk.count, SHARED, keys=keys)
    with pytest.raises(ValueError, match="one checkpoint reader"):
        cache.projection_keys(Checkpoint(), PREFIX, SHARED)


@pytest.mark.parametrize("pick", [
    [[0, 0]], [[2, 1, 2]], [[0, -1, 0, 999]], [[-1, -1], [3, 3]],
])
def test_native_snapshot_rejects_only_duplicate_valid_ids_before_member_writes(pick):
    original = np.array(pick, dtype=np.int32)
    before = original.copy()
    with pytest.raises(ValueError, match="distinct valid experts"):
        host._validate_native_picks(original, 4)
    assert np.array_equal(original, before)


@pytest.mark.parametrize("pick", [
    [[0, 1, 2, 3]], [[-1, -1, -5, -5]], [[4, 4, 999, 999]], [[0, 4, -1, 1], [1, 4, -1, 0]],
    [[]],
])
def test_native_snapshot_permits_invalid_sentinel_repetition_and_cross_row_reuse(pick):
    original = np.array(pick, dtype=np.int32)
    before = original.copy()
    host._validate_native_picks(original, 4)
    assert np.array_equal(original, before)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_scale_fails_with_part_context_before_gpu_transfer(bad):
    pk = Checkpoint()
    name = PREFIX + ".0.gate_proj.suh"
    pk.values[name][0] = bad
    # A CPU-only image has no CUDA driver. The explicit CUDA destination proves
    # validation fails before reaching any .to(device) transfer.
    with pytest.raises(ValueError, match=r"experts\.0\.gate_proj\.suh.*finite"):
        host.load_compact(pk, PREFIX, pk.count, SHARED, device="cuda")


def test_borrowed_joint_fields_preserve_bits_with_unaligned_scalar_marker_storage():
    pk = Checkpoint()
    read = pk.read

    def offset_two(file, begin, end):
        original = read(file, begin, end)
        backing = torch.empty(original.numel() + 2, dtype=torch.uint8)
        view = backing[2:]
        view.copy_(original)
        return view

    pk.read = offset_two
    authority, _ = host.load_compact(pk, PREFIX, pk.count, SHARED)
    for e, name in enumerate(pk.names):
        for j, projection in enumerate(PROJECTIONS):
            assert torch.equal(authority.projection(e, j), pk.values[name + "." + projection + ".trellis"])


def test_joint_pipeline_has_bounded_windows_and_no_field_clones(monkeypatch):
    pk = Checkpoint(widths=((4, 4, 4),) * 16 + ((8, 8, 8),), dims=2560, width=640)
    returned = set()
    read, validate = pk.read, host.native.validate_scale_payload

    def tracked(file, begin, end):
        assert len(pk.reads) == len(pk.names) * 3  # every original marker was validated before any range read
        value = read(file, begin, end)
        returned.add(value.untyped_storage().data_ptr())
        return value

    def original_view(value, context):
        assert value.untyped_storage().data_ptr() in returned
        return validate(value, context)

    pk.read = tracked
    monkeypatch.setattr(host.native, "validate_scale_payload", original_view)
    authority, _ = host.load_compact(pk, PREFIX, pk.count, SHARED)
    assert authority.count == 17 and len(pk.range_reads) >= 2
    assert max(end - begin for _, begin, end in pk.range_reads) <= 16 << 20
    assert sum(end - begin for _, begin, end in pk.range_reads) == sum(t.nbytes for t in pk.values.values())


def test_marker_failure_drains_other_failed_future_preserving_primary_and_reader_lifetime(monkeypatch):
    pk = Checkpoint()
    for expert, name in enumerate(pk.names):
        for key in pk.where:
            if key.startswith(name + "."):
                pk.where[key] = pk.headers[key][0] = f"frame{expert}.safetensors"
    failed, active = threading.Event(), []
    read = pk.read

    def fault(file, begin, end):
        active.append(file)
        try:
            if file == "frame1.safetensors":
                failed.set()
                raise IOError("secondary queued reader failure")
            value = read(file, begin, end)
            if file == "frame0.safetensors":
                key = PREFIX + ".0.gate_proj.mul1"
                offset = pk.headers[key][1] - begin
                value[offset] ^= 1
            return value
        finally:
            active.remove(file)

    unpack = host.struct.unpack

    def wait_then_unpack(code, value):
        assert failed.wait(5), "second queued read must complete before primary marker failure"
        return unpack(code, value)

    pk.read = fault
    monkeypatch.setattr(host.struct, "unpack", wait_then_unpack)
    with pytest.raises(ValueError, match="codebook marker payload") as caught:
        host.load_compact(pk, PREFIX, pk.count, SHARED)
    assert any("secondary queued reader failure" in note for note in caught.value.__notes__)
    assert active == [] and failed.is_set()


@pytest.mark.parametrize("kind", ["dtype", "shape", "contiguity", "scale"])
def test_oversize_streaming_decoded_payload_must_match_validated_header(kind):
    # The original decoded-reader boundary remains reachable in the large,
    # sequential projection region; retain its failure classes unchanged.
    pk = Checkpoint(widths=((12, 12, 12),) * 2, dims=4096, width=2048)
    key = PREFIX + ".0.gate_proj.trellis"
    if kind == "dtype":
        pk.values[key] = pk.values[key].float()
    elif kind == "shape":
        pk.values[key] = pk.values[key][:-1]
    elif kind == "contiguity":
        pk.values[key] = pk.values[key].transpose(0, 1)
    else:
        pk.values[PREFIX + ".0.gate_proj.suh"] = torch.zeros(pk.dims, dtype=torch.float32)
    with pytest.raises(ValueError, match="payload disagrees|invalid FP16 scale"):
        host.load_compact(pk, PREFIX, pk.count, SHARED)
    assert pk.range_reads == []


def test_real_oversize_checkpoint_preserves_streaming_bytes_and_loading_admission(tmp_path):
    from safetensors.torch import save_file
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack
    from tensorfold.families.qwen4_exp import ram_experts

    pk = Checkpoint(widths=((12, 12, 12),) * 2, dims=4096, width=2048)
    base = "model.language_model.layers.0.mlp"
    prefix, shared = base + ".experts", base + ".shared_expert"
    tensors = {name.replace(PREFIX, prefix).replace(SHARED, shared): value for name, value in pk.values.items()}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        name: "model.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {
        "hidden_size": 4096, "moe_intermediate_size": 2048, "shared_expert_intermediate_size": 2048,
        "num_experts": 1, "num_experts_per_tok": 1, "num_hidden_layers": 1, "max_position_embeddings": 128},
        "quantization_config": {"quant_method": "exl3"}}))
    layout = ram_experts.layout(tmp_path, .1, mtp=False)
    reader = Pack(tmp_path)
    get, returned = reader.get, []

    def tracked(name):
        value = get(name)
        returned.append((name, value.untyped_storage().nbytes()))
        return value

    reader.get = tracked
    try:
        authority, tables = host.load_compact(reader, prefix, 1, shared)
        assert authority.max_entry_bytes > 16 << 20
        assert authority.payload_bytes == layout.host_bytes and authority.max_entry_bytes == layout.entry_bytes
        assert authority.metadata_bytes == layout.metadata_host_bytes
        for e, name in enumerate(pk.names):
            for j, projection in enumerate(PROJECTIONS):
                assert torch.equal(authority.projection(e, j), pk.values[name + "." + projection + ".trellis"])
        max_projection = max(size for name, size in returned if name.endswith(".trellis"))
        max_scale = max(size for name, size in returned if name.endswith((".suh", ".svh")))
        # One original projection plus scales, including its possible Reader
        # alignment relocation, is inside the unchanged2*largest admission.
        assert 2 * max_projection + 2 * max_scale < layout.loading_bytes
        assert layout.loading_bytes >= 2 * layout.entry_bytes
        assert tables.k2_gu == tables.k2_d == (12, 12)
    finally:
        reader.io.close()
