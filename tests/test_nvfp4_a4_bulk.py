"""File-owned native bank loading: exact bytes, ordering, bounds and lifetime."""

import __future__

import ast
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from nvfp4_a4_inputs import PREFIX, Context, swiglu
from test_nvfp4_a4_cache_control import CPUCache
from test_nvfp4_a4_host_control import oracle_blob

from tensorfold.cuda.direct_read import Reader, SafeTensors
from tensorfold.cuda.nvfp4 import experts_a4, host_layout
from tensorfold.cuda.nvfp4 import format as fmt
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4

ROOT = Path(__file__).parents[1]
BACKBONE = ROOT / "src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py"


class FileContext(Context):
    """Actual owned-reader/context closure; no fake CUDA payload is constructed."""

    def materialize(self, tmp_path, *, prefix=0, direct=False):
        self.files = SafeTensors.__new__(SafeTensors)
        self.files.reader = Reader()
        self.files.reader.direct = direct and self.files.reader.direct
        self.files.where = {}
        path = tmp_path / "bank.bin"
        blob = bytearray(prefix)
        for name in sorted(self.payload, key=lambda name: (self.info[name][0], name)):
            value = self.payload[name]
            raw = value.view(torch.uint8).numpy().tobytes()
            self.files.where[name] = (path, len(blob), len(raw), *self.info[name])
            blob.extend(raw)
        path.write_bytes(blob)
        self.where = dict.fromkeys(self.files.where)
        self.run_reads, self.run_refs = [], []
        node = next(node for node in ast.walk(ast.parse(BACKBONE.read_bytes()))
                    if isinstance(node, ast.FunctionDef) and node.name == "prepare_cpu_run")
        scope = {"t": self}
        exec(compile(ast.Module([node], []), str(BACKBONE), "exec",  # noqa: S102 - actual owned loading closure.
                     flags=__future__.annotations.compiler_flag), scope)
        actual = scope["prepare_cpu_run"]

        def prepare(names, **geometry):
            names = tuple(names)
            run = actual(names, **geometry)
            if run is None:
                return None

            def read(first, stop):
                value = run(first, stop)
                self.run_reads.append(names[first:stop])
                root = value
                while isinstance(root, torch.Tensor):
                    self.run_refs.append(weakref.ref(root))
                    root = root._base
                self.reads.extend(names[first:stop])
                return value

            return read

        self.prepare_cpu_run = prepare
        return self

    def read(self, name, device="cpu"):
        del self.where[name]
        self.reads.append(name)
        return self.files.get(name, device)

    def read_f32_scalars(self, names, *, max_bytes):
        names = tuple(names)
        result = self.files.get_f32_scalars(names, max_bytes=max_bytes)
        for name in names:
            del self.where[name]
        self.reads.extend(names)
        self.batches.append((names, max_bytes))
        return result


def expected_authority(context, fields, ids):
    for local, expert in enumerate(ids):
        for p, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            owner = f"{PREFIX}.shared_expert" if expert == "shared" else f"{PREFIX}.experts.{expert}"
            base = f"{owner}.{projection}"
            want = oracle_blob(context.payload[base + ".weight_packed"], context.payload[base + ".weight_scale"])
            assert torch.equal(fields[0][local, p] if p < 2 else fields[1][local], want)
            weight = float(context.payload[base + ".weight_global_scale"][0])
            original = context.payload[base + ".input_global_scale"][0]
            slot = p if p < 2 else 0
            actual_scale = fields[2 if p < 2 else 3][local, slot]
            actual_input = fields[4 if p < 2 else 5][local, slot]
            assert torch.equal(actual_scale.view(torch.int32), torch.tensor(1.0 / weight).view(torch.int32))
            assert torch.equal(actual_input.view(torch.int32), original.view(torch.int32))


@pytest.mark.parametrize("width,dims", [(64, 64), (64, 128), (192, 128), (128, 192)])
@pytest.mark.parametrize("direct", [False, True])
def test_owned_bank_has_independent_oracle_bytes_original_globals_and_no_retained_read_owner(tmp_path, width, dims, direct):
    context = FileContext(count=2, width=width, dims=dims).materialize(tmp_path, direct=direct)
    cache = CPUCache(width, dims)
    try:
        bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache,
                                    owned_ids=[1, 0], swiglu=swiglu)
        fields = cache.source_views(cache._layers[0].source[0])
        expected_authority(context, fields, [1, 0])
        assert bank.count == 2 and len(context.run_reads) == 2
        assert all(name.endswith("weight_scale") for name in context.run_reads[0])
        assert all(name.endswith("weight_packed") for name in context.run_reads[1])
        gc.collect()
        assert all(ref() is None for ref in context.run_refs)
        assert not context.where
        assert not any(isinstance(value, torch.Tensor) for value in vars(bank).values())
    finally:
        context.files.close()


def test_admitted_window_forces_partial_chunks_without_reading_or_writing_other_cells(tmp_path, monkeypatch):
    count, width, dims = 5, 64, 128
    scales = count * 3 * width * dims // 16
    monkeypatch.setattr(host_layout, "READER_PIECE_BYTES", scales + 6 * 4096 + 2 * 3 * width * dims // 2)
    context = FileContext(count=count, width=width, dims=dims).materialize(tmp_path)
    cache = CPUCache(width, dims)
    try:
        load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert list(map(len, context.run_reads)) == [15, 6, 6, 3]
        expected_authority(context, cache.source_views(cache._layers[0].source[0]), range(count))
    finally:
        context.files.close()


@pytest.mark.parametrize("width,dims", [(64, 128), (192, 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_batched_word_and_sf_planes_have_independent_nonperiodic_all_byte_oracle(width, dims, strided):
    generator = torch.Generator().manual_seed(3903 + width + dims)
    count, selected = 3, [2, 0]
    cache = CPUCache(width, dims)
    if strided:
        source = torch.full((count, cache.entry_bytes), 0xa5, dtype=torch.uint8)
        expected = source.clone()
        fields, reference = cache.source_views(source), cache.source_views(expected)
    else:
        shapes = host_layout.authority_fields_a4(count, width, dims)
        fields = tuple(torch.full(shape, 0xa5, dtype=torch.uint8) for shape, _dtype in shapes[:2])
        reference = tuple(t.clone() for t in fields)
    raw_codes, raw_scales = [], []
    for local in selected:
        for p, (n, k) in enumerate(((dims, width), (width, dims), (width, dims))):
            codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
            scales = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=generator)
            scales.view(-1)[:256] = torch.randperm(256, generator=generator).to(torch.uint8)
            assert torch.unique(codes, dim=0).shape[0] == n
            raw_codes.append(codes.reshape(-1))
            raw_scales.append(scales.reshape(-1))
            blob = oracle_blob(codes, scales)
            (reference[1][local] if p == 0 else reference[0][local, p - 1]).copy_(blob)
    codes, scales = torch.cat(raw_codes), torch.cat(raw_scales)
    before_codes, before_scales = codes.clone(), scales.clone()
    experts_a4.pack_bank_a4(codes, scales, fields[0], fields[1], torch.tensor(selected), width, dims)
    assert torch.equal(codes, before_codes) and torch.equal(scales, before_scales)
    if strided:
        assert torch.equal(source, expected)  # calibration cells/padding/unselected rows are untouched
    else:
        assert all(torch.equal(a, b) for a, b in zip(fields, reference, strict=True))


def test_shared_cell_and_resident_final_field_control_use_the_same_owned_chunks(tmp_path, monkeypatch):
    """CPU transfer/control emulation; actual CUDA consumers require native gates."""
    context = FileContext(count=2).alias_shared_expert(0).materialize(tmp_path)
    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None and torch.device(device).type == "cuda":
            return value.clone()
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(experts_a4, "ExpertsA4", lambda fields, _width, _dims, **_kwargs:
                        SimpleNamespace(fields=fields, count=fields[0].shape[0]))
    try:
        bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", shared=True, swiglu=swiglu)
        assert bank.count == 3 and len(context.run_reads) == 2
        expected_authority(context, bank.fields, [0, 1, "shared"])
        assert not context.where and all(ref() is None for ref in context.run_refs)
    finally:
        context.files.close()


@pytest.mark.parametrize("fault", ["unaligned", "unowned_gap"])
def test_unsupported_owned_run_uses_canonical_payload_order_without_reading_unowned_fields(tmp_path, fault):
    context = FileContext(count=3).materialize(tmp_path, prefix=int(fault == "unaligned"))
    ids = [2, 0] if fault == "unowned_gap" else [0, 1, 2]
    cache = CPUCache()
    try:
        load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, owned_ids=ids, swiglu=swiglu)
        assert not context.run_reads
        expected_authority(context, cache.source_views(cache._layers[0].source[0]), ids)
        if fault == "unowned_gap":
            assert not any(".experts.1." in name for name in context.reads)
    finally:
        context.files.close()


@pytest.mark.parametrize("case", ["first_global", "earlier_scale_later_global", "numeric_not_lexical_scale"])
def test_eligible_preflight_preserves_exact_canonical_first_invalid_error_and_no_invalid_global_code_read(tmp_path, case):
    canonical = Context(count=12 if case == "numeric_not_lexical_scale" else 3)
    if case == "first_global":
        canonical.payload[PREFIX + ".experts.0.gate_proj.input_global_scale"][0] = 0
    else:
        first = 2 if case == "numeric_not_lexical_scale" else 0
        canonical.payload[f"{PREFIX}.experts.{first}.gate_proj.weight_scale"].view(torch.uint8)[0, 0] = 0xff
        if case == "earlier_scale_later_global":
            canonical.payload[PREFIX + ".experts.1.up_proj.weight_global_scale"][0] = float("inf")
        else:
            canonical.payload[PREFIX + ".experts.10.down_proj.weight_scale"].view(torch.uint8)[0, 0] = 0x80
    context = FileContext(count=canonical.cfg.experts)
    context.payload = {name: value.clone() for name, value in canonical.payload.items()}
    context.materialize(tmp_path)
    try:
        errors = []
        for value in (canonical, context):
            cache = CPUCache()
            with pytest.raises(ValueError) as caught:
                load_host_experts_a4(value, PREFIX, value.cfg, "cuda:0", cache=cache, swiglu=swiglu)
            assert type(caught.value) is ValueError
            errors.append(str(caught.value))
            assert not cache._layers and not value.used
        assert errors[0] == errors[1]
        if case == "first_global":
            assert not context.run_reads and not any(name.endswith(("weight_packed", "weight_scale")) for name in context.reads)
        elif case == "earlier_scale_later_global":
            assert not context.run_reads
        else:
            assert len(context.run_reads) == 1
            assert not any(".experts.10." in name and name.endswith("weight_packed") for name in context.reads)
    finally:
        context.files.close()


@pytest.mark.parametrize("failure", [MemoryError("native allocation"), ValueError("unexpected helper defect"),
                                     KeyboardInterrupt("cancelled")])
def test_unexpected_preflight_helper_failure_is_never_canonical_eligibility(tmp_path, monkeypatch, failure):
    context = FileContext(count=2).materialize(tmp_path)
    cache = CPUCache()

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(fmt, "_native_factors_result", fail)
    try:
        with pytest.raises(type(failure)) as caught:
            load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert caught.value is failure and not context.run_reads and not cache._layers and not context.used
    finally:
        context.files.close()


def test_partial_indexed_pack_failure_never_publishes_or_retains_reader_backing(tmp_path, monkeypatch):
    context = FileContext(count=2).materialize(tmp_path)
    cache = CPUCache()
    failure = OSError("private destination copy failed")

    def fail(codes, _scales, up, _down, *_args):
        up[0, 0, 0] = codes[0]
        raise failure

    monkeypatch.setattr(experts_a4, "pack_bank_a4", fail)
    try:
        with pytest.raises(OSError) as caught:
            load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert caught.value is failure and not cache._layers and not context.used
        # The retained primary traceback legitimately owns its frame/locals.
        del caught
        failure.__traceback__ = None
        gc.collect()
        assert all(ref() is None for ref in context.run_refs)
    finally:
        context.files.close()
