"""Real owned CPU scalar coalescing through the same Reader/future lifetime."""

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from tensorfold.cuda.nvfp4.host_experts import _read
from tensorfold.cuda.nvfp4.modelopt_layout import a4_read_plan
from tensorfold.families.qwen4_exp.cuda.reader import _Reader


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_native_scalar_queue_returns_original_cpu_bits_without_cuda_or_fallback(tmp_path, monkeypatch, dtype, shape):
    prefix = "model.language_model.layers.0.mlp"
    values = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = prefix + ".experts.0." + projection
        values[base + ".weight"] = torch.arange(64 * 64, dtype=torch.int64).to(torch.uint8).reshape(64, 64)
        values[base + ".weight_scale"] = torch.full((64, 8), 0x23, dtype=torch.uint8).view(torch.float8_e4m3fn)
        values[base + ".input_scale"] = torch.full(shape, .001964750699698925, dtype=dtype)
        values[base + ".weight_scale_2"] = torch.full(shape, .00020781018247362226, dtype=dtype)
    file = tmp_path / "bank.safetensors"
    save_file(values, str(file))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(values, file.name)}))
    reader = _Reader(tmp_path, "cuda:0")

    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU scalar queue touched CUDA or attempted an unqueued device read")

    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr(torch.cuda, "Stream", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(reader.io, "_to_device", forbidden)
    try:
        device, host = a4_read_plan(list(values), {prefix}, 1)
        assert not device and len(host) == 6
        reader.queue_host_scalars(host)
        # Duplicate lookahead submission must preserve the original future.
        accepted = dict(reader.reads.ahead)
        reader.queue_host_scalars(host)
        assert reader.reads.ahead == accepted
        for name in host:
            got = _read(reader, name)
            assert got.device.type == "cpu" and got.shape == values[name].shape
            assert got.dtype == dtype
            assert torch.equal(got.reshape(-1).view(torch.uint8), values[name].reshape(-1).view(torch.uint8))
        assert not reader.reads.ahead and reader.reads.stream is None
    finally:
        reader.close()
    assert reader.reads.pool is None and not reader.io.staging


def test_queue_default_device_and_host_refusal_preserve_the_existing_reader_contract():
    reader = _Reader.__new__(_Reader)
    recorded = []
    reader.device = "cuda:0"
    reader.where, reader.touched = {"value": "bank"}, set()
    reader.info = lambda _name: {"data_offsets": [8, 12], "path": "bank", "byte_offset": 108,
                               "dtype": "F32", "shape": ()}
    reader.reads = type("Queue", (), {"queue": lambda _self, items, device: recorded.append((items, device))})()
    reader.queue(["value"])
    assert recorded[-1] == ([("value", "bank", 108, 112, None)], "cuda:0")
    reader.queue_host_scalars(["value"])
    assert recorded[-1] == ([("value", "bank", 108, 112, None)], None)
    before = len(recorded)
    reader.info = lambda _name: {"dtype": "F32", "shape": (2,)}
    with pytest.raises(ValueError, match="floating singleton"):
        reader.queue_host_scalars(["value"])
    assert len(recorded) == before


def test_cpu_scalar_queue_never_coalesces_unrequested_matrix_gaps():
    reader = _Reader.__new__(_Reader)
    recorded = []
    reader.device = "cuda:0"
    reader.where, reader.touched = dict.fromkeys(("a", "b", "c"), "bank"), set()
    offsets = {"a": 100, "b": 104, "c": 4096}
    reader.info = lambda name: {"data_offsets": [0, 4], "path": "bank", "byte_offset": offsets[name],
                               "dtype": "F32", "shape": ()}
    reader.reads = type("Queue", (), {"queue": lambda _self, items, device: recorded.append((items[:], device))})()
    reader.queue_host_scalars(["c", "b", "a"])
    assert recorded == [([("a", "bank", 100, 104, None), ("b", "bank", 104, 108, None)], None),
                        ([("c", "bank", 4096, 4100, None)], None)]
    assert sum(items[-1][3] - items[0][2] for items, _ in recorded) == 12
    reader.info = lambda _name: {"data_offsets": [0, 4], "path": "bank", "byte_offset": 0,
                                "dtype": "F32", "shape": (2,)}
    with pytest.raises(ValueError, match="floating singleton"):
        reader.queue_host_scalars(["a"])
    assert len(recorded) == 2
