"""CPU controls for GPU packing dispatch, scalar policy and failed publication."""

import struct

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from nvfp4_a4_inputs import PREFIX, Context, swiglu
from test_nvfp4_a4_host_control import Cache

from tensorfold.cuda.nvfp4.experts_a4 import pack_projection_a4
from tensorfold.cuda.nvfp4.host_experts import _original_projection, load_host_experts_a4


def test_modelopt_original_payload_requests_pack_device_but_global_conversion_stays_cpu():
    fields = {"p.weight": torch.arange(2048, dtype=torch.int64).to(torch.uint8).view(64, 32),
              "p.weight_scale": torch.arange(256, dtype=torch.int64).to(torch.uint8).view(64, 4)
                                        .view(torch.float8_e4m3fn),
              "p.weight_scale_2": torch.tensor([.03125], dtype=torch.bfloat16)}
    calls = []

    class Reader:
        def info(self, name):
            return {"shape": fields[name].shape}

        def get_rows(self, name, lo, hi, device="cpu"):
            # Dispatch control only: intentionally retain CPU payload here.
            calls.append((name, lo, hi, str(device)))
            return fields[name][lo:hi]

    codes, scales, global_scale = _original_projection(Reader(), "p", torch.device("cuda:0"))
    assert calls == [("p.weight", 0, 64, "cuda:0"), ("p.weight_scale", 0, 64, "cuda:0"),
                     ("p.weight_scale_2", 0, 1, "cpu")]
    assert torch.equal(codes[0], fields["p.weight"])
    assert torch.equal(scales[0].view(torch.uint8), fields["p.weight_scale"].view(torch.uint8))
    assert global_scale.device.type == "cpu" and global_scale.dtype == torch.float32
    assert global_scale.view(torch.int32)[0] == torch.tensor(.03125).view(torch.int32)


def test_ct_cpu_packing_failure_preserves_primary_and_never_publishes_partial_bank(monkeypatch):
    from tensorfold.cuda.nvfp4 import experts_a4

    ctx, cache = Context(), Cache()
    primary = KeyboardInterrupt("packing interrupted")
    calls = []

    def fail(codes, scales):
        calls.append((codes.device.type, scales.device.type))
        raise primary

    monkeypatch.setattr(experts_a4, "pack_projection_a4", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert caught.value is primary and calls == [("cpu", "cpu")]
    assert ctx.reads == [f"{PREFIX}.experts.0.gate_proj.{suffix}" for suffix in
                         ("weight_global_scale", "input_global_scale", "weight_packed", "weight_scale")]
    assert not cache._layers and not ctx.used
    raw = float(ctx.payload[PREFIX + ".experts.0.gate_proj.input_global_scale"][0])
    assert struct.unpack("I", struct.pack("f", raw))[0] == 0x3e63ffff


def test_cpu_packer_refuses_non_cpu_payload_before_use():
    codes = torch.zeros((64, 32), dtype=torch.uint8)
    scales = torch.empty((64, 4), dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(ValueError, match="requires CPU"):
        pack_projection_a4(codes, scales)
