"""Host argument/scratch contracts; explicitly no native arithmetic substitute."""
import pytest

pytestmark = pytest.mark.torch

from tensorfold.cuda.sm70.gguf_ops import owned_workspace_bytes


def test_owned_scratch_counts_routes_and_fp16_activations_without_output_double_count():
    assert owned_workspace_bytes(33, 5) == 33 * 5 * (8 + 320)
    assert owned_workspace_bytes(1, 65535) == 65535 * 328


@pytest.mark.parametrize("rows,slots,width", [(0, 5, 160), (True, 5, 160), (1, 0, 160),
                                             (1, 65536, 160), (1, 5, 80), (1, 5, 320), (1, 5, 160.0)])
def test_invalid_or_non_etp4_workspace_refuses(rows, slots, width):
    with pytest.raises(ValueError):
        owned_workspace_bytes(rows, slots, width)


@pytest.mark.parametrize("output_dtype", ["float16", "float32"])
def test_owned_wrapper_preserves_global_inventory_and_passes_caller_scratch(monkeypatch, output_dtype):
    from types import SimpleNamespace
    import torch
    from tensorfold.cuda.sm70 import gguf_ops

    calls = []
    monkeypatch.setattr(gguf_ops, "extension", lambda: SimpleNamespace(experts_owned=lambda *args: calls.append(args)))
    x = torch.empty((3, 256), dtype=torch.float16)
    picks = torch.tensor([[0, 255, 256, 511, 512]] * 3)
    bank = SimpleNamespace(data=object(), ggml_type=13, column_offset=0)
    shared = torch.empty_like(x, dtype=torch.float32)
    local = torch.empty_like(picks)
    activation = torch.empty((15, 160), dtype=torch.float16)
    out = torch.empty((3, 5, 256), dtype=getattr(torch, output_dtype))
    assert gguf_ops.experts_owned(x, picks, bank, bank, bank, first=256, experts=512, width=160,
        shared=shared, local=local, activation=activation, out=out) is out
    assert len(calls) == 1
    args = calls[0]
    assert args[0] is x and args[1] is picks
    assert args[11:14] == (160, 256, 512)
    assert all(a is b for a, b in zip(args[14:], (shared, local, activation, out)))
    with pytest.raises(ValueError, match="row matrices"):
        gguf_ops.experts_owned(x, picks[0], bank, bank, bank, first=256, experts=512, width=160,
            shared=shared, local=local, activation=activation, out=out)
    with pytest.raises(ValueError, match="native integer"):
        gguf_ops.experts_owned(x, picks, bank, bank, bank, first=True, experts=512, width=160,
            shared=shared, local=local, activation=activation, out=out)
    assert len(calls) == 1
