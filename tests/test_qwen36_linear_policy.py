"""Preparation and dispatch of the Qwen3.6 BF16 projections without a GPU kernel."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen3_5.cuda import b16
from tensorfold.families.qwen3_5.cuda.nvfp4_load import Plain8
from tensorfold.families.qwen3_5.cuda.weights import Plain
from tensorfold.families.qwen3_5_moe.cuda import linear


class FakeWeight:
    """Shape/dtype/device metadata of a CUDA weight; row slices keep metadata (no storage is touched)."""

    def __init__(self, shape, dtype=torch.bfloat16, device="cuda", contiguous=True, ptr=0):
        self.shape, self.dtype, self.device, self.ndim = tuple(shape), dtype, NS(type=device), len(shape)
        self._contiguous, self._ptr = contiguous, ptr

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self.shape[0] * self.shape[1]

    def element_size(self):
        return 2


def test_every_row_count_takes_the_prompt_arithmetic(monkeypatch):
    calls = []
    monkeypatch.setattr(linear, "isinstance", lambda value, kind: True, raising=False)
    monkeypatch.setattr(b16, "prompt", lambda x, w: calls.append(("prompt", x, w)) or "prompt")
    monkeypatch.setattr(b16, "rows16", lambda x, w: calls.append(("rows16", x, w)) or "rows16")
    monkeypatch.setattr(b16, "matmul", lambda *args: pytest.fail("decode rows must not take the per-row FMA kernel"))
    weight = FakeWeight((4096, 2048))
    projection = linear.MmaLinear(weight)
    assert (projection.n, projection.k, projection.nbytes()) == (4096, 2048, 4096 * 2048 * 2)
    for rows in (1, 2, 3, 5, 8, 15, 16, 17, 64, 128, 4096):
        x = torch.zeros(rows, 2048, dtype=torch.bfloat16)
        kernel = "rows16" if rows <= 16 else "prompt"
        assert projection(x) == kernel and calls[-1] == (kernel, x, weight)
        assert projection.prefill(x) == "prompt" and calls[-1] == ("prompt", x, weight)
    assert not isinstance(projection, Plain)


@pytest.mark.parametrize("options", [{"dtype": torch.float16}, {"device": "cpu"}, {"contiguous": False},
                                     {"shape": (64, 2050)}, {"shape": (64, 32)}, {"shape": (0, 2048)},
                                     {"ptr": 8}])
def test_unsupported_storage_is_refused_at_preparation(monkeypatch, options):
    monkeypatch.setattr(linear, "isinstance", lambda value, kind: True, raising=False)
    with pytest.raises(ValueError, match="MMA projections"):
        linear.MmaLinear(FakeWeight(**{"shape": (64, 2048), **options}))


def test_prepare_binds_bf16_gdn_and_head_and_stacks_the_gates(monkeypatch):
    monkeypatch.setattr(linear, "MmaLinear", lambda w: ("mma", w))
    g = torch.Generator().manual_seed(0)

    def plain(n, kind=Plain, **extra):
        return kind(torch.randn(n, 64, generator=g).to(torch.bfloat16), **extra)

    def gdn_of(z):
        return NS(qkv=plain(96), z=z, b=plain(8), a=plain(8), out=plain(64), zba=None)

    gdn, fp8, other = gdn_of(plain(64)), gdn_of(plain(64, Plain8, rows8=object())), gdn_of(plain(64))
    other.qkv = object()
    head = plain(256)
    stored = {name: getattr(gdn, name).weight for name in ("qkv", "z", "b", "a", "out")}
    kept = [(layer, dict(vars(layer))) for layer in (fp8, other)]
    weights = NS(layers=[NS(gdn=gdn), NS(gdn=None), NS(gdn=fp8), NS(gdn=other)], head=head)
    linear.prepare(weights)
    assert gdn.qkv[1] is stored["qkv"] and gdn.out[1] is stored["out"]
    stacked = gdn.zba[1]                                  # one storage; z, b and a are its row views
    assert torch.equal(stacked, torch.cat([stored["z"], stored["b"], stored["a"]])) and stacked.is_contiguous()
    assert [p[1].shape[0] for p in (gdn.z, gdn.b, gdn.a)] == [64, 8, 8]
    assert [p[1].data_ptr() for p in (gdn.z, gdn.b, gdn.a)] == [stacked[r].data_ptr() for r in (0, 64, 72)]
    for layer, fields in kept:                           # an FP8 prompt copy or another format: kept as stored
        assert all(getattr(layer, name) is value for name, value in fields.items())
    assert weights.head[1] is head.weight


def test_non_bf16_head_keeps_its_object():
    head = object()
    weights = NS(layers=[], head=head)
    linear.prepare(weights)
    assert weights.head is head
