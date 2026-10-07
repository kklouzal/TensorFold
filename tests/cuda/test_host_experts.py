"""Native CUDA pack is the independent bitwise oracle for CPU expert spill."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# Keep collection safe when Torch is absent or CUDA is unavailable.
from tensorfold.cuda import experts  # noqa: E402
from tensorfold.cuda.host_experts import load_host_experts, pack_cpu  # noqa: E402


def inputs(e, n, k, gs, seed):
    rng = np.random.default_rng(seed)
    words = torch.from_numpy(rng.integers(0, 2**32, (e, n, k // 8), dtype=np.uint32).view(np.int32))
    scales, biases = [
        torch.from_numpy(rng.integers(0, 2**16, (e, n, k // gs), dtype=np.uint16).view(np.int16)).view(torch.bfloat16)
        for _ in range(2)
    ]
    return words, scales, biases


@pytest.mark.parametrize("gs", [32, 64])
@pytest.mark.parametrize("shape", [(1, 32, 64), (5, 64, 128)])
def test_cpu_pack_is_bitwise_native_cuda_pack(gs, shape):
    source = inputs(*shape, gs, 73)
    actual = pack_cpu(*source, gs)
    reference = experts.pack(*(t.cuda() for t in source), gs).cpu()
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("chunk_experts", [1, 2])
def test_host_load_matches_native_grouped_make_with_shared_expert(chunk_experts):
    base, tensors = "mtp.layers.0.mlp", {}
    for p, (n, k) in zip(("gate_proj", "up_proj", "down_proj"), ((32, 64), (32, 64), (64, 32))):
        for kind, e in (("switch_mlp", 3), ("shared_expert", 1)):
            for field, tensor in zip(("weight", "scales", "biases"), inputs(e, n, k, 32, 91 + e)):
                tensors[f"{base}.{kind}.{p}.{field}"] = tensor if e == 3 else tensor[0]

    class Reader:
        def info(self, name):
            t = tensors[name]
            return {"shape": tuple(t.shape), "dtype": "U32" if t.dtype == torch.int32 else "BF16"}

        def get_rows(self, name, lo, hi):
            return tensors[name][lo:hi].clone()

    host = load_host_experts(Reader(), base, chunk_experts=chunk_experts)
    native_inputs = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        native_inputs.append(
            tuple(
                torch.cat(
                    (
                        tensors[f"{base}.switch_mlp.{projection}.{part}"],
                        tensors[f"{base}.shared_expert.{projection}.{part}"][None],
                    )
                ).cuda()
                for part in ("weight", "scales", "biases")
            )
        )
    native = experts.make(native_inputs[:2], native_inputs[2], 32)
    assert torch.equal(host.up, native.up.cpu())
    assert torch.equal(host.down, native.down.cpu())
    assert (host.width, host.dims, host.gs, host.count, host.swiglu, host.limit) == (
        native.width,
        native.dims,
        native.gs,
        native.count,
        native.swiglu,
        native.limit,
    )
