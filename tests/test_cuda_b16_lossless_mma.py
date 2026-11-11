"""CPU boundary/JIT specialization contract; GPU operand/math proof is separate."""
import pytest
import torch

from tensorfold.cuda import b16_lossless_mma as mma
from tensorfold.cuda.b16_lossless import pack


def test_import_and_cpu_boundary_never_initialize_cuda():
    assert not torch.cuda.is_initialized()
    encoded = pack(torch.zeros(1, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='expected device'):
        mma.unpack(encoded)
    with pytest.raises(ValueError, match='expected device'):
        mma.matmul_rows(torch.zeros(1, 64, dtype=torch.bfloat16), encoded)
    assert not torch.cuda.is_initialized()


def test_runtime_row_and_pool_count_are_not_jit_specialization_inputs():
    # Actual pinned Triton parameter metadata, not an imitation of its compiler.
    by_name = {parameter.name: parameter for parameter in mma._matmul.params}
    assert by_name['M'].do_not_specialize and not by_name['M'].is_constexpr
    assert by_name['RAW_BLOCKS'].do_not_specialize and not by_name['RAW_BLOCKS'].is_constexpr
    assert by_name['N'].is_constexpr and by_name['K'].is_constexpr


def test_decoder_pool_count_uses_same_runtime_boundary():
    by_name = {parameter.name: parameter for parameter in mma._decode.params}
    assert by_name['RAW_BLOCKS'].do_not_specialize
    assert not torch.cuda.is_initialized()
