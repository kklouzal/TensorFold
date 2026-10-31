"""Process-global native precision ownership and primary-preserving retirement."""
from types import SimpleNamespace
import pytest

from tensorfold.families.gemma4.cuda.e2b_policy import E2BBlasPolicy


class Matmul:
    def __init__(self):
        self.reduced, self.split = False, False

    @property
    def allow_bf16_reduced_precision_reduction(self):
        return self.reduced

    @allow_bf16_reduced_precision_reduction.setter
    def allow_bf16_reduced_precision_reduction(self, value):
        self.reduced, self.split = value

    @property
    def allow_bf16_reduced_precision_reduction_split_k(self):
        return self.split


def backend():
    owner = SimpleNamespace(matmul=Matmul(), library="original", fail_restore=False)

    def preferred(value=None):
        if value is not None:
            if owner.fail_restore and value == "original":
                raise RuntimeError("preference restore failed")
            owner.library = value
        return owner.library

    owner.preferred_blas_library = preferred
    return owner


def test_lease_restores_independently_captured_split_k_and_rejects_overlap():
    cuda = backend()
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=cuda))
    lease = E2BBlasPolicy(torch)
    try:
        assert (cuda.library, cuda.matmul.reduced, cuda.matmul.split) == ("cublaslt", False, False)
        with pytest.raises(RuntimeError, match="active engine"):
            E2BBlasPolicy(torch)
    finally:
        lease.close()
    lease.close()
    assert (cuda.library, cuda.matmul.reduced, cuda.matmul.split) == ("original", False, False)


def test_failed_restore_retains_guard_and_attempts_all_settings_until_retry():
    cuda = backend()
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=cuda))
    lease = E2BBlasPolicy(torch)
    try:
        cuda.fail_restore = True
        with pytest.raises(RuntimeError, match="restore"):
            lease.close()
        assert (cuda.matmul.reduced, cuda.matmul.split) == (False, False)
        with pytest.raises(RuntimeError, match="active engine"):
            E2BBlasPolicy(torch)
    finally:
        cuda.fail_restore = False
        lease.close()
